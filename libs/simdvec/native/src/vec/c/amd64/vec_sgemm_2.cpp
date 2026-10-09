/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

 // Single-precision matrix multiply C = A(m x k) * B(k x n), all row-major.
 // Tier-2 (AVX-512 / icelake) implementation. Compiled with -march=icelake-client
 // via the *_2.cpp Makefile rule.
 //
 // Design goal: maximise FMA-per-load (arithmetic intensity) to relieve the
 // load/store-bound behaviour of a naive kernel. Uses a register-blocked
 // micro-kernel with a packed B panel so B stays L2/L1-resident.
 //
 // Micro-kernel tile: MR rows x 2 ZMM columns (2*16 = 32 output cols).
 //   - C accumulator: MR x 2 ZMM registers, kept resident across the whole k loop
 //     (C is loaded/stored once per tile, never per k-step).
 //   - Inner k-step: load 2 B vectors (32 cols of one B row) + broadcast MR A values
 //     => (MR+2) loads for (MR*2) FMAs. At MR=14: 16 loads -> 28 FMAs = 1.75 FMA/load,
 //     vs ~1.33 for the 4x2 Panama kernel.
 // B is packed into a contiguous (k x 32) panel once per column strip so every
 // row-tile re-reads it sequentially from cache.

#include <stddef.h>
#include <stdint.h>
#include <cstring>
#include <immintrin.h>
#include "vec.h"
#include "vec_common.h"

namespace {

constexpr int32_t VL = 16;          // floats per ZMM
constexpr int32_t NR = 2 * VL;      // 32 output columns per panel (2 ZMM wide)
constexpr int32_t MR = 14;          // rows per micro-tile (14*2 = 28 C accumulators)

// Pack a (k x width) column strip of B (row-major, leading dim n) starting at column j0
// into a contiguous (k x NR) buffer, zero-padding columns beyond `width`.
static inline void pack_b_panel(const f32_t* B, int32_t n, int32_t k, int32_t j0, int32_t width, f32_t* panel) {
    if (width == NR) {
        for (int32_t p = 0; p < k; ++p) {
            std::memcpy(panel + (size_t)p * NR, B + (size_t)p * n + j0, NR * sizeof(f32_t));
        }
    } else {
        for (int32_t p = 0; p < k; ++p) {
            f32_t* dst = panel + (size_t)p * NR;
            const f32_t* src = B + (size_t)p * n + j0;
            int32_t c = 0;
            for (; c < width; ++c) dst[c] = src[c];
            for (; c < NR; ++c) dst[c] = 0.0f;
        }
    }
}

// Micro-kernel: computes C[i0..i0+mr, j0..j0+NR) += A[i0.., 0..k) * packed_B(k x NR).
// mr <= MR. Writes `width` valid columns back to C (masked store for the tail panel).
static inline void micro_kernel(
    const f32_t* A, int32_t aRowStride,
    const f32_t* packedB, f32_t* C, int32_t n,
    int32_t i0, int32_t j0, int32_t mr, int32_t k, int32_t width
) {
    __m512 c0[MR];
    __m512 c1[MR];
    for (int32_t r = 0; r < mr; ++r) {
        c0[r] = _mm512_setzero_ps();
        c1[r] = _mm512_setzero_ps();
    }

    for (int32_t p = 0; p < k; ++p) {
        const f32_t* bp = packedB + (size_t)p * NR;
        __m512 b0 = _mm512_load_ps(bp);        // packed panel is 64-byte aligned
        __m512 b1 = _mm512_load_ps(bp + VL);
        const f32_t* ap = A + (size_t)i0 * aRowStride + p;
        // Broadcast A[i0+r, p] for each of the mr rows and FMA into the C tile.
        for (int32_t r = 0; r < mr; ++r) {
            __m512 a = _mm512_set1_ps(ap[(size_t)r * aRowStride]);
            c0[r] = _mm512_fmadd_ps(a, b0, c0[r]);
            c1[r] = _mm512_fmadd_ps(a, b1, c1[r]);
        }
    }

    if (width == NR) {
        for (int32_t r = 0; r < mr; ++r) {
            f32_t* cr = C + (size_t)(i0 + r) * n + j0;
            _mm512_storeu_ps(cr, c0[r]);
            _mm512_storeu_ps(cr + VL, c1[r]);
        }
    } else {
        // Tail panel: mask the stores to `width` columns.
        int32_t w0 = width < VL ? width : VL;
        int32_t w1 = width - w0;
        __mmask16 m0 = (w0 >= VL) ? (__mmask16)0xFFFF : (__mmask16)((1u << w0) - 1);
        __mmask16 m1 = (w1 <= 0) ? (__mmask16)0 : ((w1 >= VL) ? (__mmask16)0xFFFF : (__mmask16)((1u << w1) - 1));
        for (int32_t r = 0; r < mr; ++r) {
            f32_t* cr = C + (size_t)(i0 + r) * n + j0;
            _mm512_mask_storeu_ps(cr, m0, c0[r]);
            if (m1) _mm512_mask_storeu_ps(cr + VL, m1, c1[r]);
        }
    }
}

} // namespace

EXPORT void vec_sgemmf32_2(
    const f32_t* a,
    const f32_t* b,
    f32_t* c,
    const int32_t m,
    const int32_t k,
    const int32_t n
) {
    // One packed B panel (k x NR), 64-byte aligned, reused across all row-tiles of a column strip.
    const size_t panel_floats = (size_t)k * NR;
    f32_t* panel = static_cast<f32_t*>(aligned_alloc(64, panel_floats * sizeof(f32_t)));

    for (int32_t j0 = 0; j0 < n; j0 += NR) {
        const int32_t width = (j0 + NR <= n) ? NR : (n - j0);
        pack_b_panel(b, n, k, j0, width, panel);

        int32_t i0 = 0;
        for (; i0 + MR <= m; i0 += MR) {
            micro_kernel(a, k, panel, c, n, i0, j0, MR, k, width);
        }
        if (i0 < m) {
            micro_kernel(a, k, panel, c, n, i0, j0, m - i0, k, width);
        }
    }

    free(panel);
}
