/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

// SVE single-precision matrix multiply C = A(m x k) * B(k x n), all row-major.
// Tier-2 (SVE) implementation, compiled with -march=armv8.4-a+sve+bf16.
//
// Mirrors the amd64 AVX-512 kernel's strategy, adapted to SVE:
//   - Register-blocked 5-row x 4-SVE-vector micro-tile: 20 C accumulators stay resident across
//     the whole k loop (C loaded/stored once per tile, not per k-step). 5x4 was the empirical
//     sweet spot on Neoverse-V2 (128-bit SVE, 32 z-registers): larger tiles spill.
//   - B is packed into a contiguous (k x 4*VL) panel once per column strip so every row-tile
//     re-reads it sequentially from cache.
//   - Inner k-step: load 4 B vectors + broadcast 5 A values => 9 loads for 20 FMAs (2.2 FMA/load).
//   - Predicated stores (svwhilelt) give clean column tails with no separate tail kernel.
// SVE vector types are "sizeless" (cannot be arrayed), so the tile rows/cols are unrolled.
// VL (floats per SVE vector) is read at runtime via svcntw().

#include <stddef.h>
#include <stdint.h>
#include <cstdlib>
#include <arm_sve.h>
#include "vec.h"
#include "vec_common.h"

namespace {

constexpr int32_t MR = 5;   // rows per tile
constexpr int32_t NC = 4;   // SVE vectors (columns) per tile

static inline void pack_b_panel(const f32_t* B, int32_t n, int32_t k, int32_t j0, int32_t width,
                                int32_t panelW, f32_t* panel) {
    for (int32_t p = 0; p < k; ++p) {
        f32_t* dst = panel + (size_t)p * panelW;
        const f32_t* src = B + (size_t)p * n + j0;
        int32_t c = 0;
        for (; c < width; ++c) dst[c] = src[c];
        for (; c < panelW; ++c) dst[c] = 0.0f;
    }
}

// 5x4 micro-kernel: 20 SVE accumulators, resident across the k loop.
static inline void micro_kernel_5(const f32_t* A, int32_t ars, const f32_t* pB, f32_t* C,
                                  int32_t n, int32_t i0, int32_t j0, int32_t k, int32_t width,
                                  int32_t VL, int32_t pw) {
    const svbool_t pt = svptrue_b32();
    svfloat32_t c0_0 = svdup_n_f32(0), c0_1 = svdup_n_f32(0), c0_2 = svdup_n_f32(0), c0_3 = svdup_n_f32(0);
    svfloat32_t c1_0 = svdup_n_f32(0), c1_1 = svdup_n_f32(0), c1_2 = svdup_n_f32(0), c1_3 = svdup_n_f32(0);
    svfloat32_t c2_0 = svdup_n_f32(0), c2_1 = svdup_n_f32(0), c2_2 = svdup_n_f32(0), c2_3 = svdup_n_f32(0);
    svfloat32_t c3_0 = svdup_n_f32(0), c3_1 = svdup_n_f32(0), c3_2 = svdup_n_f32(0), c3_3 = svdup_n_f32(0);
    svfloat32_t c4_0 = svdup_n_f32(0), c4_1 = svdup_n_f32(0), c4_2 = svdup_n_f32(0), c4_3 = svdup_n_f32(0);

    const f32_t* a0 = A + (size_t)(i0 + 0) * ars;
    const f32_t* a1 = A + (size_t)(i0 + 1) * ars;
    const f32_t* a2 = A + (size_t)(i0 + 2) * ars;
    const f32_t* a3 = A + (size_t)(i0 + 3) * ars;
    const f32_t* a4 = A + (size_t)(i0 + 4) * ars;

    for (int32_t p = 0; p < k; ++p) {
        const f32_t* bp = pB + (size_t)p * pw;
        svfloat32_t b0 = svld1_f32(pt, bp);
        svfloat32_t b1 = svld1_f32(pt, bp + VL);
        svfloat32_t b2 = svld1_f32(pt, bp + 2 * VL);
        svfloat32_t b3 = svld1_f32(pt, bp + 3 * VL);
        svfloat32_t av;
        av = svdup_n_f32(a0[p]); c0_0 = svmla_f32_x(pt, c0_0, av, b0); c0_1 = svmla_f32_x(pt, c0_1, av, b1); c0_2 = svmla_f32_x(pt, c0_2, av, b2); c0_3 = svmla_f32_x(pt, c0_3, av, b3);
        av = svdup_n_f32(a1[p]); c1_0 = svmla_f32_x(pt, c1_0, av, b0); c1_1 = svmla_f32_x(pt, c1_1, av, b1); c1_2 = svmla_f32_x(pt, c1_2, av, b2); c1_3 = svmla_f32_x(pt, c1_3, av, b3);
        av = svdup_n_f32(a2[p]); c2_0 = svmla_f32_x(pt, c2_0, av, b0); c2_1 = svmla_f32_x(pt, c2_1, av, b1); c2_2 = svmla_f32_x(pt, c2_2, av, b2); c2_3 = svmla_f32_x(pt, c2_3, av, b3);
        av = svdup_n_f32(a3[p]); c3_0 = svmla_f32_x(pt, c3_0, av, b0); c3_1 = svmla_f32_x(pt, c3_1, av, b1); c3_2 = svmla_f32_x(pt, c3_2, av, b2); c3_3 = svmla_f32_x(pt, c3_3, av, b3);
        av = svdup_n_f32(a4[p]); c4_0 = svmla_f32_x(pt, c4_0, av, b0); c4_1 = svmla_f32_x(pt, c4_1, av, b1); c4_2 = svmla_f32_x(pt, c4_2, av, b2); c4_3 = svmla_f32_x(pt, c4_3, av, b3);
    }

    int32_t rem = width;
    svbool_t q0 = rem > 0 ? svwhilelt_b32(0, rem < VL ? rem : VL) : svpfalse(); rem -= VL;
    svbool_t q1 = rem > 0 ? svwhilelt_b32(0, rem < VL ? rem : VL) : svpfalse(); rem -= VL;
    svbool_t q2 = rem > 0 ? svwhilelt_b32(0, rem < VL ? rem : VL) : svpfalse(); rem -= VL;
    svbool_t q3 = rem > 0 ? svwhilelt_b32(0, rem < VL ? rem : VL) : svpfalse();
    f32_t* cr;
    cr = C + (size_t)(i0 + 0) * n + j0; svst1_f32(q0, cr, c0_0); svst1_f32(q1, cr + VL, c0_1); svst1_f32(q2, cr + 2 * VL, c0_2); svst1_f32(q3, cr + 3 * VL, c0_3);
    cr = C + (size_t)(i0 + 1) * n + j0; svst1_f32(q0, cr, c1_0); svst1_f32(q1, cr + VL, c1_1); svst1_f32(q2, cr + 2 * VL, c1_2); svst1_f32(q3, cr + 3 * VL, c1_3);
    cr = C + (size_t)(i0 + 2) * n + j0; svst1_f32(q0, cr, c2_0); svst1_f32(q1, cr + VL, c2_1); svst1_f32(q2, cr + 2 * VL, c2_2); svst1_f32(q3, cr + 3 * VL, c2_3);
    cr = C + (size_t)(i0 + 3) * n + j0; svst1_f32(q0, cr, c3_0); svst1_f32(q1, cr + VL, c3_1); svst1_f32(q2, cr + 2 * VL, c3_2); svst1_f32(q3, cr + 3 * VL, c3_3);
    cr = C + (size_t)(i0 + 4) * n + j0; svst1_f32(q0, cr, c4_0); svst1_f32(q1, cr + VL, c4_1); svst1_f32(q2, cr + 2 * VL, c4_2); svst1_f32(q3, cr + 3 * VL, c4_3);
}

// Single-row kernel for the m tail (m not a multiple of MR).
static inline void micro_kernel_1(const f32_t* A, int32_t ars, const f32_t* pB, f32_t* C,
                                  int32_t n, int32_t i, int32_t j0, int32_t k, int32_t width,
                                  int32_t VL, int32_t pw) {
    const svbool_t pt = svptrue_b32();
    svfloat32_t c0 = svdup_n_f32(0), c1 = svdup_n_f32(0), c2 = svdup_n_f32(0), c3 = svdup_n_f32(0);
    const f32_t* ar = A + (size_t)i * ars;
    for (int32_t p = 0; p < k; ++p) {
        const f32_t* bp = pB + (size_t)p * pw;
        svfloat32_t av = svdup_n_f32(ar[p]);
        c0 = svmla_f32_x(pt, c0, av, svld1_f32(pt, bp));
        c1 = svmla_f32_x(pt, c1, av, svld1_f32(pt, bp + VL));
        c2 = svmla_f32_x(pt, c2, av, svld1_f32(pt, bp + 2 * VL));
        c3 = svmla_f32_x(pt, c3, av, svld1_f32(pt, bp + 3 * VL));
    }
    int32_t rem = width;
    f32_t* cr = C + (size_t)i * n + j0;
    svst1_f32(svwhilelt_b32(0, rem < VL ? rem : VL), cr, c0); rem -= VL;
    if (rem > 0) svst1_f32(svwhilelt_b32(0, rem < VL ? rem : VL), cr + VL, c1); rem -= VL;
    if (rem > 0) svst1_f32(svwhilelt_b32(0, rem < VL ? rem : VL), cr + 2 * VL, c2); rem -= VL;
    if (rem > 0) svst1_f32(svwhilelt_b32(0, rem < VL ? rem : VL), cr + 3 * VL, c3);
}

} // namespace

EXPORT void vec_sgemmf32_2(const f32_t* a, const f32_t* b, f32_t* c,
                           const int32_t m, const int32_t k, const int32_t n) {
    const int32_t VL = (int32_t)svcntw();   // floats per SVE vector (runtime)
    const int32_t pw = NC * VL;             // columns per packed panel

    f32_t* panel = (f32_t*)aligned_alloc(64, (size_t)k * pw * sizeof(f32_t));

    for (int32_t j0 = 0; j0 < n; j0 += pw) {
        const int32_t width = (j0 + pw <= n) ? pw : (n - j0);
        pack_b_panel(b, n, k, j0, width, pw, panel);

        int32_t i0 = 0;
        for (; i0 + MR <= m; i0 += MR) {
            micro_kernel_5(a, k, panel, c, n, i0, j0, k, width, VL, pw);
        }
        for (; i0 < m; ++i0) {
            micro_kernel_1(a, k, panel, c, n, i0, j0, k, width, VL, pw);
        }
    }

    free(panel);
}
