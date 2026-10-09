/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

 // Tier-1 (baseline / AVX2) single-precision matrix multiply C = A(m x k) * B(k x n),
 // all row-major. Correct, portable fallback used when the tier-2 (AVX-512) kernel is
 // unavailable. Row-streaming accumulation: for each output row, accumulate B rows scaled
 // by A[i,p], so B is read sequentially.

#include <stddef.h>
#include <stdint.h>
#include <cstring>
#include "vec.h"
#include "vec_common.h"

EXPORT void vec_sgemmf32(
    const f32_t* a,
    const f32_t* b,
    f32_t* c,
    const int32_t m,
    const int32_t k,
    const int32_t n
) {
    for (int32_t i = 0; i < m; ++i) {
        f32_t* crow = c + (size_t)i * n;
        std::memset(crow, 0, (size_t)n * sizeof(f32_t));
        const f32_t* arow = a + (size_t)i * k;
        for (int32_t p = 0; p < k; ++p) {
            const f32_t aip = arow[p];
            const f32_t* brow = b + (size_t)p * n;
            for (int32_t j = 0; j < n; ++j) {
                crow[j] += aip * brow[j];
            }
        }
    }
}
