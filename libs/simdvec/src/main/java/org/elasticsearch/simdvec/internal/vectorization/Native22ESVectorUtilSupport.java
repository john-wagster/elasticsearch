/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec.internal.vectorization;

import org.apache.lucene.util.BytesRef;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.simdvec.MultiBFloat16VectorsSource;
import org.elasticsearch.simdvec.MultiByteVectorsSource;
import org.elasticsearch.simdvec.MultiFloatVectorsSource;
import org.elasticsearch.simdvec.MultiVectorsSource;
import org.elasticsearch.simdvec.SimdVecLibrary;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public final class Native22ESVectorUtilSupport extends PanamaESVectorUtilSupport {

    private static final SimdVecLibrary DISTANCE_FUNCS = SimdVecLibrary.instance().orElseThrow(AssertionError::new);

    // Optional thread pool to parallelise the native GEMM across output row-bands. The native
    // kernel computes disjoint C rows from disjoint A rows (sharing B read-only), so splitting by
    // rows is embarrassingly parallel with no coordination. Driven by -Dbench.matmul.threads=N
    // (N<=1 or unset => single-threaded). Benchmark scope; a production version would use an
    // ES-managed executor rather than a private pool.
    private static final int MM_THREADS = Integer.getInteger("bench.matmul.threads", 1);
    private static final int MM_MIN_PARALLEL_ROWS = 64;
    private static final ExecutorService MM_POOL = MM_THREADS > 1 ? Executors.newFixedThreadPool(MM_THREADS) : null;

    /*
     * This is technically separate to the Panama22 implementation, but there's
     * only a few things which are native-only, it's easiest to inherit implementations
     * and override the ones we want, rather than delegating all methods in the interface.
     */

    @Override
    public float dotProduct(float[] a, float[] b) {
        return DISTANCE_FUNCS.dotProductF32(MemorySegment.ofArray(a), MemorySegment.ofArray(b), a.length);
    }

    @Override
    public void matrixMultiply(float[] a, float[] b, int m, int k, int n, float[] result) {
        // Native single-precision GEMM: C(m x n) = A(m x k) * B(k x n), all row-major.
        // Heap arrays are passed zero-copy via MemorySegment.ofArray; the @Critical binding
        // pins them for the duration of the call.
        if (MM_POOL == null || m < MM_MIN_PARALLEL_ROWS) {
            DISTANCE_FUNCS.sgemmF32(
                MemorySegment.ofArray(a),
                MemorySegment.ofArray(b),
                MemorySegment.ofArray(result),
                m,
                k,
                n
            );
            return;
        }
        // Split the output rows into bands, one task per band. Bands are disjoint and B is shared.
        int bands = Math.min(m, MM_THREADS);
        int rowsPerBand = (m + bands - 1) / bands;
        MemorySegment as = MemorySegment.ofArray(a);
        MemorySegment bs = MemorySegment.ofArray(b);
        MemorySegment cs = MemorySegment.ofArray(result);
        List<Future<?>> futures = new ArrayList<>(bands);
        for (int r0 = 0; r0 < m; r0 += rowsPerBand) {
            final int rowStart = r0;
            final int mb = Math.min(rowsPerBand, m - r0);
            futures.add(MM_POOL.submit(() -> {
                MemorySegment aBand = as.asSlice((long) rowStart * k * Float.BYTES, (long) mb * k * Float.BYTES);
                MemorySegment cBand = cs.asSlice((long) rowStart * n * Float.BYTES, (long) mb * n * Float.BYTES);
                DISTANCE_FUNCS.sgemmF32(aBand, bs, cBand, mb, k, n);
            }));
        }
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public float dotProduct(float[] a, int aOffset, float[] b, int bOffset, int length) {
        return DISTANCE_FUNCS.dotProductF32(
            MemorySegment.ofArray(a).asSlice((long) aOffset * Float.BYTES, (long) length * Float.BYTES),
            MemorySegment.ofArray(b).asSlice((long) bOffset * Float.BYTES, (long) length * Float.BYTES),
            length
        );
    }

    @Override
    public float squareDistance(float[] a, float[] b) {
        return DISTANCE_FUNCS.squareDistanceF32(MemorySegment.ofArray(a), MemorySegment.ofArray(b), a.length);
    }

    @Override
    public float squareDistance(float[] a, float[] b, int offset, int length) {
        return DISTANCE_FUNCS.squareDistanceF32(
            MemorySegment.ofArray(a).asSlice((long) offset * Float.BYTES, (long) length * Float.BYTES),
            MemorySegment.ofArray(b).asSlice((long) offset * Float.BYTES, (long) length * Float.BYTES),
            length
        );
    }

    @Override
    public float cosine(byte[] a, byte[] b) {
        return DISTANCE_FUNCS.cosineI8(MemorySegment.ofArray(a), MemorySegment.ofArray(b), a.length);
    }

    @Override
    public float dotProduct(byte[] a, byte[] b) {
        return DISTANCE_FUNCS.dotProductI8(MemorySegment.ofArray(a), MemorySegment.ofArray(b), a.length);
    }

    @Override
    public float dotProduct(byte[] a, byte[] b, int offset, int length) {
        return DISTANCE_FUNCS.dotProductI8(
            MemorySegment.ofArray(a).asSlice(offset, length),
            MemorySegment.ofArray(b).asSlice(offset, length),
            length
        );
    }

    @Override
    public float squareDistance(byte[] a, byte[] b) {
        return DISTANCE_FUNCS.squareDistanceI8(MemorySegment.ofArray(a), MemorySegment.ofArray(b), a.length);
    }

    @Override
    public float squareDistance(byte[] a, byte[] b, int offset, int length) {
        return DISTANCE_FUNCS.squareDistanceI8(
            MemorySegment.ofArray(a).asSlice(offset, length),
            MemorySegment.ofArray(b).asSlice(offset, length),
            length
        );
    }

    @Override
    public float maxSimDotProduct(MultiFloatVectorsSource source, float[][] query, float[] scoresScratch) {
        if (canUseF32BulkPath(source)) {
            final BytesRef vectors = source.vectorBytes();
            final MemorySegment vectorsSegment = MemorySegment.ofArray(vectors.bytes)
                .asSlice(vectors.offset, (long) source.vectorByteSize() * source.vectorCount());
            final MemorySegment scoresSegment = MemorySegment.ofArray(scoresScratch);
            float sum = 0f;
            for (float[] floats : query) {
                DISTANCE_FUNCS.dotProductF32Bulk(
                    vectorsSegment,
                    MemorySegment.ofArray(floats),
                    source.vectorDims(),
                    source.vectorCount(),
                    scoresSegment
                );
                sum += ESVectorUtil.max(scoresScratch, source.vectorCount());
            }
            return sum;
        }
        return super.maxSimDotProduct(source, query, scoresScratch);
    }

    @Override
    public float maxSimDotProduct(MultiBFloat16VectorsSource source, float[][] query, float[] scoresScratch) {
        if (canUseBFloat16Path(source)) {
            final BytesRef vectors = source.vectorBytes();
            final MemorySegment vectorsSegment = MemorySegment.ofArray(vectors.bytes)
                .asSlice(vectors.offset, (long) source.vectorByteSize() * source.vectorCount());
            final MemorySegment scoresSegment = MemorySegment.ofArray(scoresScratch);
            float sum = 0f;
            for (float[] floats : query) {
                DISTANCE_FUNCS.dotProductDBF16QF32Bulk(
                    vectorsSegment,
                    MemorySegment.ofArray(floats),
                    source.vectorDims(),
                    source.vectorCount(),
                    scoresSegment
                );
                sum += ESVectorUtil.max(scoresScratch, source.vectorCount());
            }
            return sum;
        }
        return super.maxSimDotProduct(source, query, scoresScratch);
    }

    @Override
    public float maxSimDotProduct(MultiByteVectorsSource source, byte[][] query, float[] scoresScratch) {
        if (canUseI8BulkPath(source)) {
            final BytesRef vectors = source.vectorBytes();
            final MemorySegment vectorsSegment = MemorySegment.ofArray(vectors.bytes)
                .asSlice(vectors.offset, (long) source.vectorByteSize() * source.vectorCount());
            final MemorySegment scoresSegment = MemorySegment.ofArray(scoresScratch);
            float sum = 0f;
            for (byte[] bytes : query) {
                DISTANCE_FUNCS.dotProductI8Bulk(
                    vectorsSegment,
                    MemorySegment.ofArray(bytes),
                    source.vectorDims(),
                    source.vectorCount(),
                    scoresSegment
                );
                sum += ESVectorUtil.max(scoresScratch, source.vectorCount());
            }
            return sum;
        }
        return super.maxSimDotProduct(source, query, scoresScratch);
    }

    private static boolean canUseBulkPath(MultiVectorsSource<?> source) {
        return source.vectorBytes() != null
            && source.vectorCount() > 0
            && source.vectorBytes().length == source.vectorCount() * source.vectorByteSize();
    }

    private static boolean canUseF32BulkPath(MultiFloatVectorsSource source) {
        return canUseBulkPath(source) && source.vectorByteSize() == source.vectorDims() * Float.BYTES;
    }

    private static boolean canUseBFloat16Path(MultiBFloat16VectorsSource source) {
        return canUseBulkPath(source) && source.vectorByteSize() == source.vectorDims() * Short.BYTES;
    }

    private static boolean canUseI8BulkPath(MultiByteVectorsSource source) {
        return canUseBulkPath(source) && source.vectorByteSize() == source.vectorDims();
    }
}
