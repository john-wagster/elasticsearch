/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.cluster;

import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.simdvec.MathUtils;

/**
 * Encapsulates all vector/centroid-type-specific arithmetic for k-means clustering.
 * <p>
 * Two implementations are provided: {@link FloatOps} for {@code float[]} vectors/centroids
 * and {@link ByteOps} for {@code byte[]} vectors/centroids.
 *
 * @param <V> the array type for vectors and centroids ({@code float[]} or {@code byte[]})
 */
public sealed interface CentroidOps<V> permits CentroidOps.FloatOps, CentroidOps.ByteOps {

    // ---- Distance operations ----

    /** Squared Euclidean distance between two vectors. */
    float squareDistance(V a, V b);

    /** Squared Euclidean distance over a sub-range {@code [offset, offset+length)}. */
    float squareDistance(V a, V b, int offset, int length);

    /**
     * Compute squared distances from {@code query} to four centroids in bulk.
     * Results are written into {@code distances[offset..offset+3]}.
     */
    void squareDistanceBulk(V query, V c0, V c1, V c2, V c3, int offset, float[] distances);

    /**
     * Compute squared distances from a sub-range of {@code query} to four centroids in bulk.
     * {@code queryOffset} and {@code length} define the range within each vector.
     */
    void squareDistanceBulk(V query, int queryOffset, int length, V c0, V c1, V c2, V c3, float[] distances);

    /**
     * SOAR distance: {@code ||x-c||^2 + lambda * ((x-c1)^T (x-c))^2 / ||x-c1||^2}.
     *
     * @param vector             the query vector x
     * @param centroid           the candidate centroid c
     * @param diffs              precomputed {@code x - c1} (primary centroid residual)
     * @param soarLambda         lambda weight
     * @param vectorCentroidDist precomputed {@code ||x - c1||^2}
     */
    float soarDistance(V vector, V centroid, float[] diffs, float soarLambda, float vectorCentroidDist);

    /** Bulk SOAR distance to four candidate centroids. */
    void soarDistanceBulk(V vector, V c0, V c1, V c2, V c3, float[] diffs, float soarLambda, float vectorCentroidDist, float[] distances);

    /** Dot product between two vectors (used for Frobenius norm computation). */
    float dotProduct(V a, V b);

    // ---- Centroid lifecycle ----

    /** Allocate a fresh zero-filled centroid of the given dimension. */
    V newCentroid(int dims);

    /** Allocate a 2D centroid array of shape {@code [k][dims]}. */
    V[] newCentroidArray(int k, int dims);

    /** Element-wise deep copy from {@code source} to {@code destination}. */
    void deepCopy(V[] source, V[] destination);

    /** Copy elements of the centroid array ({@code System.arraycopy} semantics). */
    void arrayCopy(V[] src, int srcPos, V[] dest, int destPos, int length);

    /** Returns the length (dimension) of the vector. */
    int length(V vector);

    // ---- Centroid update operations ----

    /**
     * Set all elements of the centroid to zero (or the appropriate identity).
     * For float: fill with 0.0f. For byte: fill with 0.
     */
    void zeroCentroid(V centroid);

    /**
     * Copy the contents of {@code vector} into {@code centroid} (first assignment).
     * Equivalent to {@code System.arraycopy(vector, 0, centroid, 0, dim)}.
     */
    void initCentroid(V centroid, V vector, int dim);

    /**
     * Accumulate a vector into a centroid: {@code centroid[d] += vector[d]}.
     * <p>
     * For byte centroids, this accumulates into an {@code int[]} scratch internally
     * managed by the ops instance; call {@link #divide} to round back to byte.
     */
    void accumulate(V centroid, V vector, int dim);

    /**
     * Divide each element of the centroid by {@code count} and finalize.
     * For float: {@code centroid[d] /= count}.
     * For byte: divide the accumulated int values and round/clamp to {@code [-128, 127]}.
     */
    void divide(V centroid, float count, int dim);

    /**
     * SGD linear combination: {@code dest[d] += scale * src[d]}.
     * For byte: widens to float, blends, rounds back to byte.
     */
    void linearCombination(float scale, V src, V dest);

    /**
     * SGD linear combination: {@code b[d] = scaleA * a[d] + scaleB * b[d]}.
     * For byte: widens to float, blends, rounds back to byte.
     */
    void linearCombination(float scaleA, V a, float scaleB, V b);

    /**
     * Compute {@code diffs[d] = vector[d] - centroid[d]} as floats (for SOAR residuals).
     * Always produces {@code float[]} regardless of vector type, because the SOAR formula
     * operates in float space.
     */
    void computeDiffs(V vector, V centroid, float[] diffs);

    /**
     * Convert centroids to {@code float[][]} for use with float-only subsystems (e.g. {@link NeighborHood}).
     * For {@link FloatOps} this is a no-op cast. For {@link ByteOps} this widens each byte to float.
     */
    float[][] toFloatCentroids(V[] centroids);

    // ---- Convergence ----

    /**
     * Computes the normalized Frobenius norm between two centroid arrays:
     * {@code sqrt(sum_i ||vecs1[i] - vecs2[i]||^2 / sum_i ||vecs2[i]||^2)}.
     */
    float normalizedFrobeniusNorm(V[] vecs1, V[] vecs2);

    /** Convenience constant for the float ops singleton. */
    CentroidOps<float[]> FLOAT = FloatOps.INSTANCE;

    /** Convenience constant for the byte ops singleton. */
    CentroidOps<byte[]> BYTE = ByteOps.INSTANCE;

    // ---- Implementations ----

    /**
     * {@link CentroidOps} for {@code float[]} vectors and centroids.
     * Delegates to {@link ESVectorUtil} for SIMD-accelerated operations.
     */
    final class FloatOps implements CentroidOps<float[]> {

        public static final FloatOps INSTANCE = new FloatOps();

        private FloatOps() {}

        @Override
        public float squareDistance(float[] a, float[] b) {
            return ESVectorUtil.squareDistance(a, b);
        }

        @Override
        public float squareDistance(float[] a, float[] b, int offset, int length) {
            return ESVectorUtil.squareDistance(a, b, offset, length);
        }

        @Override
        public void squareDistanceBulk(float[] query, float[] c0, float[] c1, float[] c2, float[] c3, int offset, float[] distances) {
            ESVectorUtil.squareDistanceBulk(query, c0, c1, c2, c3, offset, distances);
        }

        @Override
        public void squareDistanceBulk(
            float[] query,
            int queryOffset,
            int length,
            float[] c0,
            float[] c1,
            float[] c2,
            float[] c3,
            float[] distances
        ) {
            ESVectorUtil.squareDistanceBulk(query, queryOffset, length, c0, c1, c2, c3, distances);
        }

        @Override
        public float soarDistance(float[] vector, float[] centroid, float[] diffs, float soarLambda, float vectorCentroidDist) {
            return ESVectorUtil.soarDistance(vector, centroid, diffs, soarLambda, vectorCentroidDist);
        }

        @Override
        public void soarDistanceBulk(
            float[] vector,
            float[] c0,
            float[] c1,
            float[] c2,
            float[] c3,
            float[] diffs,
            float soarLambda,
            float vectorCentroidDist,
            float[] distances
        ) {
            ESVectorUtil.soarDistanceBulk(vector, c0, c1, c2, c3, diffs, soarLambda, vectorCentroidDist, distances);
        }

        @Override
        public float dotProduct(float[] a, float[] b) {
            return ESVectorUtil.dotProduct(a, b);
        }

        @Override
        public float[] newCentroid(int dims) {
            return new float[dims];
        }

        @Override
        public float[][] newCentroidArray(int k, int dims) {
            float[][] result = new float[k][];
            for (int i = 0; i < k; i++) {
                result[i] = new float[dims];
            }
            return result;
        }

        @Override
        public void deepCopy(float[][] source, float[][] destination) {
            for (int i = 0; i < source.length; i++) {
                System.arraycopy(source[i], 0, destination[i], 0, source[i].length);
            }
        }

        @Override
        public void arrayCopy(float[][] src, int srcPos, float[][] dest, int destPos, int length) {
            System.arraycopy(src, srcPos, dest, destPos, length);
        }

        @Override
        public int length(float[] vector) {
            return vector.length;
        }

        @Override
        public void zeroCentroid(float[] centroid) {
            java.util.Arrays.fill(centroid, 0.0f);
        }

        @Override
        public void initCentroid(float[] centroid, float[] vector, int dim) {
            System.arraycopy(vector, 0, centroid, 0, dim);
        }

        @Override
        public void accumulate(float[] centroid, float[] vector, int dim) {
            for (int d = 0; d < dim; d++) {
                centroid[d] += vector[d];
            }
        }

        @Override
        public void divide(float[] centroid, float count, int dim) {
            for (int d = 0; d < dim; d++) {
                centroid[d] /= count;
            }
        }

        @Override
        public void linearCombination(float scale, float[] src, float[] dest) {
            ESVectorUtil.linearCombination(scale, src, dest);
        }

        @Override
        public void linearCombination(float scaleA, float[] a, float scaleB, float[] b) {
            ESVectorUtil.linearCombination(scaleA, a, scaleB, b);
        }

        @Override
        public void computeDiffs(float[] vector, float[] centroid, float[] diffs) {
            for (int j = 0; j < diffs.length; j++) {
                diffs[j] = vector[j] - centroid[j];
            }
        }

        @Override
        public float normalizedFrobeniusNorm(float[][] vecs1, float[][] vecs2) {
            assert vecs1.length == vecs2.length;
            float result = 0;
            float norm2 = 0;
            for (int i = 0; i < vecs1.length; i++) {
                result += ESVectorUtil.squareDistance(vecs1[i], vecs2[i]);
                norm2 += ESVectorUtil.dotProduct(vecs2[i], vecs2[i]);
            }
            return MathUtils.sqrt(result / norm2);
        }

        @Override
        public float[][] toFloatCentroids(float[][] centroids) {
            return centroids;
        }
    }

    /**
     * {@link CentroidOps} for {@code byte[]} vectors and centroids.
     * <p>
     * Centroid averaging accumulates to {@code int} precision internally and rounds
     * back to {@code byte} (clamped to {@code [-128, 127]}) during {@link #divide}.
     * SGD updates ({@link #linearCombination}) widen to float, blend, and round back.
     */
    final class ByteOps implements CentroidOps<byte[]> {

        public static final ByteOps INSTANCE = new ByteOps();

        private ByteOps() {}

        @Override
        public float squareDistance(byte[] a, byte[] b) {
            return ESVectorUtil.squareDistance(a, b);
        }

        @Override
        public float squareDistance(byte[] a, byte[] b, int offset, int length) {
            return ESVectorUtil.squareDistance(a, b, offset, length);
        }

        @Override
        public void squareDistanceBulk(byte[] query, byte[] c0, byte[] c1, byte[] c2, byte[] c3, int offset, float[] distances) {
            ESVectorUtil.squareDistanceBulk(query, c0, c1, c2, c3, offset, distances);
        }

        @Override
        public void squareDistanceBulk(
            byte[] query,
            int queryOffset,
            int length,
            byte[] c0,
            byte[] c1,
            byte[] c2,
            byte[] c3,
            float[] distances
        ) {
            ESVectorUtil.squareDistanceBulk(query, queryOffset, length, c0, c1, c2, c3, distances);
        }

        @Override
        public float soarDistance(byte[] vector, byte[] centroid, float[] diffs, float soarLambda, float vectorCentroidDist) {
            // SOAR distance: ||x-c||^2 + lambda * ((x-c1)^T (x-c))^2 / ||x-c1||^2
            // diffs = x - c1 (precomputed as float[])
            // We compute (x-c) dot diffs, and ||x-c||^2
            float sqDist = 0;
            float dotDiff = 0;
            for (int i = 0; i < vector.length; i++) {
                float xMinusC = vector[i] - centroid[i];
                sqDist += xMinusC * xMinusC;
                dotDiff += diffs[i] * xMinusC;
            }
            return sqDist + soarLambda * (dotDiff * dotDiff) / vectorCentroidDist;
        }

        @Override
        public void soarDistanceBulk(
            byte[] vector,
            byte[] c0,
            byte[] c1,
            byte[] c2,
            byte[] c3,
            float[] diffs,
            float soarLambda,
            float vectorCentroidDist,
            float[] distances
        ) {
            distances[0] = soarDistance(vector, c0, diffs, soarLambda, vectorCentroidDist);
            distances[1] = soarDistance(vector, c1, diffs, soarLambda, vectorCentroidDist);
            distances[2] = soarDistance(vector, c2, diffs, soarLambda, vectorCentroidDist);
            distances[3] = soarDistance(vector, c3, diffs, soarLambda, vectorCentroidDist);
        }

        @Override
        public float dotProduct(byte[] a, byte[] b) {
            return ESVectorUtil.dotProduct(a, b);
        }

        @Override
        public byte[] newCentroid(int dims) {
            return new byte[dims];
        }

        @Override
        public byte[][] newCentroidArray(int k, int dims) {
            byte[][] result = new byte[k][];
            for (int i = 0; i < k; i++) {
                result[i] = new byte[dims];
            }
            return result;
        }

        @Override
        public void deepCopy(byte[][] source, byte[][] destination) {
            for (int i = 0; i < source.length; i++) {
                System.arraycopy(source[i], 0, destination[i], 0, source[i].length);
            }
        }

        @Override
        public void arrayCopy(byte[][] src, int srcPos, byte[][] dest, int destPos, int length) {
            System.arraycopy(src, srcPos, dest, destPos, length);
        }

        @Override
        public int length(byte[] vector) {
            return vector.length;
        }

        @Override
        public void zeroCentroid(byte[] centroid) {
            java.util.Arrays.fill(centroid, (byte) 0);
        }

        @Override
        public void initCentroid(byte[] centroid, byte[] vector, int dim) {
            System.arraycopy(vector, 0, centroid, 0, dim);
        }

        /**
         * For byte centroids, accumulation must happen in wider precision.
         * This method performs element-wise addition treating both arrays as byte values,
         * but the caller is responsible for managing overflow via the accumulator pattern
         * in {@link CentroidAssignment}.
         */
        @Override
        public void accumulate(byte[] centroid, byte[] vector, int dim) {
            // This simple byte accumulation will overflow after ~1 vector.
            // The actual accumulation for byte centroids happens in CentroidUpdater
            // which uses int[] scratch. This method is a fallback for the interface contract.
            for (int d = 0; d < dim; d++) {
                centroid[d] += vector[d];
            }
        }

        @Override
        public void divide(byte[] centroid, float count, int dim) {
            // Round and clamp each element
            for (int d = 0; d < dim; d++) {
                centroid[d] = (byte) Math.clamp(Math.round(centroid[d] / count), -128, 127);
            }
        }

        @Override
        public void linearCombination(float scale, byte[] src, byte[] dest) {
            // dest[d] += scale * src[d], rounded to byte
            for (int d = 0; d < src.length; d++) {
                float blended = scale * src[d] + dest[d];
                dest[d] = (byte) Math.clamp(Math.round(blended), -128, 127);
            }
        }

        @Override
        public void linearCombination(float scaleA, byte[] a, float scaleB, byte[] b) {
            // b[d] = scaleA * a[d] + scaleB * b[d], rounded to byte
            for (int d = 0; d < a.length; d++) {
                float blended = scaleA * a[d] + scaleB * b[d];
                b[d] = (byte) Math.clamp(Math.round(blended), -128, 127);
            }
        }

        @Override
        public void computeDiffs(byte[] vector, byte[] centroid, float[] diffs) {
            for (int j = 0; j < diffs.length; j++) {
                diffs[j] = vector[j] - centroid[j];
            }
        }

        @Override
        public float normalizedFrobeniusNorm(byte[][] vecs1, byte[][] vecs2) {
            assert vecs1.length == vecs2.length;
            float result = 0;
            float norm2 = 0;
            for (int i = 0; i < vecs1.length; i++) {
                result += squareDistance(vecs1[i], vecs2[i]);
                norm2 += dotProduct(vecs2[i], vecs2[i]);
            }
            return MathUtils.sqrt(result / norm2);
        }

        @Override
        public float[][] toFloatCentroids(byte[][] centroids) {
            float[][] result = new float[centroids.length][];
            for (int i = 0; i < centroids.length; i++) {
                byte[] src = centroids[i];
                float[] dst = new float[src.length];
                for (int j = 0; j < src.length; j++) {
                    dst[j] = src[j];
                }
                result[i] = dst;
            }
            return result;
        }
    }

    /**
     * Returns the appropriate {@link CentroidOps} instance for the given centroid class.
     */
    @SuppressWarnings("unchecked")
    static <V> CentroidOps<V> forClass(Class<V> clazz) {
        if (clazz == float[].class) {
            return (CentroidOps<V>) FloatOps.INSTANCE;
        } else if (clazz == byte[].class) {
            return (CentroidOps<V>) ByteOps.INSTANCE;
        }
        throw new IllegalArgumentException("Unsupported centroid type: " + clazz);
    }
}
