/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.ash;

import java.util.Random;
import java.util.function.IntFunction;

/**
 * Asymmetric Hashing quantizer. Learns a projection matrix W that maps vectors from
 * original space to a low-dimensional latent space optimized for quantization fidelity.
 * <p>
 * The algorithm:
 * <ol>
 *   <li>KMeans clustering partitions vectors (handled externally via HierarchicalKMeans)</li>
 *   <li>Vectors are centered by subtracting their cluster centroid and normalized</li>
 *   <li>A rotation matrix W is learned (PCA init + Procrustes iterations) to maximize
 *       inner product preservation after quantization in the projected space</li>
 *   <li>Vectors are projected via W, then quantized (binary or multi-bit spherical)</li>
 *   <li>Per-vector scale and offset (float16) are stored for dot product reconstruction</li>
 * </ol>
 * <p>
 * At query time, the query is projected via W but NOT quantized (asymmetric scoring),
 * yielding higher recall than symmetric approaches.
 */
public final class AsymmetricHashingQuantizer {

    /** Training method for the projection matrix W. */
    public enum Method {
        /** Learn W via PCA + iterative Procrustes optimization. */
        LEARNED,
        /** Use a random orthogonal matrix (no training iterations). */
        RANDOM
    }

    private final float projectedDimsFraction;
    private final Method method;
    private final int nTrainingIterations;
    private final int trainingFactor;
    private final long seed;
    private final AshDimQuantizer quantizer;

    /**
     * @param projectedDimsFraction fraction of original dimensions to project to (e.g. 0.5 for half)
     * @param bitsPerDim bits per projected dimension in the body
     * @param method training method for W
     * @param nTrainingIterations number of Procrustes iterations (for LEARNED)
     * @param trainingFactor multiplier on dimension for training sample size
     * @param seed random seed
     */
    public AsymmetricHashingQuantizer(
        float projectedDimsFraction,
        int bitsPerDim,
        Method method,
        int nTrainingIterations,
        int trainingFactor,
        long seed
    ) {
        if (projectedDimsFraction <= 0 || projectedDimsFraction > 1.0f) {
            throw new IllegalArgumentException("projectedDimsFraction must be in (0, 1]");
        }
        if (bitsPerDim <= 0) {
            throw new IllegalArgumentException("bitsPerDim must be positive");
        }
        this.projectedDimsFraction = projectedDimsFraction;
        this.method = method;
        this.nTrainingIterations = nTrainingIterations;
        this.trainingFactor = trainingFactor;
        this.seed = seed;

        if (bitsPerDim == 1) {
            this.quantizer = new AshBinaryQuantizer();
        } else {
            this.quantizer = new AshSphericalScalarQuantizer(bitsPerDim);
        }
    }

    /**
     * Computes the number of projected dimensions for a given original dimension.
     */
    public int nDims(int originalDim) {
        return (int) (originalDim * projectedDimsFraction);
    }

    /**
     * Trains the projection matrix W on the given vectors and their cluster assignments.
     *
     * @param vectors all vectors in the segment, shape (nVectors, originalDim)
     * @param centroids cluster centroids, fetched by ordinal
     * @param assignments cluster assignment per vector, length nVectors
     * @return the learned projection matrix W, shape (originalDim, nDims)
     */
    public float[][] train(float[][] vectors, IntFunction<float[]> centroids, int[] assignments) {
        int originalDim = vectors[0].length;
        int nDims = nDims(originalDim);

        if (method == Method.RANDOM) {
            return randomOrthogonal(originalDim, nDims);
        }

        // Too few vectors for meaningful PCA training; fall back to random projection
        if (method == Method.LEARNED && vectors.length < nDims * 2) {
            return randomOrthogonal(originalDim, nDims);
        }

        int trainingSize = Math.min(originalDim * trainingFactor, vectors.length);
        int[] sampleIndices = sampleIndices(vectors.length, trainingSize);

        // Center and normalize the sampled vectors into a fresh array. We must not mutate
        // `vectors` in place — the writer reuses it for per-posting-list encoding later.
        float[][] xTraining = new float[trainingSize][originalDim];
        for (int i = 0; i < trainingSize; i++) {
            int srcIdx = sampleIndices[i];
            float[] centroid = centroids.apply(assignments[srcIdx]);
            double normSq = 0;
            for (int d = 0; d < originalDim; d++) {
                xTraining[i][d] = vectors[srcIdx][d] - centroid[d];
                normSq += (double) xTraining[i][d] * xTraining[i][d];
            }
            float invNorm = (float) (1.0 / Math.sqrt(normSq));
            if (Float.isFinite(invNorm)) {
                for (int d = 0; d < originalDim; d++) {
                    xTraining[i][d] *= invNorm;
                }
            }
        }

        // LEARNED: PCA init + Procrustes
        return learnedTraining(xTraining, originalDim, nDims);
    }

    /**
     * Result of encoding a single (vector, centroid) pair.
     *
     * @param xEnc quantized code in latent space, shape (nDims,)
     * @param scale scale factor applied at scoring time (norm / codeNorm)
     * @param offset additive correction term for dot product reconstruction
     */
    public record EncodedVector(float[] xEnc, float scale, float offset) {}

    /**
     * Encodes a single vector against a single centroid. This is the per-(vector,centroid) slice
     * of the batch {@link #encode} method, useful when each posting list needs vectors encoded
     * against its own centroid (e.g. for SOAR overspill, where one vector may be encoded twice
     * against two different centroids).
     *
     * @param vector the input vector, length originalDim
     * @param centroid the centroid to center against, length originalDim
     * @param w the trained projection matrix, shape (originalDim, nDims)
     * @return xEnc/scale/offset for this (vector, centroid) pair
     */
    public EncodedVector encodeOne(float[] vector, float[] centroid, float[][] w) {
        int originalDim = vector.length;
        int nDims = w[0].length;

        // Center
        float[] centered = new float[originalDim];
        double normSq = 0;
        for (int d = 0; d < originalDim; d++) {
            centered[d] = vector[d] - centroid[d];
            normSq += (double) centered[d] * centered[d];
        }
        float norm = (float) Math.sqrt(normSq);

        // Normalize
        float invNorm = norm > 0 ? 1.0f / norm : 0;
        for (int d = 0; d < originalDim; d++) {
            centered[d] *= invNorm;
        }

        // Project: xLatent = centered @ W
        float[] xLatent = new float[nDims];
        for (int j = 0; j < nDims; j++) {
            double sum = 0;
            for (int d = 0; d < originalDim; d++) {
                sum += (double) centered[d] * w[d][j];
            }
            xLatent[j] = (float) sum;
        }

        // Quantize via the existing batch API with a single row
        AshDimQuantizer.QuantizeResult qr = quantizer.encode(new float[][] { xLatent });
        float[] xEnc = qr.centeredCodes()[0];
        float codeNorm = qr.codeNorms()[0];

        // Scale: norm / codeNorm
        float scale = codeNorm > 0 ? norm / codeNorm : 0;

        // Offset = dot(vector, centroid) - dot(centroid, centroid) - scale * sum_j (centroid @ W)[j] * xEnc[j]
        double dotVecCent = 0;
        double dotCentCent = 0;
        for (int d = 0; d < originalDim; d++) {
            dotVecCent += (double) vector[d] * centroid[d];
            dotCentCent += (double) centroid[d] * centroid[d];
        }
        float offset = (float) (dotVecCent - dotCentCent);
        for (int j = 0; j < nDims; j++) {
            double transformedCent = 0;
            for (int d = 0; d < originalDim; d++) {
                transformedCent += (double) centroid[d] * w[d][j];
            }
            offset -= (float) (scale * transformedCent * xEnc[j]);
        }

        return new EncodedVector(xEnc, scale, offset);
    }

    /**
     * Encodes vectors using the trained projection matrix W.
     *
     * @param vectors all vectors, shape (nVectors, originalDim)
     * @param centroids cluster centroids, fetched by ordinal
     * @param assignments cluster assignment per vector
     * @param w the trained projection matrix, shape (originalDim, nDims)
     * @return encoding result with codes, scales, and offsets
     */
    public AsymmetricHashingResult encode(float[][] vectors, IntFunction<float[]> centroids, int[] assignments, float[][] w) {
        int nClusters = Integer.MIN_VALUE;
        int originalDim = vectors[0].length;
        int nDims = w[0].length;
        int nVectors = vectors.length;

        // Project centered+normalized vectors via W
        float[][] xLatent = new float[nVectors][nDims];
        float[] norms = new float[nVectors];
        float[] scales = new float[nVectors];
        float[] offsets = new float[nVectors];

        for (int i = 0; i < nVectors; i++) {
            int clusterLabel = assignments[i];
            float[] centroid = centroids.apply(clusterLabel);

            nClusters = Math.max(nClusters, clusterLabel);

            // Center
            double normSq = 0;
            float[] centered = new float[originalDim];
            for (int d = 0; d < originalDim; d++) {
                centered[d] = vectors[i][d] - centroid[d];
                normSq += (double) centered[d] * centered[d];
            }
            norms[i] = (float) Math.sqrt(normSq);

            // Normalize
            float invNorm = norms[i] > 0 ? 1.0f / norms[i] : 0;
            for (int d = 0; d < originalDim; d++) {
                centered[d] *= invNorm;
            }

            // Project: xLatent[i] = centered @ W
            for (int j = 0; j < nDims; j++) {
                double sum = 0;
                for (int d = 0; d < originalDim; d++) {
                    sum += (double) centered[d] * w[d][j];
                }
                xLatent[i][j] = (float) sum;
            }
        }

        // Quantize in latent space
        AshDimQuantizer.QuantizeResult qr = quantizer.encode(xLatent);
        float[][] xEnc = qr.centeredCodes();
        float[] codeNorms = qr.codeNorms();

        // Compute scale and offset
        for (int i = 0; i < nVectors; i++) {
            // scale = norm / codeNorm (stored as float16)
            scales[i] = codeNorms[i] > 0 ? norms[i] / codeNorms[i] : 0;

            // offset = dot(vector, centroid) - dot(centroid, centroid) - scale[i] * dot (centroids @ W, xEnc)
            float[] centroid = centroids.apply(assignments[i]);

            double dotVecCent = 0;
            double dotCentCent = 0;
            for (int d = 0; d < originalDim; d++) {
                dotVecCent += (double) vectors[i][d] * centroid[d];
                dotCentCent += (double) centroid[d] * centroid[d];
            }
            offsets[i] += (float) (dotVecCent - dotCentCent);

            for (int j = 0; j < nDims; j++) {
                double transformedCent = 0;
                for (int d = 0; d < originalDim; d++) {
                    transformedCent += (double) centroid[d] * w[d][j];
                }
                offsets[i] -= (float) (scales[i] * transformedCent * xEnc[i][j]);
            }
        }

        nClusters++; // we have the max cluster ID so far, add one to get the size

        return new AsymmetricHashingResult(w, xEnc, scales, offsets, nClusters);
    }

    private float[][] learnedTraining(float[][] xTraining, int originalDim, int nDims) {
        // PCA initialization: extract top nDims right singular vectors via power iteration
        // This is much faster than full SVD when nDims << originalDim
        float[][] topVectors = SvdUtil.topKRightSingularVectors(xTraining, xTraining.length, originalDim, nDims, seed);
        // P = top nDims right singular vectors transposed: rows of topVectors are the vectors
        // P shape: (originalDim × nDims) where each column is a right singular vector
        float[][] p = new float[originalDim][nDims];
        for (int i = 0; i < originalDim; i++) {
            for (int j = 0; j < nDims; j++) {
                p[i][j] = topVectors[j][i];
            }
        }

        // Project training data: X_ld = xTraining @ P (nTraining × nDims)
        int nTraining = xTraining.length;
        float[][] xLd = new float[nTraining][nDims];
        for (int i = 0; i < nTraining; i++) {
            for (int j = 0; j < nDims; j++) {
                double sum = 0;
                for (int d = 0; d < originalDim; d++) {
                    sum += (double) xTraining[i][d] * p[d][j];
                }
                xLd[i][j] = (float) sum;
            }
        }

        // Initialize random M (nDims × nDims)
        Random rng = new Random(seed);
        float[][] m = new float[nDims][nDims];
        for (int i = 0; i < nDims; i++) {
            for (int j = 0; j < nDims; j++) {
                m[i][j] = (float) rng.nextGaussian();
            }
        }

        // Iterative Procrustes
        float[][] r = null;
        for (int epoch = 0; epoch <= nTrainingIterations; epoch++) {
            // R = procrustes(M)
            r = SvdUtil.procrustes(m, nDims);

            if (epoch < nTrainingIterations) {
                // X_transformed = X_ld @ R
                float[][] xTransformed = matMul(xLd, r, nTraining, nDims, nDims);
                // Quantize
                AshDimQuantizer.QuantizeResult qr = quantizer.encode(xTransformed);
                float[][] xEnc = qr.centeredCodes();
                float[] codeNorms = qr.codeNorms();
                // Normalize encoded: xEnc[i] /= codeNorms[i]
                for (int i = 0; i < nTraining; i++) {
                    if (codeNorms[i] > 0) {
                        float inv = 1.0f / codeNorms[i];
                        for (int j = 0; j < nDims; j++) {
                            xEnc[i][j] *= inv;
                        }
                    }
                }
                // M = X_ld.T @ X_enc (nDims × nDims)
                m = matMulTransposeA(xLd, xEnc, nTraining, nDims, nDims);
            }
        }

        // W = P @ R (originalDim × nDims)
        return matMul(p, r, originalDim, nDims, nDims);
    }

    private float[][] randomOrthogonal(int originalDim, int nDims) {
        Random rng = new Random(seed);
        // Generate random matrix and orthogonalize columns via modified Gram-Schmidt
        float[][] q = new float[originalDim][nDims];
        for (int i = 0; i < originalDim; i++) {
            for (int j = 0; j < nDims; j++) {
                q[i][j] = (float) rng.nextGaussian();
            }
        }
        // Modified Gram-Schmidt: orthogonalize column by column
        for (int j = 0; j < nDims; j++) {
            // Subtract projections of previous columns
            for (int prev = 0; prev < j; prev++) {
                double dot = 0;
                for (int i = 0; i < originalDim; i++) {
                    dot += (double) q[i][j] * q[i][prev];
                }
                for (int i = 0; i < originalDim; i++) {
                    q[i][j] -= (float) dot * q[i][prev];
                }
            }
            // Normalize
            double normSq = 0;
            for (int i = 0; i < originalDim; i++) {
                normSq += (double) q[i][j] * q[i][j];
            }
            float invNorm = (float) (1.0 / Math.sqrt(normSq));
            for (int i = 0; i < originalDim; i++) {
                q[i][j] *= invNorm;
            }
        }
        return q;
    }

    /**
     * Returns an array of {@code sampleSize} distinct indices in [0, n), chosen via a
     * Fisher-Yates partial shuffle seeded with this quantizer's seed. If {@code sampleSize >= n}
     * the returned array is just [0, n) in order.
     */
    private int[] sampleIndices(int n, int sampleSize) {
        if (sampleSize >= n) {
            int[] all = new int[n];
            for (int i = 0; i < n; i++) all[i] = i;
            return all;
        }
        Random rng = new Random(seed);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        for (int i = 0; i < sampleSize; i++) {
            int j = i + rng.nextInt(n - i);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
        int[] picked = new int[sampleSize];
        System.arraycopy(indices, 0, picked, 0, sampleSize);
        return picked;
    }

    private float[][] sampleRows(float[][] data, int sampleSize) {
        if (sampleSize >= data.length) {
            return data;
        }
        Random rng = new Random(seed);
        // Fisher-Yates partial shuffle
        int[] indices = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            indices[i] = i;
        }
        for (int i = 0; i < sampleSize; i++) {
            int j = i + rng.nextInt(data.length - i);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
        float[][] sample = new float[sampleSize][];
        for (int i = 0; i < sampleSize; i++) {
            sample[i] = data[indices[i]];
        }
        return sample;
    }

    /** C = A @ B where A is (m × k), B is (k × n) */
    private static float[][] matMul(float[][] a, float[][] b, int m, int k, int n) {
        float[][] c = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int l = 0; l < k; l++) {
                    sum += (double) a[i][l] * b[l][j];
                }
                c[i][j] = (float) sum;
            }
        }
        return c;
    }

    /** C = A.T @ B where A is (m × k), B is (m × n), result is (k × n) */
    private static float[][] matMulTransposeA(float[][] a, float[][] b, int m, int k, int n) {
        float[][] c = new float[k][n];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int l = 0; l < m; l++) {
                    sum += (double) a[l][i] * b[l][j];
                }
                c[i][j] = (float) sum;
            }
        }
        return c;
    }

    /**
     * Truncates a float to float16 precision (round to nearest, ties to even).
     */
    private static float toFloat16(float value) {
        return Float.float16ToFloat(Float.floatToFloat16(value));
    }

    /**
     * Result of running ASH-specific k-means clustering.
     */
    public record AshKMeansResult(float[][] centroids, int[] assignments) {}

    /**
     * Runs a simple k-means clustering on the given vectors to produce ASH centering clusters.
     * This is independent of the IVF clustering used for posting list layout.
     *
     * @param vectors all vectors, shape (nVectors, dim)
     * @param nClusters number of ASH centering clusters (typically 16)
     * @param maxIterations maximum k-means iterations
     * @param seed random seed for initialization
     * @return centroids and per-vector assignments
     */
    public static AshKMeansResult runAshKMeans(float[][] vectors, int nClusters, int maxIterations, long seed) {
        int nVectors = vectors.length;
        int dim = vectors[0].length;
        Random rng = new Random(seed);

        // k-means++ initialization
        float[][] centroids = new float[nClusters][dim];
        int firstIdx = rng.nextInt(nVectors);
        System.arraycopy(vectors[firstIdx], 0, centroids[0], 0, dim);

        double[] minDist = new double[nVectors];
        for (int c = 1; c < nClusters; c++) {
            // Compute distances to nearest existing centroid
            double totalDist = 0;
            for (int i = 0; i < nVectors; i++) {
                double dist = squaredDistance(vectors[i], centroids[c - 1], dim);
                if (c == 1 || dist < minDist[i]) {
                    minDist[i] = dist;
                }
                totalDist += minDist[i];
            }
            // Sample proportional to distance
            double threshold = rng.nextDouble() * totalDist;
            double cumulative = 0;
            int chosen = nVectors - 1;
            for (int i = 0; i < nVectors; i++) {
                cumulative += minDist[i];
                if (cumulative >= threshold) {
                    chosen = i;
                    break;
                }
            }
            System.arraycopy(vectors[chosen], 0, centroids[c], 0, dim);
        }

        // Lloyd's iterations
        int[] assignments = new int[nVectors];
        for (int iter = 0; iter < maxIterations; iter++) {
            // Assign
            boolean changed = false;
            for (int i = 0; i < nVectors; i++) {
                int bestC = 0;
                double bestDist = squaredDistance(vectors[i], centroids[0], dim);
                for (int c = 1; c < nClusters; c++) {
                    double dist = squaredDistance(vectors[i], centroids[c], dim);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestC = c;
                    }
                }
                if (assignments[i] != bestC) {
                    assignments[i] = bestC;
                    changed = true;
                }
            }
            if (!changed) break;

            // Update centroids
            int[] counts = new int[nClusters];
            for (int c = 0; c < nClusters; c++) {
                java.util.Arrays.fill(centroids[c], 0f);
            }
            for (int i = 0; i < nVectors; i++) {
                int c = assignments[i];
                counts[c]++;
                for (int d = 0; d < dim; d++) {
                    centroids[c][d] += vectors[i][d];
                }
            }
            for (int c = 0; c < nClusters; c++) {
                if (counts[c] > 0) {
                    float inv = 1.0f / counts[c];
                    for (int d = 0; d < dim; d++) {
                        centroids[c][d] *= inv;
                    }
                }
            }
        }
        return new AshKMeansResult(centroids, assignments);
    }

    private static double squaredDistance(float[] a, float[] b, int dim) {
        double sum = 0;
        for (int d = 0; d < dim; d++) {
            double diff = a[d] - b[d];
            sum += diff * diff;
        }
        return sum;
    }
}
