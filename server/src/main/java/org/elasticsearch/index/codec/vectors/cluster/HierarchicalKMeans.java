/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.cluster;

import org.apache.lucene.search.TaskExecutor;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * An implementation of the hierarchical k-means algorithm that better partitions data than naive k-means.
 *
 * @param <V> the array type for vectors and centroids ({@code float[]} or {@code byte[]})
 */
public class HierarchicalKMeans<V> {

    public static final int MAXK = 128;
    public static final int SAMPLES_PER_CLUSTER_DEFAULT = 64;
    public static final float DEFAULT_SOAR_LAMBDA = 1.0f;
    public static final int NO_SOAR_ASSIGNMENT = -1;
    private static final int MIN_VECTORS_PRE_THREAD = 64;

    public static final boolean USE_BALANCING = true;
    public static final int MAX_ITERATIONS_DEFAULT = USE_BALANCING ? 2 : 6;

    final int dimension;
    final int maxIterations;
    final int samplesPerCluster;
    final int clustersPerNeighborhood;
    final float soarLambda;
    final CentroidOps<V> ops;

    private final TaskExecutor executor;
    private final int numWorkers;

    private HierarchicalKMeans(
        CentroidOps<V> ops,
        int dimension,
        TaskExecutor executor,
        int numWorkers,
        int maxIterations,
        int samplesPerCluster,
        int clustersPerNeighborhood,
        float soarLambda
    ) {
        this.ops = ops;
        this.dimension = dimension;
        this.executor = executor;
        this.numWorkers = numWorkers;
        this.maxIterations = maxIterations;
        this.samplesPerCluster = samplesPerCluster;
        this.clustersPerNeighborhood = clustersPerNeighborhood;
        this.soarLambda = soarLambda;
    }

    public static <V> HierarchicalKMeans<V> ofSerial(CentroidOps<V> ops, int dimension) {
        return ofSerial(ops, dimension, MAX_ITERATIONS_DEFAULT, SAMPLES_PER_CLUSTER_DEFAULT, MAXK, DEFAULT_SOAR_LAMBDA);
    }

    public static <V> HierarchicalKMeans<V> ofSerial(
        CentroidOps<V> ops,
        int dimension,
        int maxIterations,
        int samplesPerCluster,
        int clustersPerNeighborhood,
        float soarLambda
    ) {
        return new HierarchicalKMeans<>(ops, dimension, null, 1, maxIterations, samplesPerCluster, clustersPerNeighborhood, soarLambda);
    }

    public static <V> HierarchicalKMeans<V> ofConcurrent(CentroidOps<V> ops, int dimension, TaskExecutor executor, int numWorkers) {
        return ofConcurrent(
            ops,
            dimension,
            executor,
            numWorkers,
            MAX_ITERATIONS_DEFAULT,
            SAMPLES_PER_CLUSTER_DEFAULT,
            MAXK,
            DEFAULT_SOAR_LAMBDA
        );
    }

    public static <V> HierarchicalKMeans<V> ofConcurrent(
        CentroidOps<V> ops,
        int dimension,
        TaskExecutor executor,
        int numWorkers,
        int maxIterations,
        int samplesPerCluster,
        int clustersPerNeighborhood,
        float soarLambda
    ) {
        return new HierarchicalKMeans<>(
            ops,
            dimension,
            executor,
            numWorkers,
            maxIterations,
            samplesPerCluster,
            clustersPerNeighborhood,
            soarLambda
        );
    }

    /**
     * clusters the set of vectors by starting with a rough number of partitions and then recursively refining those
     * lastly a pass is made to adjust nearby neighborhoods and add an extra assignment per vector to nearby neighborhoods
     *
     * @param vectors the vectors to cluster
     * @param targetSize the rough number of vectors that should be attached to a cluster
     * @return the centroids and the vectors assignments and SOAR (spilled from nearby neighborhoods) assignments
     * @throws IOException is thrown if vectors is inaccessible
     */
    public KMeansResult<V> cluster(ClusteringVectorValues<V> vectors, int targetSize) throws IOException {
        if (vectors.size() == 0) {
            return KMeansIntermediate.empty(ops);
        }

        // if we have a small number of vectors calculate the centroid directly
        if (vectors.size() <= targetSize) {
            V centroid;
            if (ops instanceof CentroidOps.ByteOps byteOps) {
                // Accumulate in int precision to avoid byte overflow, then round once
                int[] acc = new int[dimension];
                for (int i = 0; i < vectors.size(); i++) {
                    byte[] vector = (byte[]) vectors.vectorValue(i);
                    for (int d = 0; d < dimension; d++) {
                        acc[d] += vector[d];
                    }
                }
                byte[] byteCentroid = new byte[dimension];
                float count = vectors.size();
                for (int d = 0; d < dimension; d++) {
                    byteCentroid[d] = (byte) Math.clamp(Math.round(acc[d] / count), -128, 127);
                }
                @SuppressWarnings("unchecked")
                V typed = (V) byteCentroid;
                centroid = typed;
            } else {
                centroid = ops.newCentroid(dimension);
                for (int i = 0; i < vectors.size(); i++) {
                    V vector = vectors.vectorValue(i);
                    ops.accumulate(centroid, vector, dimension);
                }
                ops.divide(centroid, vectors.size(), dimension);
            }
            V[] centroids = ops.newCentroidArray(1, 0);
            centroids[0] = centroid;
            return new KMeansIntermediate<>(centroids, new int[vectors.size()]);
        }

        // partition the space
        KMeansIntermediate<V> kMeansIntermediate = clusterAndSplit(vectors, targetSize);

        if (kMeansIntermediate.centroids().length > 1 && kMeansIntermediate.centroids().length < vectors.size()) {
            int localSampleSize = Math.min(kMeansIntermediate.centroids().length * samplesPerCluster / 2, vectors.size());
            KMeansLocal<V> kMeansLocal = buildKmeansLocalFinal(vectors.size(), localSampleSize);
            kMeansLocal.cluster(vectors, kMeansIntermediate, clustersPerNeighborhood, soarLambda);
        }
        return kMeansIntermediate;
    }

    private KMeansIntermediate<V> clusterAndSplit(final ClusteringVectorValues<V> vectors, final int targetSize) throws IOException {
        if (vectors.size() <= targetSize) {
            return KMeansIntermediate.empty(ops);
        }

        int k = Math.clamp((int) ((vectors.size() + targetSize / 2.0f) / (float) targetSize), 2, MAXK);
        int m = Math.min(k * samplesPerCluster, vectors.size());

        // TODO: instead of creating a sub-cluster assignments reuse the parent array each time
        int[] assignments = new int[vectors.size()];
        // ensure we don't over assign to cluster 0 without adjusting it
        Arrays.fill(assignments, -1);
        V[] centroids = KMeansLocal.pickInitialCentroids(vectors, k, ops);
        KMeansIntermediate<V> kMeansIntermediate = new KMeansIntermediate<>(centroids, assignments, vectors::ordToDoc);
        KMeansLocal<V> kMeansLocal = buildKmeansLocal(vectors.size(), m);
        kMeansLocal.cluster(vectors, kMeansIntermediate);

        // TODO: consider adding cluster size counts to the kmeans algo
        // handle assignment here so we can track distance and cluster size
        int[] centroidVectorCount = new int[centroids.length];
        int effectiveCluster = -1;
        int effectiveK = 0;
        for (int assigment : assignments) {
            centroidVectorCount[assigment]++;
            // this cluster has received an assignment, its now effective, but only count it once
            if (centroidVectorCount[assigment] == 1) {
                effectiveK++;
                effectiveCluster = assigment;
            }
        }

        if (effectiveK == 1) {
            V[] singleClusterCentroid = ops.newCentroidArray(1, 0);
            singleClusterCentroid[0] = centroids[effectiveCluster];
            kMeansIntermediate.setCentroids(singleClusterCentroid);
            Arrays.fill(kMeansIntermediate.assignments(), 0);
            return kMeansIntermediate;
        }

        int centroidIndexOffset = 0; // tracks the cumulative change in centroid indices due to splits and removals
        for (int c = 0; c < centroidVectorCount.length; c++) {

            // Recurse for each cluster which is larger than targetSize
            // Give ourselves 30% margin for the target size
            final int count = centroidVectorCount[c];
            final int adjustedCentroid = c + centroidIndexOffset;
            if (count > targetSize) {
                final ClusteringVectorValues<V> sample = createClusterSlice(count, adjustedCentroid, vectors, assignments);

                // TODO: consider iterative here instead of recursive
                // recursive call to build out the sub partitions around this centroid c
                // subsequently reconcile and flatten the space of all centroids and assignments into one structure we can return
                KMeansIntermediate<V> subPartitions = clusterAndSplit(sample, targetSize);
                // Update offset: split replaces 1 centroid with subPartitions.centroids().length centroids
                centroidIndexOffset += updateAssignmentsWithRecursiveSplit(kMeansIntermediate, adjustedCentroid, subPartitions);
            } else if (count == 0) {
                // remove empty clusters
                final int newSize = kMeansIntermediate.centroids().length - 1;
                final V[] newCentroids = ops.newCentroidArray(newSize, 0);
                ops.arrayCopy(kMeansIntermediate.centroids(), 0, newCentroids, 0, adjustedCentroid);
                ops.arrayCopy(
                    kMeansIntermediate.centroids(),
                    adjustedCentroid + 1,
                    newCentroids,
                    adjustedCentroid,
                    newSize - adjustedCentroid
                );
                // we need to update the assignments to reflect the new centroid ordinals
                for (int i = 0; i < kMeansIntermediate.assignments().length; i++) {
                    if (kMeansIntermediate.assignments()[i] > adjustedCentroid) {
                        kMeansIntermediate.assignments()[i]--;
                    }
                }
                kMeansIntermediate.setCentroids(newCentroids);
                // Update offset: removed 1 centroid
                centroidIndexOffset--;
            }
        }

        return kMeansIntermediate;
    }

    private KMeansLocal<V> buildKmeansLocal(int numVectors, int localSampleSize) {
        int numWorkers = Math.min(this.numWorkers, numVectors / MIN_VECTORS_PRE_THREAD);
        // if there is no executor or there is no enough vectors for more than one thread, use the serial version
        if (USE_BALANCING) {
            return executor == null || numWorkers <= 1
                ? new BalancedOTKMeansLocalSerial<>(ops, localSampleSize, maxIterations)
                : new BalancedOTKMeansLocalConcurrent<>(ops, executor, numWorkers, localSampleSize, maxIterations);
        } else {
            return executor == null || numWorkers <= 1
                ? new LloydKMeansLocalSerial<>(ops, localSampleSize, maxIterations)
                : new LloydKMeansLocalConcurrent<>(ops, executor, numWorkers, localSampleSize, maxIterations);
        }
    }

    private KMeansLocal<V> buildKmeansLocalFinal(int numVectors, int localSampleSize) {
        int numWorkers = Math.min(this.numWorkers, numVectors / MIN_VECTORS_PRE_THREAD);
        // if there is no executor or there is no enough vectors for more than one thread, use the serial version
        if (USE_BALANCING) {
            return executor == null || numWorkers <= 1
                ? new BalancedASKMeansLocalSerial<>(ops, localSampleSize, maxIterations)
                : new BalancedASKMeansLocalConcurrent<>(ops, executor, numWorkers, localSampleSize, maxIterations);
        } else {
            return executor == null || numWorkers <= 1
                ? new LloydKMeansLocalSerial<>(ops, localSampleSize, maxIterations)
                : new LloydKMeansLocalConcurrent<>(ops, executor, numWorkers, localSampleSize, maxIterations);
        }
    }

    static <V> ClusteringVectorValues<V> createClusterSlice(
        int clusterSize,
        int cluster,
        ClusteringVectorValues<V> vectors,
        int[] assignments
    ) {
        assert assignments.length == vectors.size();
        int[] slice = new int[clusterSize];
        int idx = 0;
        for (int i = 0; i < assignments.length; i++) {
            if (assignments[i] == cluster) {
                slice[idx] = i;
                idx++;
            }
        }
        assert idx == clusterSize;

        return new ClusteringVectorValuesSlice<>(vectors, i -> slice[i], slice.length);
    }

    /**
     * Merge the child centroids in {@code subPartitions} into the K means results.
     * {@code subPartitions} are inserted into the results next to the parent centroid
     * at position {@code cluster}.
     * @param current Current results
     * @param cluster Index of the split centroid
     * @param subPartitions The new centroids resulting from the split
     * @return The number of centroids added excluding the one that is replaced
     */
    int updateAssignmentsWithRecursiveSplit(KMeansIntermediate<V> current, int cluster, KMeansIntermediate<V> subPartitions) {
        if (subPartitions.centroids().length == 0) {
            return 0; // nothing to do, sub-partitions is empty
        }
        int orgCentroidsSize = current.centroids().length;
        int newCentroidsSize = current.centroids().length + subPartitions.centroids().length - 1;

        // update based on the outcomes from the split clusters recursion
        V[] newCentroids = ops.newCentroidArray(newCentroidsSize, 0);

        // copy centroids prior to the split
        ops.arrayCopy(current.centroids(), 0, newCentroids, 0, cluster);
        // insert the split partitions replacing the original cluster
        ops.arrayCopy(subPartitions.centroids(), 0, newCentroids, cluster, subPartitions.centroids().length);
        // append the remainder
        ops.arrayCopy(
            current.centroids(),
            cluster + 1,
            newCentroids,
            cluster + subPartitions.centroids().length,
            orgCentroidsSize - cluster - 1
        );

        assert Arrays.stream(newCentroids).allMatch(Objects::nonNull);
        current.setCentroids(newCentroids);

        // update the remaining cluster assignments to point to the new centroids after the split cluster
        // IMPORTANT: Do this BEFORE updating split cluster assignments to avoid double-updating
        for (int i = 0; i < current.assignments().length; i++) {
            if (current.assignments()[i] > cluster) {
                current.assignments()[i] = current.assignments()[i] - 1 + subPartitions.centroids().length;
            }
        }
        // update the split cluster assignments to point to the new centroids from the split cluster
        for (int i = 0; i < subPartitions.assignments().length; i++) {
            int parentOrd = subPartitions.ordToDoc(i);
            assert current.assignments()[parentOrd] == cluster;
            current.assignments()[parentOrd] = cluster + subPartitions.assignments()[i];
        }

        // number of items inserted with 1 element replaced
        return subPartitions.centroids().length - 1;
    }
}
