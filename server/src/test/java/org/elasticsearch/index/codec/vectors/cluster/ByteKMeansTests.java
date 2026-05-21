/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.index.codec.vectors.cluster;

import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.elasticsearch.index.codec.vectors.cluster.HierarchicalKMeans.NO_SOAR_ASSIGNMENT;

/**
 * Tests for generic KMeans clustering with {@link CentroidOps#BYTE}.
 */
public class ByteKMeansTests extends ESTestCase {

    public void testHierarchicalKMeansByte() throws IOException {
        int nVectors = randomIntBetween(200, 2000);
        int dims = randomIntBetween(4, 32);
        int nClusters = randomIntBetween(2, 20);
        int sampleSize = randomIntBetween(Math.min(nVectors, 100), nVectors);
        int maxIterations = randomIntBetween(1, 20);
        int clustersPerNeighborhood = randomIntBetween(2, 512);
        float soarLambda = randomFloat() * 1.0f + 0.5f;

        ClusteringByteVectorValues vectors = generateByteData(nVectors, dims, nClusters);

        int targetSize = nVectors / nClusters;
        HierarchicalKMeans<byte[]> hkmeans = HierarchicalKMeans.ofSerial(
            CentroidOps.BYTE,
            dims,
            maxIterations,
            sampleSize,
            clustersPerNeighborhood,
            soarLambda
        );

        KMeansResult<byte[]> result = hkmeans.cluster(vectors, targetSize);

        byte[][] centroids = result.centroids();
        int[] assignments = result.assignments();
        int[] soarAssignments = result.soarAssignments();

        // Should produce roughly the expected number of clusters
        assertTrue("Expected at least 1 centroid", centroids.length >= 1);
        assertTrue("Expected at most " + nClusters + " centroids but got " + centroids.length, centroids.length <= nClusters + 25);
        assertEquals(nVectors, assignments.length);

        // All assignments valid
        for (int assignment : assignments) {
            assertTrue(assignment >= 0 && assignment < centroids.length);
        }

        // Verify cluster counts
        int[] counts = new int[centroids.length];
        for (int a : assignments) {
            counts[a]++;
        }
        for (int count : counts) {
            assertTrue("Empty cluster found", count > 0);
        }

        // SOAR assignments valid
        if (centroids.length > 1 && centroids.length < nVectors) {
            assertEquals(nVectors, soarAssignments.length);
            for (int i = 0; i < assignments.length; i++) {
                int soar = soarAssignments[i];
                assertTrue(soar == NO_SOAR_ASSIGNMENT || (soar >= 0 && soar < centroids.length));
                assertNotEquals(assignments[i], soar);
            }
        }

        // Centroids have correct dimensions
        for (byte[] centroid : centroids) {
            assertEquals(dims, centroid.length);
        }
    }

    public void testBalancedOTKMeansLocalByte() throws IOException {
        int nClusters = randomIntBetween(2, 10);
        int nVectors = nClusters * randomIntBetween(50, 200);
        int dims = randomIntBetween(4, 32);
        int sampleSize = randomIntBetween(100, nVectors);
        int maxIterations = randomIntBetween(2, 20);

        ClusteringByteVectorValues vectors = generateByteData(nVectors, dims, nClusters);

        byte[][] centroids = KMeansLocal.pickInitialCentroids(vectors, nClusters, CentroidOps.BYTE);
        int[] assignments = new int[nVectors];
        KMeansIntermediate<byte[]> kMeansIntermediate = new KMeansIntermediate<>(centroids, assignments);

        KMeansLocal<byte[]> kMeansLocal = new BalancedOTKMeansLocalSerial<>(CentroidOps.BYTE, sampleSize, maxIterations);
        kMeansLocal.cluster(vectors, kMeansIntermediate, nClusters, -1f);

        // All assignments valid
        for (int a : kMeansIntermediate.assignments()) {
            assertTrue("Invalid assignment: " + a, a >= 0 && a < centroids.length);
        }

        // Verify total assignment count
        int[] counts = new int[centroids.length];
        for (int a : kMeansIntermediate.assignments()) {
            counts[a]++;
        }
        int totalCount = 0;
        for (int count : counts) {
            totalCount += count;
        }
        assertEquals(nVectors, totalCount);
    }

    public void testLloydKMeansLocalByte() throws IOException {
        int nClusters = randomIntBetween(2, 8);
        int nVectors = nClusters * randomIntBetween(50, 200);
        int dims = randomIntBetween(4, 32);
        int sampleSize = randomIntBetween(100, nVectors);
        int maxIterations = randomIntBetween(2, 20);

        ClusteringByteVectorValues vectors = generateByteData(nVectors, dims, nClusters);

        byte[][] centroids = KMeansLocal.pickInitialCentroids(vectors, nClusters, CentroidOps.BYTE);

        LloydKMeansLocal.cluster(vectors, CentroidOps.BYTE, centroids, sampleSize, maxIterations);

        // Centroids should have been updated (not all zeros unless input was zeros)
        boolean anyNonZero = false;
        for (byte[] centroid : centroids) {
            for (byte b : centroid) {
                if (b != 0) {
                    anyNonZero = true;
                    break;
                }
            }
        }
        assertTrue("Centroids should have non-zero values for non-zero input", anyNonZero);
    }

    public void testByteKMeansClusterQuality() throws IOException {
        // Verifies that byte clustering produces reasonable assignments:
        // vectors near the same true centroid should mostly get the same assignment.
        int nClusters = 4;
        int nVectors = nClusters * 200;
        int dims = 16;

        // Generate well-separated clusters
        byte[][] trueCentroids = new byte[nClusters][dims];
        for (int i = 0; i < nClusters; i++) {
            for (int j = 0; j < dims; j++) {
                // Spread centroids far apart
                trueCentroids[i][j] = (byte) (((i * 64) - 128 + randomIntBetween(-5, 5)) & 0xFF);
            }
        }

        List<byte[]> vectorList = new ArrayList<>(nVectors);
        int[] trueLabels = new int[nVectors];
        for (int i = 0; i < nVectors; i++) {
            int cluster = i % nClusters;
            trueLabels[i] = cluster;
            byte[] vector = new byte[dims];
            for (int j = 0; j < dims; j++) {
                vector[j] = (byte) Math.clamp(trueCentroids[cluster][j] + randomIntBetween(-3, 3), -128, 127);
            }
            vectorList.add(vector);
        }

        ClusteringByteVectorValues vectors = ClusteringByteVectorValues.build(vectorList, null, dims);

        HierarchicalKMeans<byte[]> hkmeans = HierarchicalKMeans.ofSerial(CentroidOps.BYTE, dims, 10, nVectors, 512, 1.0f);
        KMeansResult<byte[]> result = hkmeans.cluster(vectors, nVectors / nClusters);

        int[] assignments = result.assignments();

        // Check that vectors in the same true cluster mostly get the same assignment
        // Build mapping from true cluster -> most common assignment
        int[][] assignmentCounts = new int[nClusters][result.centroids().length];
        for (int i = 0; i < nVectors; i++) {
            assignmentCounts[trueLabels[i]][assignments[i]]++;
        }

        int correctCount = 0;
        for (int c = 0; c < nClusters; c++) {
            int maxCount = Arrays.stream(assignmentCounts[c]).max().orElse(0);
            correctCount += maxCount;
        }
        // Expect at least 70% of vectors assigned to their correct cluster
        float accuracy = (float) correctCount / nVectors;
        assertTrue("Byte KMeans accuracy too low: " + accuracy, accuracy >= 0.7f);
    }

    public void testRemoveEmptyClustersWithByte() throws IOException {
        // Test that removeEmptyClusters works correctly when V=byte[]
        // This exercises the fix for the ClassCastException
        int dims = 8;
        int nVectors = 100;

        // Create vectors that naturally cluster into 2 groups, but request more clusters
        List<byte[]> vectorList = new ArrayList<>(nVectors);
        for (int i = 0; i < nVectors; i++) {
            byte[] vector = new byte[dims];
            byte base = (i < nVectors / 2) ? (byte) -50 : (byte) 50;
            for (int j = 0; j < dims; j++) {
                vector[j] = (byte) Math.clamp(base + randomIntBetween(-5, 5), -128, 127);
            }
            vectorList.add(vector);
        }

        ClusteringByteVectorValues vectors = ClusteringByteVectorValues.build(vectorList, null, dims);

        // Request more clusters than natural groups — some should end up empty and get removed
        int targetSize = nVectors / 10; // aim for ~10 clusters but data only has 2 natural ones
        HierarchicalKMeans<byte[]> hkmeans = HierarchicalKMeans.ofSerial(CentroidOps.BYTE, dims, 5, nVectors, 512, -1f);
        KMeansResult<byte[]> result = hkmeans.cluster(vectors, targetSize);

        // Should not throw ClassCastException
        assertNotNull(result);
        assertTrue(result.centroids().length > 0);

        // All assignments valid
        for (int a : result.assignments()) {
            assertTrue(a >= 0 && a < result.centroids().length);
        }
    }

    private static ClusteringByteVectorValues generateByteData(int nSamples, int nDims, int nClusters) {
        List<byte[]> vectors = new ArrayList<>(nSamples);
        byte[][] centroids = new byte[nClusters][nDims];
        // Generate random centroids spread across byte range
        for (int i = 0; i < nClusters; i++) {
            for (int j = 0; j < nDims; j++) {
                centroids[i][j] = (byte) randomIntBetween(-128, 127);
            }
        }
        // Generate data points around centroids
        for (int i = 0; i < nSamples; i++) {
            int cluster = randomInt(nClusters - 1);
            byte[] vector = new byte[nDims];
            for (int j = 0; j < nDims; j++) {
                vector[j] = (byte) Math.clamp(centroids[cluster][j] + randomIntBetween(-10, 10), -128, 127);
            }
            vectors.add(vector);
        }
        return ClusteringByteVectorValues.build(vectors, null, nDims);
    }
}
