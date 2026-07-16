/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.index.codec.vectors.diskbbq;

import org.elasticsearch.test.ESTestCase;

/**
 * Tests for {@link CentroidAssignments} and {@link CentroidInformation}.
 */
public class CentroidAssignmentsTests extends ESTestCase {

    public void testBasicCentroidInformation() {
        int dims = 4;
        float[][] centroids = new float[][] { { 1, 2, 3, 4 }, { -1, -2, -3, -4 }, { 50, 60, 70, 80 } };
        int[] assignments = new int[] { 0, 1, 2, 0, 1 };
        int[] overspill = new int[] { 1, 0, 1, 2, 0 };

        CentroidInformation<float[]> ci = new CentroidInformation<>(dims, centroids, assignments, new SoarAssignments(overspill));

        assertEquals(3, ci.numCentroids());
        assertSame(centroids, ci.centroids());
        assertSame(assignments, ci.assignments());
    }

    public void testGlobalCentroid() {
        int dims = 3;
        float[][] centroids = new float[][] { { 10, 20, 30 }, { -10, -20, -30 } };
        int[] assignments = new int[] { 0, 1 };

        CentroidInformation<float[]> ci = new CentroidInformation<>(dims, centroids, assignments, OverspillAssignments.NONE);

        float[] globalCentroid = ci.globalCentroid();
        assertNotNull(globalCentroid);
        assertEquals(dims, globalCentroid.length);
        // Global centroid = mean of centroids: (10 + -10)/2 = 0, (20 + -20)/2 = 0, (30 + -30)/2 = 0
        assertEquals(0.0f, globalCentroid[0], 1e-5f);
        assertEquals(0.0f, globalCentroid[1], 1e-5f);
        assertEquals(0.0f, globalCentroid[2], 1e-5f);
    }

    public void testGlobalCentroidAsymmetric() {
        int dims = 2;
        float[][] centroids = new float[][] { { 10, 20 }, { 30, 40 }, { 50, 60 } };
        int[] assignments = new int[] { 0, 1, 2 };

        CentroidInformation<float[]> ci = new CentroidInformation<>(dims, centroids, assignments, OverspillAssignments.NONE);

        float[] globalCentroid = ci.globalCentroid();
        // Mean: (10+30+50)/3 = 30, (20+40+60)/3 = 40
        assertEquals(30.0f, globalCentroid[0], 1e-5f);
        assertEquals(40.0f, globalCentroid[1], 1e-5f);
    }

    public void testEmptyCentroids() {
        int dims = 4;
        float[][] centroids = new float[0][];
        int[] assignments = new int[0];

        CentroidInformation<float[]> ci = new CentroidInformation<>(dims, centroids, assignments, OverspillAssignments.NONE);

        assertEquals(0, ci.numCentroids());
        assertEquals(0, ci.centroids().length);
    }

    public void testSingleCentroid() {
        int dims = 5;
        float[][] centroids = new float[][] { { 10, 20, 30, 40, 50 } };
        int[] assignments = new int[] { 0, 0, 0 };

        CentroidInformation<float[]> ci = new CentroidInformation<>(dims, centroids, assignments, OverspillAssignments.NONE);

        assertEquals(1, ci.numCentroids());
        // Global centroid = the single centroid itself
        float[] globalCentroid = ci.globalCentroid();
        assertEquals(10.0f, globalCentroid[0], 1e-5f);
        assertEquals(50.0f, globalCentroid[4], 1e-5f);
    }

    public void testWithCentroidSlices() {
        int dims = 2;
        float[][] centroids = new float[][] { { 1, 2 }, { 3, 4 } };
        int[] assignments = new int[] { 0, 0, 1, 1 };
        int[] overspill = new int[] { 1, 1, 0, 0 };
        CentroidSlices slices = new CentroidSlices(new int[] { 0, 2 }, new int[] { 2, 2 });

        CentroidInformation<float[]> ci = new CentroidInformation<>(dims, centroids, assignments, new SoarAssignments(overspill), slices);

        assertEquals(2, ci.numCentroids());
        assertSame(slices, ci.centroidAssignments().centroidSlices());
    }

    public void testCentroidAssignmentsRecord() {
        int numCentroids = 3;
        int[] assignments = new int[] { 0, 1, 2, 0, 1 };
        float[] globalCentroid = new float[] { 1.0f, 2.0f, 3.0f };

        CentroidAssignments ca = new CentroidAssignments(numCentroids, assignments, OverspillAssignments.NONE, globalCentroid);

        assertEquals(numCentroids, ca.numCentroids());
        assertSame(assignments, ca.assignments());
        assertSame(globalCentroid, ca.globalCentroid());
        assertNull(ca.centroidSlices());
    }
}
