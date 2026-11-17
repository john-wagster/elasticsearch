/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.ash;

import org.apache.lucene.store.Directory;
import org.elasticsearch.index.store.ShardCodecStateDirectory;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.UUID;

import static org.hamcrest.Matchers.sameInstance;

public class AshProjectionRegistryTests extends ESTestCase {

    private static final int REQUIRED = 100;

    private static AshProjectionMatrix matrix(int originalDim, int nDims, boolean learned, int sampleSize) {
        return new AshProjectionMatrix(new float[originalDim * nDims], originalDim, nDims, learned, UUID.randomUUID(), sampleSize);
    }

    public void testShareableMatrixBecomesShared() {
        AshProjectionRegistry registry = new AshProjectionRegistry();
        assertNull(registry.shared("f", 8, 4));
        AshProjectionMatrix m = matrix(8, 4, true, REQUIRED);
        assertThat(registry.offer("f", m, REQUIRED), sameInstance(m));
        assertThat(registry.shared("f", 8, 4), sameInstance(m));
        assertNull("other fields have their own shared matrix", registry.shared("g", 8, 4));
        assertNull("a matrix of another shape is not returned", registry.shared("f", 8, 2));
    }

    public void testMatrixFromPartialSampleIsNotShared() {
        AshProjectionRegistry registry = new AshProjectionRegistry();
        registry.offer("f", matrix(8, 4, true, REQUIRED - 1), REQUIRED);
        registry.offer("f", matrix(8, 4, false, REQUIRED), REQUIRED);
        assertNull(registry.shared("f", 8, 4));
    }

    public void testSharedMatrixIsKept() {
        AshProjectionRegistry registry = new AshProjectionRegistry();
        AshProjectionMatrix first = matrix(8, 4, true, REQUIRED);
        registry.offer("f", first, REQUIRED);
        registry.offer("f", matrix(8, 4, true, REQUIRED * 10), REQUIRED);
        assertThat("replacing the shared matrix would split the shard's segments again", registry.shared("f", 8, 4), sameInstance(first));
    }

    public void testSharedMatrixOfAnotherShapeIsReplaced() {
        AshProjectionRegistry registry = new AshProjectionRegistry();
        registry.offer("f", matrix(8, 4, true, REQUIRED), REQUIRED);
        AshProjectionMatrix reshaped = matrix(8, 2, true, REQUIRED);
        registry.offer("f", reshaped, REQUIRED);
        assertThat(registry.shared("f", 8, 2), sameInstance(reshaped));
        assertNull(registry.shared("f", 8, 4));
    }

    public void testClaimAdoptsMatrixSharedWhileTraining() {
        AshProjectionRegistry registry = new AshProjectionRegistry();
        // nothing shared yet: a full-sample matrix is used and becomes shared
        AshProjectionMatrix first = matrix(8, 4, true, REQUIRED);
        assertThat(registry.claim("f", first, REQUIRED), sameInstance(first));
        assertThat(registry.shared("f", 8, 4), sameInstance(first));
        // a concurrent flush or merge that learned its own matrix meanwhile switches to the shared one
        AshProjectionMatrix concurrent = matrix(8, 4, true, REQUIRED * 2);
        assertThat(registry.claim("f", concurrent, REQUIRED), sameInstance(first));
        // even if it learned from a partial sample
        assertThat(registry.claim("f", matrix(8, 4, true, 1), REQUIRED), sameInstance(first));
        assertThat(registry.shared("f", 8, 4), sameInstance(first));
    }

    public void testClaimKeepsPartialSampleMatrixUnshared() {
        AshProjectionRegistry registry = new AshProjectionRegistry();
        AshProjectionMatrix partial = matrix(8, 4, true, REQUIRED - 1);
        assertThat(registry.claim("f", partial, REQUIRED), sameInstance(partial));
        assertNull(registry.shared("f", 8, 4));
        // a matrix of another shape is not adopted
        AshProjectionMatrix shared = matrix(8, 2, true, REQUIRED);
        registry.claim("f", shared, REQUIRED);
        AshProjectionMatrix otherShape = matrix(8, 4, true, 1);
        assertThat(registry.claim("f", otherShape, REQUIRED), sameInstance(otherShape));
        assertThat(registry.shared("f", 8, 2), sameInstance(shared));
    }

    public void testInternReturnsOneInstancePerId() {
        AshProjectionRegistry registry = new AshProjectionRegistry();
        UUID id = UUID.randomUUID();
        AshProjectionMatrix first = new AshProjectionMatrix(new float[32], 8, 4, true, id, REQUIRED);
        AshProjectionMatrix sameContent = new AshProjectionMatrix(new float[32], 8, 4, true, id, REQUIRED);
        AshProjectionMatrix other = matrix(8, 4, true, REQUIRED);
        assertThat(registry.intern(first), sameInstance(first));
        assertThat(registry.intern(sameContent), sameInstance(first));
        assertThat(registry.intern(other), sameInstance(other));
        assertEquals(2, registry.canonicalCount());
        // offer interns too, and hands back the canonical instance
        assertThat(registry.offer("f", sameContent, REQUIRED), sameInstance(first));
    }

    public void testForDirectory() throws IOException {
        try (Directory plain = newDirectory()) {
            assertNull(AshProjectionRegistry.forDirectory(plain));
        }
        try (Directory wrapped = new ShardCodecStateDirectory(newDirectory())) {
            AshProjectionRegistry registry = AshProjectionRegistry.forDirectory(wrapped);
            assertNotNull(registry);
            assertThat(AshProjectionRegistry.forDirectory(wrapped), sameInstance(registry));
        }
    }
}
