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
import org.elasticsearch.core.Nullable;
import org.elasticsearch.index.store.ShardCodecStateDirectory;

import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The ASH projection matrices of one shard, so that its segments share a single projection matrix W per field rather than
 * each learning its own.
 * <p>
 * Every segment stores its own W and can be read without this registry. Sharing is purely an optimization, with two
 * effects:
 * <ul>
 *     <li>Search: a query is projected once per distinct W rather than once per segment, see
 *     {@code QueryTransformCache}. The projection is a matrix-vector multiply whose cost is dominated by streaming W
 *     through the CPU caches, so with a W per segment its cost grows with the number of segments.</li>
 *     <li>Indexing: a flush or merge that reuses the shared W skips training a projection.</li>
 * </ul>
 * Only a matrix learned from a full training sample is shared, so reusing it is no worse for a segment than learning its
 * own from its own sample. Once a field has a shared matrix of a given shape, it is kept: replacing it would split the
 * shard's segments between two matrices again.
 * <p>
 * The registry also keeps one canonical in-memory copy of each distinct W (by {@link AshProjectionMatrix#id()}), so
 * segments that share W also share its heap.
 */
public final class AshProjectionRegistry {

    private final ConcurrentHashMap<String, AshProjectionMatrix> shared = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<UUID, MatrixRef> canonical = new ConcurrentHashMap<>();
    private final ReferenceQueue<AshProjectionMatrix> deadRefs = new ReferenceQueue<>();

    /**
     * Returns the registry of the shard that owns {@code directory}, or {@code null} if the directory does not carry
     * per-shard codec state, in which case every segment uses its own projection.
     */
    @Nullable
    public static AshProjectionRegistry forDirectory(Directory directory) {
        ShardCodecStateDirectory stateDirectory = ShardCodecStateDirectory.unwrap(directory);
        if (stateDirectory == null) {
            return null;
        }
        return stateDirectory.getOrCreate(AshProjectionRegistry.class, AshProjectionRegistry::new);
    }

    /**
     * Returns the shared projection matrix for {@code field} if it has the given shape, otherwise {@code null}.
     */
    @Nullable
    public AshProjectionMatrix shared(String field, int originalDim, int nDims) {
        AshProjectionMatrix matrix = shared.get(field);
        return matrix != null && matrix.hasShape(originalDim, nDims) ? matrix : null;
    }

    /**
     * Offers a projection matrix used by a segment of {@code field}. It becomes the field's shared matrix if it was learned
     * from at least {@code requiredSampleSize} vectors and the field has no shared matrix of the same shape yet.
     *
     * @return the canonical instance of {@code matrix}, to use in its place
     */
    public AshProjectionMatrix offer(String field, AshProjectionMatrix matrix, int requiredSampleSize) {
        AshProjectionMatrix instance = intern(matrix);
        if (instance.isShareable(requiredSampleSize)) {
            shared.compute(
                field,
                (f, existing) -> existing != null && existing.hasShape(instance.originalDim(), instance.nDims()) ? existing : instance
            );
        }
        return instance;
    }

    /**
     * Resolves the projection matrix a segment of {@code field} should use, having just learned {@code candidate}: the
     * field's shared matrix if it has one of the same shape, which another flush or merge may have registered while this
     * segment was training, otherwise {@code candidate}, which becomes the shared matrix if it was learned from at least
     * {@code requiredSampleSize} vectors. Called after training and before encoding, so that concurrent flushes and merges
     * converge on one matrix rather than each keeping the one it learned.
     *
     * @return the canonical instance of the matrix to use
     */
    public AshProjectionMatrix claim(String field, AshProjectionMatrix candidate, int requiredSampleSize) {
        AshProjectionMatrix instance = intern(candidate);
        AshProjectionMatrix current = shared.compute(field, (f, existing) -> {
            if (existing != null && existing.hasShape(instance.originalDim(), instance.nDims())) {
                return existing;
            }
            return instance.isShareable(requiredSampleSize) ? instance : existing;
        });
        return current != null && current.hasShape(instance.originalDim(), instance.nDims()) ? current : instance;
    }

    /**
     * Returns the canonical instance of the matrix with {@code matrix}'s id, registering {@code matrix} as that instance if
     * there is none. The canonical instances are held weakly, so they are released once no segment reader or writer uses
     * them.
     */
    public AshProjectionMatrix intern(AshProjectionMatrix matrix) {
        drainDeadRefs();
        UUID id = matrix.id();
        while (true) {
            MatrixRef existing = canonical.get(id);
            AshProjectionMatrix instance = existing == null ? null : existing.get();
            if (instance != null) {
                return instance;
            }
            MatrixRef fresh = new MatrixRef(id, matrix, deadRefs);
            if (existing == null ? canonical.putIfAbsent(id, fresh) == null : canonical.replace(id, existing, fresh)) {
                return matrix;
            }
            // Lost a race against another thread registering or replacing the instance, retry
        }
    }

    /** Number of distinct projection matrices currently held, for tests and diagnostics. */
    int canonicalCount() {
        drainDeadRefs();
        return canonical.size();
    }

    private void drainDeadRefs() {
        Reference<? extends AshProjectionMatrix> ref;
        while ((ref = deadRefs.poll()) != null) {
            MatrixRef matrixRef = (MatrixRef) ref;
            canonical.remove(matrixRef.id, matrixRef);
        }
    }

    private static final class MatrixRef extends WeakReference<AshProjectionMatrix> {
        final UUID id;

        MatrixRef(UUID id, AshProjectionMatrix matrix, ReferenceQueue<AshProjectionMatrix> queue) {
            super(matrix, queue);
            this.id = id;
        }
    }
}
