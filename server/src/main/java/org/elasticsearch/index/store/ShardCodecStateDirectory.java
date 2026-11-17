/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.store;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.elasticsearch.common.util.FeatureFlag;
import org.elasticsearch.core.Nullable;

import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * A {@link FilterDirectory} that holds state a codec shares across the segments of one shard, for codecs whose segments
 * are better off agreeing on something they would otherwise each derive for themselves.
 * <p>
 * Lucene builds the codec that reads a segment by name, so a reader has no handle on the shard it belongs to other than its
 * {@code SegmentInfo.dir}, which is the shard's {@code Store} directory chain for every segment written and read by that
 * shard. A codec reaches this state through {@link #unwrap(Directory)} on that directory, from both its writers (flush and
 * merge) and its readers. Like {@link FieldInfoCachingDirectory}, it sits inside the {@code Store}'s chain so that
 * {@code store.directory()} keeps returning the same {@code StoreDirectory}.
 * <p>
 * The state must only ever be a hint: segments stay self-describing, and a codec must behave correctly, if less
 * efficiently, when the directory is not wrapped (tests, other directory chains) or the state is empty (a restarted node).
 */
public final class ShardCodecStateDirectory extends ByteSizeDirectory {

    public static final FeatureFlag FEATURE_FLAG = new FeatureFlag("shard_codec_state_directory");

    private final ConcurrentHashMap<Class<?>, Object> states = new ConcurrentHashMap<>();

    public ShardCodecStateDirectory(Directory delegate) {
        super(delegate);
    }

    /**
     * Wraps the given directory with a {@link ShardCodecStateDirectory} if the feature flag is enabled and the directory is
     * not already wrapped. Otherwise returns the directory unchanged. Idempotent.
     */
    public static Directory wrapIfEnabled(Directory directory) {
        if (FEATURE_FLAG.isEnabled() == false) {
            return directory;
        }
        if (unwrap(directory) != null) {
            return directory;
        }
        return new ShardCodecStateDirectory(directory);
    }

    /**
     * Walks the {@link FilterDirectory} chain and returns the first {@link ShardCodecStateDirectory}, or {@code null} if
     * none is present.
     */
    @Nullable
    public static ShardCodecStateDirectory unwrap(Directory directory) {
        Directory d = directory;
        while (d != null) {
            if (d instanceof ShardCodecStateDirectory cd) {
                return cd;
            }
            if (d instanceof FilterDirectory fd) {
                d = fd.getDelegate();
            } else {
                return null;
            }
        }
        return null;
    }

    /**
     * Returns the shard's state of the given type, creating it with {@code factory} on first use. The state lives as long
     * as the shard's {@code Store}.
     */
    public <T> T getOrCreate(Class<T> type, Supplier<T> factory) {
        return type.cast(states.computeIfAbsent(type, k -> factory.get()));
    }

    @Override
    public long estimateSizeInBytes() throws IOException {
        Directory delegate = getDelegate();
        if (delegate instanceof ByteSizeDirectory bsd) {
            return bsd.estimateSizeInBytes();
        }
        return estimateSizeInBytes(delegate);
    }

    @Override
    public long estimateDataSetSizeInBytes() throws IOException {
        Directory delegate = getDelegate();
        if (delegate instanceof ByteSizeDirectory bsd) {
            return bsd.estimateDataSetSizeInBytes();
        }
        return estimateSizeInBytes(delegate);
    }
}
