/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.index.FloatVectorValues;

import java.io.IOException;

/**
 * Sealed interface representing vector values for IVF posting-list writing.
 * <p>
 * The byte-vs-float decision is made once at construction so that downstream
 * quantization code can pattern-match instead of probing at runtime.
 */
public sealed interface WriterVectorValues {

    /**
     * Maps a vector ordinal to its document ID.
     */
    int ordToDoc(int ord);

    /**
     * Returns the float vector for the given ordinal. Always available regardless
     * of the underlying encoding.
     */
    float[] vectorValue(int ord) throws IOException;

    /**
     * Float-backed vector values. Used for float element_type fields and for
     * COSINE byte fields whose vectors have been L2-normalized to floats.
     */
    record FloatValues(FloatVectorValues values) implements WriterVectorValues {
        @Override
        public int ordToDoc(int ord) {
            return values.ordToDoc(ord);
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            return values.vectorValue(ord);
        }
    }
}
