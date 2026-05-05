/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.common.lucene.Lucene;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.VectorPreconditioner;

import java.io.IOException;
import java.util.Arrays;

/**
 * An IVF kNN query for byte-encoded vector fields. For COSINE similarity, the query vector is
 * normalized to float[] in {@link #preconditionQuery} so the search always routes through the
 * float path. For non-COSINE, the raw byte[] is passed directly to the codec.
 */
public class IVFKnnByteVectorQuery extends AbstractIVFKnnVectorQuery {

    private final byte[] query;
    private boolean isQueryPreconditioned = false;
    private float[] preconditionedQuery = null;

    /**
     * Creates a new {@link IVFKnnByteVectorQuery}.
     * @param field the field to search
     * @param query the byte query vector
     * @param k the number of nearest neighbors to return
     * @param numCands the number of nearest neighbors to gather per shard
     * @param filter the filter to apply to the results
     * @param visitRatio the ratio of vectors to score for the IVF search strategy
     * @param doPrecondition whether to apply preconditioning
     */
    public IVFKnnByteVectorQuery(String field, byte[] query, int k, int numCands, Query filter, float visitRatio, boolean doPrecondition) {
        super(field, visitRatio, k, numCands, filter, doPrecondition);
        this.query = query;
    }

    public byte[] getQuery() {
        return query;
    }

    @Override
    public String toString(String field) {
        StringBuilder buffer = new StringBuilder();
        buffer.append(getClass().getSimpleName())
            .append(":")
            .append(this.field)
            .append("[")
            .append(query[0])
            .append(",...]")
            .append("[")
            .append(k)
            .append("]");
        if (this.filter != null) {
            buffer.append("[").append(this.filter).append("]");
        }
        return buffer.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (super.equals(o) == false) return false;
        IVFKnnByteVectorQuery that = (IVFKnnByteVectorQuery) o;
        return Arrays.equals(query, that.query);
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + Arrays.hashCode(query);
        return result;
    }

    @Override
    protected void preconditionQuery(LeafReaderContext context) throws IOException {
        if (isQueryPreconditioned) {
            return;
        }
        LeafReader reader = context.reader();
        FieldInfo fieldInfo = reader.getFieldInfos().fieldInfo(field);
        if (fieldInfo == null) {
            return;
        }

        // Attempt preconditioning via the segment's VectorPreconditioner
        SegmentReader segmentReader = Lucene.tryUnwrapSegmentReader(reader);
        if (segmentReader != null) {
            KnnVectorsReader fieldsReader = segmentReader.getVectorReader();
            if (fieldsReader instanceof PerFieldKnnVectorsFormat.FieldsReader) {
                KnnVectorsReader knnVectorsReader = ((PerFieldKnnVectorsFormat.FieldsReader) fieldsReader).getFieldReader(field);
                if (knnVectorsReader instanceof VectorPreconditioner) {
                    Preconditioner preconditioner = ((VectorPreconditioner) knnVectorsReader).getPreconditioner(fieldInfo);
                    if (preconditioner != null) {
                        float[] out = new float[query.length];
                        if (fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE) {
                            float[] floatQuery = new float[query.length];
                            for (int i = 0; i < query.length; i++) {
                                floatQuery[i] = query[i];
                            }
                            VectorUtil.l2normalize(floatQuery);
                            preconditioner.applyTransform(floatQuery, out);
                        } else {
                            preconditioner.applyTransform(query, out);
                        }
                        preconditionedQuery = out;
                        isQueryPreconditioned = true;
                        return;
                    }
                }
            }
        }

        // For COSINE, convert byte to float and normalize so that approximateSearch routes through the float[]
        // search path (avoiding redundant conversion in IVFVectorsReader)
        if (fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE) {
            float[] normalized = new float[query.length];
            for (int i = 0; i < query.length; i++) {
                normalized[i] = query[i];
            }
            VectorUtil.l2normalize(normalized);
            preconditionedQuery = normalized;
            isQueryPreconditioned = true;
        }
    }

    @Override
    protected TopDocs approximateSearch(
        LeafReaderContext context,
        AcceptDocs acceptDocs,
        int visitedLimit,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) throws IOException {
        LeafReader reader = context.reader();
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(visitRatio, numCands, k, knnCollectorManager.longAccumulator);
        AbstractMaxScoreKnnCollector knnCollector = knnCollectorManager.newCollector(visitedLimit, strategy, context);
        if (knnCollector == null) {
            return NO_RESULTS;
        }
        strategy.setCollector(knnCollector);
        if (preconditionedQuery != null) {
            reader.searchNearestVectors(field, preconditionedQuery, knnCollector, acceptDocs);
        } else {
            reader.searchNearestVectors(field, query, knnCollector, acceptDocs);
        }
        TopDocs results = knnCollector instanceof BulkKnnCollector bulkKnnCollector
            ? bulkKnnCollector.unsortedTopK()
            : knnCollector.topDocs();
        return results != null ? results : NO_RESULTS;
    }
}
