/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;
import org.junit.Before;

import java.io.IOException;

import static com.carrotsearch.randomizedtesting.RandomizedTest.randomFloat;

public class IVFKnnFloatSlicedVectorQueryTests extends AbstractIVFKnnVectorQueryTestCase {

    private static final String SLICE_FIELD = "_slice";
    private int numSlices;
    private BytesRef querySlice;

    @Override
    IVFKnnFloatVectorQuery getKnnVectorQuery(String field, float[] query, int k, Query queryFilter, float visitRatio) {
        return new IVFKnnFloatSlicedVectorQuery(field, query, k, k, queryFilter, visitRatio, testResolver(), SLICE_FIELD, querySlice);
    }

    @Override
    IVFKnnFloatVectorQuery getStableKnnVectorQuery(String field, float[] query, int k, Query queryFilter, float visitRatio) {
        return new IVFKnnFloatVectorQuery(field, query, k, k, queryFilter, visitRatio, testResolver());
    }

    @Before
    @Override
    public void setUp() throws Exception {
        super.setUp();
        format = new ESNextDiskBBQVectorsFormat(128, 4, SLICE_FIELD);
        // only one slice so it behaves as a normal index
        this.numSlices = 1;
        querySlice = new BytesRef("" + 0);
    }

    @Override
    float[] randomVector(int dim) {
        float[] vector = new float[dim];
        for (int i = 0; i < dim; i++) {
            vector[i] = randomFloat();
        }
        VectorUtil.l2normalize(vector);
        return vector;
    }

    @Override
    Field getKnnVectorField(String name, float[] vector, VectorSimilarityFunction similarityFunction) {
        return new KnnFloatVectorField(name, vector, similarityFunction);
    }

    @Override
    Field getKnnVectorField(String name, float[] vector) {
        return new KnnFloatVectorField(name, vector);
    }

    public void testToString() throws IOException {
        try (
            Directory indexStore = getIndexStore("field", new float[] { 0, 1 }, new float[] { 1, 2 }, new float[] { 0, 0 });
            IndexReader reader = DirectoryReader.open(indexStore)
        ) {
            AbstractIVFKnnVectorQuery query = getKnnVectorQuery("field", new float[] { 0.0f, 1.0f }, 10);
            assertEquals("IVFKnnFloatSlicedVectorQuery:field[0.0,...][10][" + SLICE_FIELD + "=[0]]", query.toString("ignored"));

            assertDocScoreQueryToString(query.rewrite(newSearcher(reader)));

            // test with filter
            Query filter = new TermQuery(new Term("id", "text"));
            query = getKnnVectorQuery("field", new float[] { 0.0f, 1.0f }, 10, filter);
            assertEquals("IVFKnnFloatSlicedVectorQuery:field[0.0,...][10][" + SLICE_FIELD + "=[0]][id:text]", query.toString("ignored"));
        }
    }

    /**
     * Unlike {@link AbstractIVFKnnVectorQueryTestCase#testBitSetQuery()}, sliced search intersects the
     * filter iterator with the slice doc-id range before materializing accept bits, so it never calls
     * {@link org.apache.lucene.util.BitSetIterator#getBitSet()} and supports filters that do not allow
     * bitset reuse.
     */
    @Override
    public void testBitSetQuery() throws IOException {
        IndexWriterConfig iwc = newIndexWriterConfig();
        decorateIWC(iwc);
        try (Directory dir = newDirectoryForTest(); IndexWriter w = new IndexWriter(dir, iwc)) {
            final int numDocs = 100;
            final int dim = 30;
            for (int i = 0; i < numDocs; ++i) {
                Document d = getDocumentToIndex();
                d.add(getKnnVectorField("vector", randomVector(dim)));
                w.addDocument(d);
            }
            w.commit();

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                // Same fixture as AbstractIVFKnnVectorQueryTestCase#testBitSetQuery(): an empty FixedBitSet
                // (all bits clear) behind a BitSetIterator; non-sliced search fails fatally in getBitSet().
                Query filter = new ThrowingBitSetQuery(new FixedBitSet(numDocs));
                TopDocs topDocs = searcher.search(getKnnVectorQuery("vector", randomVector(dim), 10, filter), numDocs);
                assertEquals(0, topDocs.scoreDocs.length);
            }
        }
    }

    @Override
    protected Document getDocumentToIndex() {
        Document doc = new Document();
        doc.add(SortedDocValuesField.indexedField(SLICE_FIELD, new BytesRef("" + random().nextInt(numSlices))));
        return doc;
    }

    @Override
    protected void decorateIWC(IndexWriterConfig indexWriterConfig) {
        indexWriterConfig.setIndexSort(new Sort(new SortField(SLICE_FIELD, SortField.Type.STRING)));
    }

}
