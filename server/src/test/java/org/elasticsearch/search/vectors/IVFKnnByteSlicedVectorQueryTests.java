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
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
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
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.cache.query.TrivialQueryCachingPolicy;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.function.BooleanSupplier;

import static org.hamcrest.Matchers.equalTo;

public class IVFKnnByteSlicedVectorQueryTests extends LuceneTestCase {

    private static final String SLICE_FIELD = "_slice";

    static {
        LogConfigurator.configureESLogging();
    }

    private ESNextDiskBBQVectorsFormat format;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        format = new ESNextDiskBBQVectorsFormat(128, 4, SLICE_FIELD);
    }

    private byte[] randomByteVector(int dim) {
        byte[] vector = new byte[dim];
        random().nextBytes(vector);
        return vector;
    }

    public void testToString() throws IOException {
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setIndexSort(new Sort(new SortField(SLICE_FIELD, SortField.Type.STRING)));
        iwc.setCodec(TestUtil.alwaysKnnVectorsFormat(format));
        try (Directory dir = newDirectory(); IndexWriter w = new IndexWriter(dir, iwc)) {
            Document doc = new Document();
            doc.add(SortedDocValuesField.indexedField(SLICE_FIELD, new BytesRef("0")));
            doc.add(new KnnByteVectorField("field", new byte[] { 0, 1 }, VectorSimilarityFunction.DOT_PRODUCT));
            w.addDocument(doc);
            w.commit();

            try (IndexReader reader = DirectoryReader.open(dir)) {
                BytesRef querySlice = new BytesRef("0");
                IVFKnnByteSlicedVectorQuery query = new IVFKnnByteSlicedVectorQuery(
                    "field",
                    new byte[] { 0, 1 },
                    10,
                    10,
                    null,
                    1.0f,
                    random().nextBoolean(),
                    SLICE_FIELD,
                    querySlice
                );
                assertEquals("IVFKnnByteSlicedVectorQuery:field[0,...][10][" + SLICE_FIELD + "=0]", query.toString("ignored"));

                // test with filter
                Query filter = new TermQuery(new Term("id", "text"));
                query = new IVFKnnByteSlicedVectorQuery(
                    "field",
                    new byte[] { 0, 1 },
                    10,
                    10,
                    filter,
                    1.0f,
                    random().nextBoolean(),
                    SLICE_FIELD,
                    querySlice
                );
                assertEquals("IVFKnnByteSlicedVectorQuery:field[0,...][10][" + SLICE_FIELD + "=0][id:text]", query.toString("ignored"));
            }
        }
    }

    public void testSlicesDense() throws IOException {
        doTestSlices(() -> true, false);
    }

    public void testSlicesDenseWithFilter() throws IOException {
        doTestSlices(() -> true, true);
    }

    public void testSlicesSparse() throws IOException {
        if (rarely()) {
            doTestSlices(() -> random().nextInt(1000) == 0, false);
        } else {
            int bound = random().nextInt(2, 50);
            doTestSlices(() -> random().nextInt(bound) == 0, false);
        }
    }

    public void testSlicesSparseWithFilter() throws IOException {
        if (rarely()) {
            doTestSlices(() -> random().nextInt(1000) == 0, true);
        } else {
            int bound = random().nextInt(2, 50);
            doTestSlices(() -> random().nextInt(bound) == 0, true);
        }
    }

    private void doTestSlices(BooleanSupplier hasVectorSupplier, boolean applyFilter) throws IOException {
        int dimensions = random().nextInt(12, 500);
        int numDocs = random().nextInt(100, 10_000);
        int numSlices = random().nextInt(1, numDocs);
        int[] docsPerSlice = new int[numSlices];
        int[] docsPerSliceFiltered = new int[numSlices];
        int[] docSlices = new int[numDocs];
        boolean[] docHasVector = new boolean[numDocs];
        boolean[] docFilterMatch = new boolean[numDocs];
        String filterField = "_filter";
        String filterValue = "match";
        String filterMiss = "miss";
        String docIdField = "_doc_id";
        IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setIndexSort(new Sort(new SortField(SLICE_FIELD, SortField.Type.STRING)));
        iwc.setCodec(TestUtil.alwaysKnnVectorsFormat(format));

        try (Directory dir = newDirectory(); IndexWriter w = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < numDocs; i++) {
                int slice = random().nextInt(numSlices);
                Document doc = new Document();
                doc.add(SortedDocValuesField.indexedField(SLICE_FIELD, new BytesRef("" + slice)));
                boolean filterMatch = random().nextBoolean();
                String filterText = filterMatch ? filterValue : filterMiss;
                doc.add(new StringField(filterField, filterText, Field.Store.NO));
                doc.add(new StoredField(filterField, new BytesRef(filterText)));
                doc.add(new StringField(docIdField, "doc_" + i, Field.Store.NO));
                boolean hasVector = hasVectorSupplier.getAsBoolean();
                if (hasVector) {
                    docsPerSlice[slice]++;
                    if (filterMatch) {
                        docsPerSliceFiltered[slice]++;
                    }
                    doc.add(new KnnByteVectorField("vector", randomByteVector(dimensions), VectorSimilarityFunction.DOT_PRODUCT));
                }
                doc.add(new StoredField(SLICE_FIELD, new BytesRef("" + slice)));
                w.addDocument(doc);
                docSlices[i] = slice;
                docHasVector[i] = hasVector;
                docFilterMatch[i] = filterMatch;
            }
            w.commit();
            if (random().nextBoolean()) {
                int deleteCount = random().nextInt(0, Math.max(1, numDocs / 10));
                Set<Integer> docsToDelete = new HashSet<>();
                while (docsToDelete.size() < deleteCount) {
                    docsToDelete.add(random().nextInt(numDocs));
                }
                for (int docId : docsToDelete) {
                    if (docHasVector[docId]) {
                        docsPerSlice[docSlices[docId]]--;
                        if (docFilterMatch[docId]) {
                            docsPerSliceFiltered[docSlices[docId]]--;
                        }
                    }
                    w.deleteDocuments(new Term(docIdField, "doc_" + docId));
                }
                if (docsToDelete.isEmpty() == false) {
                    w.commit();
                }
            } else if (random().nextBoolean()) {
                w.forceMerge(1);
            }
            byte[] queryVector = randomByteVector(dimensions);
            try (IndexReader reader = DirectoryReader.open(w)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                searcher.setQueryCachingPolicy(TrivialQueryCachingPolicy.ALWAYS);
                Query filterQuery = null;
                if (applyFilter) {
                    filterQuery = new TermQuery(new Term(filterField, filterValue));
                }
                for (int iters = 0; iters < 2; iters++) {
                    for (int slice = 0; slice < numSlices; slice++) {
                        int expectedDocs = applyFilter ? docsPerSliceFiltered[slice] : docsPerSlice[slice];
                        int k = 2 * Math.max(1, expectedDocs);
                        Query kvq = new IVFKnnByteSlicedVectorQuery(
                            "vector",
                            queryVector,
                            k,
                            k,
                            filterQuery,
                            1.0f,
                            random().nextBoolean(),
                            SLICE_FIELD,
                            new BytesRef("" + slice)
                        );
                        TopDocs topDocs = searcher.search(kvq, k);
                        assertEquals(expectedDocs, topDocs.scoreDocs.length);
                        for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                            Document document = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                            assertThat(document.getField(SLICE_FIELD).binaryValue().utf8ToString(), equalTo("" + slice));
                            if (applyFilter) {
                                assertThat(document.getField(filterField).binaryValue().utf8ToString(), equalTo(filterValue));
                            }
                        }
                    }
                    // Query a non-existent slice
                    Query kvq = new IVFKnnByteSlicedVectorQuery(
                        "vector",
                        queryVector,
                        3,
                        3,
                        filterQuery,
                        1.0f,
                        random().nextBoolean(),
                        SLICE_FIELD,
                        new BytesRef("invalid")
                    );
                    TopDocs topDocs = searcher.search(kvq, 3);
                    assertEquals(0, topDocs.scoreDocs.length);
                }
            }
        }
    }
}
