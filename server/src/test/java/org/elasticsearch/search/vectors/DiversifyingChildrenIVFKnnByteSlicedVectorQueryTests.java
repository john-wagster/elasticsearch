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
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.CheckJoinIndex;
import org.apache.lucene.search.join.QueryBitSetProducer;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.cache.query.TrivialQueryCachingPolicy;
import org.elasticsearch.index.codec.vectors.diskbbq.TestIvfQueryConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;
import org.elasticsearch.index.engine.Engine;
import org.elasticsearch.index.mapper.RoutingFieldMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BooleanSupplier;

import static org.hamcrest.Matchers.equalTo;

/**
 * Tests for {@link DiversifyingChildrenIVFKnnByteSlicedVectorQuery}.
 * <p>
 * This is a standalone test class that verifies the byte-vector diversifying-children sliced query
 * correctly handles multi-slice queries with nested (block-join) documents.
 */
public class DiversifyingChildrenIVFKnnByteSlicedVectorQueryTests extends LuceneTestCase {

    static {
        LogConfigurator.configureESLogging();
    }

    private ESNextDiskBBQVectorsFormat format;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        format = new ESNextDiskBBQVectorsFormat(128, 4, RoutingFieldMapper.NAME);
    }

    private IndexWriterConfig slicedIndexWriterConfig() {
        return newIndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(format))
            .setMergePolicy(newMergePolicy(random(), false))
            .setIndexSort(new Sort(new SortField(RoutingFieldMapper.NAME, SortField.Type.STRING)))
            .setParentField(Engine.ROOT_DOC_FIELD_NAME);
    }

    private TestIvfQueryConfigResolver testResolver() {
        return new TestIvfQueryConfigResolver(
            ESNextDiskBBQVectorsFormat.CentroidIndexFormat.FLAT,
            ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY,
            false,
            1.0f
        );
    }

    private static void addRoutingSlice(Document doc, BytesRef sliceId) {
        doc.add(SortedDocValuesField.indexedField(RoutingFieldMapper.NAME, sliceId));
    }

    private static Document makeParent(int[] children) {
        Document parent = new Document();
        parent.add(new StringField("docType", "_parent", Field.Store.NO));
        parent.add(newStringField("id", Arrays.toString(children), Field.Store.YES));
        return parent;
    }

    private static BitSetProducer parentFilter(IndexReader reader) throws IOException {
        BitSetProducer parentsFilter = new QueryBitSetProducer(new TermQuery(new Term("docType", "_parent")));
        CheckJoinIndex.check(reader, parentsFilter);
        return parentsFilter;
    }

    private byte[] randomByteVector(int dim) {
        byte[] vector = new byte[dim];
        random().nextBytes(vector);
        return vector;
    }

    public void testSlicesDense() throws IOException {
        doTestSlices(() -> true);
    }

    public void testMultiSlice() throws IOException {
        int dimensions = random().nextInt(12, 128);
        int numSlices = random().nextInt(3, 8);
        int numParents = random().nextInt(50, 200);
        int[] parentsPerSlice = new int[numSlices];

        try (Directory dir = newDirectory(); IndexWriter w = new IndexWriter(dir, slicedIndexWriterConfig())) {
            for (int i = 0; i < numParents; i++) {
                int slice = random().nextInt(numSlices);
                BytesRef sliceRef = new BytesRef("" + slice);
                List<Document> block = new ArrayList<>();

                int numChildren = random().nextInt(1, 4);
                boolean blockHasVector = false;
                for (int c = 0; c < numChildren; c++) {
                    Document child = new Document();
                    addRoutingSlice(child, sliceRef);
                    child.add(new KnnByteVectorField("vector", randomByteVector(dimensions), VectorSimilarityFunction.DOT_PRODUCT));
                    child.add(new StoredField(RoutingFieldMapper.NAME, sliceRef));
                    block.add(child);
                    blockHasVector = true;
                }
                if (blockHasVector) {
                    parentsPerSlice[slice]++;
                }

                Document parent = makeParent(new int[] { i });
                addRoutingSlice(parent, sliceRef);
                parent.add(new StoredField(RoutingFieldMapper.NAME, sliceRef));
                block.add(parent);
                w.addDocuments(block);
            }
            w.commit();

            byte[] queryVector = randomByteVector(dimensions);
            try (IndexReader reader = DirectoryReader.open(w)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                searcher.setQueryCachingPolicy(TrivialQueryCachingPolicy.ALWAYS);
                BitSetProducer parents = parentFilter(reader);

                // Query two slices at once
                int sliceA = 0;
                int sliceB = Math.min(1, numSlices - 1);
                int expectedTotal = parentsPerSlice[sliceA] + (sliceA != sliceB ? parentsPerSlice[sliceB] : 0);
                int k = 2 * Math.max(1, expectedTotal);

                Query kvq = new DiversifyingChildrenIVFKnnByteSlicedVectorQuery(
                    "vector",
                    queryVector,
                    k,
                    k,
                    null,
                    parents,
                    1.0f,
                    testResolver(),
                    RoutingFieldMapper.NAME,
                    new BytesRef("" + sliceA),
                    new BytesRef("" + sliceB)
                );
                TopDocs topDocs = searcher.search(kvq, k);
                assertEquals(expectedTotal, topDocs.scoreDocs.length);

                // Verify all results come from the requested slices
                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document document = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String sliceValue = document.getField(RoutingFieldMapper.NAME).binaryValue().utf8ToString();
                    assertTrue(
                        "Expected slice " + sliceA + " or " + sliceB + " but got " + sliceValue,
                        sliceValue.equals("" + sliceA) || sliceValue.equals("" + sliceB)
                    );
                }
            }
        }
    }

    public void testAllSlices() throws IOException {
        int dimensions = random().nextInt(12, 128);
        int numSlices = random().nextInt(3, 8);
        int numParents = random().nextInt(50, 200);
        int totalParentsWithVectors = 0;

        try (Directory dir = newDirectory(); IndexWriter w = new IndexWriter(dir, slicedIndexWriterConfig())) {
            for (int i = 0; i < numParents; i++) {
                int slice = random().nextInt(numSlices);
                BytesRef sliceRef = new BytesRef("" + slice);
                List<Document> block = new ArrayList<>();

                int numChildren = random().nextInt(1, 4);
                boolean blockHasVector = false;
                for (int c = 0; c < numChildren; c++) {
                    Document child = new Document();
                    addRoutingSlice(child, sliceRef);
                    child.add(new KnnByteVectorField("vector", randomByteVector(dimensions), VectorSimilarityFunction.DOT_PRODUCT));
                    child.add(new StoredField(RoutingFieldMapper.NAME, sliceRef));
                    block.add(child);
                    blockHasVector = true;
                }
                if (blockHasVector) {
                    totalParentsWithVectors++;
                }

                Document parent = makeParent(new int[] { i });
                addRoutingSlice(parent, sliceRef);
                parent.add(new StoredField(RoutingFieldMapper.NAME, sliceRef));
                block.add(parent);
                w.addDocuments(block);
            }
            w.commit();

            byte[] queryVector = randomByteVector(dimensions);
            try (IndexReader reader = DirectoryReader.open(w)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                searcher.setQueryCachingPolicy(TrivialQueryCachingPolicy.ALWAYS);
                BitSetProducer parents = parentFilter(reader);

                int k = 2 * Math.max(1, totalParentsWithVectors);
                // Empty sliceIds = search all slices
                Query kvq = new DiversifyingChildrenIVFKnnByteSlicedVectorQuery(
                    "vector",
                    queryVector,
                    k,
                    k,
                    null,
                    parents,
                    1.0f,
                    testResolver(),
                    RoutingFieldMapper.NAME
                );
                TopDocs topDocs = searcher.search(kvq, k);
                assertEquals(totalParentsWithVectors, topDocs.scoreDocs.length);
            }
        }
    }

    private void doTestSlices(BooleanSupplier hasVectorSupplier) throws IOException {
        int dimensions = random().nextInt(12, 500);
        int numParents = random().nextInt(50, 200);
        int numSlices = random().nextInt(3, 8);
        // Diversified kNN returns at most one hit per parent; track qualifying parents per slice.
        int[] parentsWithHitPerSlice = new int[numSlices];
        int[] blockSlice = new int[numParents];
        boolean[] blockCountsUnfiltered = new boolean[numParents];
        String docIdField = "_doc_id";

        try (Directory dir = newDirectory(); IndexWriter w = new IndexWriter(dir, slicedIndexWriterConfig())) {
            for (int i = 0; i < numParents; i++) {
                int slice = random().nextInt(numSlices);
                BytesRef sliceRef = new BytesRef("" + slice);
                List<Document> block = new ArrayList<>();

                int numChildren = random().nextInt(1, 4);
                boolean blockHasQualifyingChild = false;

                for (int c = 0; c < numChildren; c++) {
                    Document child = new Document();
                    addRoutingSlice(child, sliceRef);
                    child.add(new StringField(docIdField, "doc_" + i, Field.Store.NO));
                    boolean hasVector = hasVectorSupplier.getAsBoolean();
                    if (hasVector) {
                        child.add(new KnnByteVectorField("vector", randomByteVector(dimensions), VectorSimilarityFunction.DOT_PRODUCT));
                        blockHasQualifyingChild = true;
                    }
                    child.add(new StoredField(RoutingFieldMapper.NAME, sliceRef));
                    block.add(child);
                }
                if (blockHasQualifyingChild) {
                    parentsWithHitPerSlice[slice]++;
                }
                blockSlice[i] = slice;
                blockCountsUnfiltered[i] = blockHasQualifyingChild;

                Document parent = makeParent(new int[] { i });
                addRoutingSlice(parent, sliceRef);
                parent.add(new StringField(docIdField, "doc_" + i, Field.Store.NO));
                parent.add(new StoredField(RoutingFieldMapper.NAME, sliceRef));
                block.add(parent);

                w.addDocuments(block);
            }
            w.commit();

            if (random().nextBoolean()) {
                int deleteCount = random().nextInt(0, Math.max(1, numParents / 10));
                Set<Integer> parentsToDelete = new HashSet<>();
                while (parentsToDelete.size() < deleteCount) {
                    parentsToDelete.add(random().nextInt(numParents));
                }
                for (int blockId : parentsToDelete) {
                    if (blockCountsUnfiltered[blockId]) {
                        parentsWithHitPerSlice[blockSlice[blockId]]--;
                    }
                    w.deleteDocuments(new Term(docIdField, "doc_" + blockId));
                }
                if (parentsToDelete.isEmpty() == false) {
                    w.commit();
                }
            } else if (random().nextBoolean()) {
                w.forceMerge(1);
            }

            byte[] queryVector = randomByteVector(dimensions);
            try (IndexReader reader = DirectoryReader.open(w)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                searcher.setQueryCachingPolicy(TrivialQueryCachingPolicy.ALWAYS);
                BitSetProducer parents = parentFilter(reader);

                for (int iters = 0; iters < 2; iters++) {
                    // single slice queries
                    for (int slice = 0; slice < numSlices; slice++) {
                        int expectedDocs = parentsWithHitPerSlice[slice];
                        int k = 2 * Math.max(1, expectedDocs);
                        BytesRef sliceRef = new BytesRef("" + slice);
                        Query kvq = new DiversifyingChildrenIVFKnnByteSlicedVectorQuery(
                            "vector",
                            queryVector,
                            k,
                            k,
                            null,
                            parents,
                            1.0f,
                            testResolver(),
                            RoutingFieldMapper.NAME,
                            sliceRef
                        );
                        TopDocs topDocs = searcher.search(kvq, k);
                        assertEquals(expectedDocs, topDocs.scoreDocs.length);
                        for (int j = 0; j < topDocs.scoreDocs.length; j++) {
                            Document document = reader.storedFields().document(topDocs.scoreDocs[j].doc);
                            assertThat(document.getField(RoutingFieldMapper.NAME).binaryValue().utf8ToString(), equalTo("" + slice));
                        }
                    }

                    // multiple slice queries
                    for (int i = 0; i < 10; i++) {
                        int numQuerySlices = random().nextInt(numSlices) + 1;
                        int[] querySlices = new int[numQuerySlices];
                        int expectedDocs = 0;
                        int prevSlice = 0;
                        for (int j = 0; j < numQuerySlices; j++) {
                            int s = random().nextInt(prevSlice, numSlices - numQuerySlices + j + 1);
                            querySlices[j] = s;
                            expectedDocs += parentsWithHitPerSlice[s];
                            prevSlice = s + 1;
                        }
                        Arrays.sort(querySlices);
                        BytesRef[] sliceRefs = new BytesRef[querySlices.length];
                        for (int j = 0; j < querySlices.length; j++) {
                            sliceRefs[j] = new BytesRef("" + querySlices[j]);
                        }
                        int k = 2 * Math.max(1, expectedDocs);
                        Query kvq = new DiversifyingChildrenIVFKnnByteSlicedVectorQuery(
                            "vector",
                            queryVector,
                            k,
                            k,
                            null,
                            parents,
                            1.0f,
                            testResolver(),
                            RoutingFieldMapper.NAME,
                            sliceRefs
                        );
                        TopDocs topDocs = searcher.search(kvq, k);
                        assertEquals(expectedDocs, topDocs.scoreDocs.length);
                        for (int idx = 0; idx < topDocs.scoreDocs.length; idx++) {
                            Document document = reader.storedFields().document(topDocs.scoreDocs[idx].doc);
                            int docSlice = Integer.parseInt(document.getField(RoutingFieldMapper.NAME).binaryValue().utf8ToString());
                            assertTrue(Arrays.stream(querySlices).anyMatch(qs -> qs == docSlice));
                        }
                    }

                    // non-existing slice
                    Query kvq = new DiversifyingChildrenIVFKnnByteSlicedVectorQuery(
                        "vector",
                        queryVector,
                        3,
                        3,
                        null,
                        parents,
                        1.0f,
                        testResolver(),
                        RoutingFieldMapper.NAME,
                        new BytesRef("non_existent")
                    );
                    TopDocs topDocs = searcher.search(kvq, 3);
                    assertEquals(0, topDocs.scoreDocs.length);
                }
            }
        }
    }
}
