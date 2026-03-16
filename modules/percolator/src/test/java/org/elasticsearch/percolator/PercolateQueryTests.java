/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.percolator;

import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.memory.MemoryIndex;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.elasticsearch.common.bytes.BytesArray;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.search.internal.ContextIndexSearcher;
import org.elasticsearch.test.ESTestCase;
import org.junit.After;
import org.junit.Before;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

import static org.hamcrest.Matchers.arrayWithSize;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.is;

public class PercolateQueryTests extends ESTestCase {

    private Directory directory;
    private IndexWriter indexWriter;
    private DirectoryReader directoryReader;

    @Before
    public void init() throws Exception {
        directory = newDirectory();
        IndexWriterConfig config = new IndexWriterConfig(new WhitespaceAnalyzer());
        config.setMergePolicy(NoMergePolicy.INSTANCE);
        indexWriter = new IndexWriter(directory, config);
    }

    @After
    public void destroy() throws Exception {
        directoryReader.close();
        directory.close();
    }

    public void testPercolateQuery() throws Exception {
        List<Iterable<? extends IndexableField>> docs = new ArrayList<>();
        List<Query> queries = new ArrayList<>();
        PercolateQuery.QueryStore queryStore = ctx -> queries::get;

        queries.add(new TermQuery(new Term("field", "fox")));
        docs.add(Collections.singleton(new StringField("select", "a", Field.Store.NO)));

        SpanNearQuery.Builder snp = new SpanNearQuery.Builder("field", true);
        snp.addClause(new SpanTermQuery(new Term("field", "jumps")));
        snp.addClause(new SpanTermQuery(new Term("field", "lazy")));
        snp.addClause(new SpanTermQuery(new Term("field", "dog")));
        snp.setSlop(2);
        queries.add(snp.build());
        docs.add(Collections.singleton(new StringField("select", "b", Field.Store.NO)));

        PhraseQuery.Builder pq1 = new PhraseQuery.Builder();
        pq1.add(new Term("field", "quick"));
        pq1.add(new Term("field", "brown"));
        pq1.add(new Term("field", "jumps"));
        pq1.setSlop(1);
        queries.add(pq1.build());
        docs.add(Collections.singleton(new StringField("select", "b", Field.Store.NO)));

        BooleanQuery.Builder bq1 = new BooleanQuery.Builder();
        bq1.add(new TermQuery(new Term("field", "quick")), BooleanClause.Occur.MUST);
        bq1.add(new TermQuery(new Term("field", "brown")), BooleanClause.Occur.MUST);
        bq1.add(new TermQuery(new Term("field", "fox")), BooleanClause.Occur.MUST);
        queries.add(bq1.build());
        docs.add(Collections.singleton(new StringField("select", "b", Field.Store.NO)));

        indexWriter.addDocuments(docs);
        indexWriter.close();
        directoryReader = DirectoryReader.open(directory);
        IndexSearcher shardSearcher = newSearcher(directoryReader);

        MemoryIndex memoryIndex = new MemoryIndex();
        memoryIndex.addField("field", "the quick brown fox jumps over the lazy dog", new WhitespaceAnalyzer());
        IndexSearcher percolateSearcher = memoryIndex.createSearcher();
        // no scoring, wrapping it in a constant score query:
        Query query = new ConstantScoreQuery(
            new PercolateQuery(
                "_name",
                queryStore,
                Collections.singletonList(new BytesArray("a")),
                new TermQuery(new Term("select", "a")),
                percolateSearcher,
                null,
                Queries.NO_DOCS_INSTANCE
            )
        );
        TopDocs topDocs = shardSearcher.search(query, 10);
        assertThat(topDocs.totalHits.value(), equalTo(1L));
        assertThat(topDocs.scoreDocs.length, equalTo(1));
        assertThat(topDocs.scoreDocs[0].doc, equalTo(0));
        Explanation explanation = shardSearcher.explain(query, 0);
        assertThat(explanation.isMatch(), is(true));
        assertThat(explanation.getValue(), equalTo(topDocs.scoreDocs[0].score));

        query = new ConstantScoreQuery(
            new PercolateQuery(
                "_name",
                queryStore,
                Collections.singletonList(new BytesArray("b")),
                new TermQuery(new Term("select", "b")),
                percolateSearcher,
                null,
                Queries.NO_DOCS_INSTANCE
            )
        );
        topDocs = shardSearcher.search(query, 10);
        assertThat(topDocs.totalHits.value(), equalTo(3L));
        assertThat(topDocs.scoreDocs.length, equalTo(3));
        assertThat(topDocs.scoreDocs[0].doc, equalTo(1));
        explanation = shardSearcher.explain(query, 1);
        assertThat(explanation.isMatch(), is(true));
        assertThat(explanation.getValue(), equalTo(topDocs.scoreDocs[0].score));

        assertThat(topDocs.scoreDocs[1].doc, equalTo(2));
        explanation = shardSearcher.explain(query, 2);
        assertThat(explanation.isMatch(), is(true));
        assertThat(explanation.getValue(), equalTo(topDocs.scoreDocs[1].score));

        assertThat(topDocs.scoreDocs[2].doc, equalTo(3));
        explanation = shardSearcher.explain(query, 2);
        assertThat(explanation.isMatch(), is(true));
        assertThat(explanation.getValue(), equalTo(topDocs.scoreDocs[2].score));

        query = new ConstantScoreQuery(
            new PercolateQuery(
                "_name",
                queryStore,
                Collections.singletonList(new BytesArray("c")),
                Queries.ALL_DOCS_INSTANCE,
                percolateSearcher,
                null,
                Queries.ALL_DOCS_INSTANCE
            )
        );
        topDocs = shardSearcher.search(query, 10);
        assertThat(topDocs.totalHits.value(), equalTo(4L));

        query = new PercolateQuery(
            "_name",
            queryStore,
            Collections.singletonList(new BytesArray("{}")),
            new TermQuery(new Term("select", "b")),
            percolateSearcher,
            null,
            Queries.NO_DOCS_INSTANCE
        );
        topDocs = shardSearcher.search(query, 10);
        assertThat(topDocs.totalHits.value(), equalTo(3L));
        assertThat(topDocs.scoreDocs.length, equalTo(3));
        assertThat(topDocs.scoreDocs[0].doc, equalTo(3));
        explanation = shardSearcher.explain(query, 3);
        assertThat(explanation.isMatch(), is(true));
        assertThat(explanation.getValue(), equalTo(topDocs.scoreDocs[0].score));
        assertThat(explanation.getDetails(), arrayWithSize(1));

        assertThat(topDocs.scoreDocs[1].doc, equalTo(2));
        explanation = shardSearcher.explain(query, 2);
        assertThat(explanation.isMatch(), is(true));
        assertThat(explanation.getValue(), equalTo(topDocs.scoreDocs[1].score));
        assertThat(explanation.getDetails(), arrayWithSize(1));

        assertThat(topDocs.scoreDocs[2].doc, equalTo(1));
        explanation = shardSearcher.explain(query, 1);
        assertThat(explanation.isMatch(), is(true));
        assertThat(explanation.getValue(), equalTo(topDocs.scoreDocs[2].score));
        assertThat(explanation.getDetails(), arrayWithSize(1));
    }

    /**
     * Tests the parallel verification path by using a ContextIndexSearcher with an executor
     * and more candidates than PARALLEL_MIN_CANDIDATES (128). Verifies both scoring and
     * non-scoring paths produce correct results and that explain works without re-running
     * full parallel verification.
     */
    public void testPercolateQueryParallelVerification() throws Exception {
        int numQueries = PercolateQuery.PARALLEL_MIN_CANDIDATES + randomIntBetween(10, 100);
        List<Query> queries = new ArrayList<>();
        PercolateQuery.QueryStore queryStore = ctx -> queries::get;

        // Create queries: half match "fox", half match "missing" (won't match the document)
        int expectedMatches = 0;
        for (int i = 0; i < numQueries; i++) {
            if (i % 2 == 0) {
                queries.add(new TermQuery(new Term("field", "fox")));
                expectedMatches++;
            } else {
                queries.add(new TermQuery(new Term("field", "missing")));
            }
            indexWriter.addDocument(Collections.singleton(new StringField("select", "a", Field.Store.NO)));
        }
        indexWriter.close();
        directoryReader = DirectoryReader.open(directory);

        // Create ContextIndexSearcher with executor to activate the parallel path
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(randomIntBetween(2, 4));
        try {
            ContextIndexSearcher shardSearcher = new ContextIndexSearcher(
                directoryReader,
                IndexSearcher.getDefaultSimilarity(),
                IndexSearcher.getDefaultQueryCache(),
                IndexSearcher.getDefaultQueryCachingPolicy(),
                false,
                executor,
                executor.getMaximumPoolSize(),
                1
            );

            MemoryIndex memoryIndex = new MemoryIndex();
            memoryIndex.addField("field", "the quick brown fox jumps over the lazy dog", new WhitespaceAnalyzer());
            memoryIndex.freeze();
            IndexSearcher percolateSearcher = memoryIndex.createSearcher();

            // Non-scoring path (ConstantScoreQuery wrapper)
            Query query = new ConstantScoreQuery(
                new PercolateQuery(
                    "_name",
                    queryStore,
                    Collections.singletonList(new BytesArray("{}")),
                    new TermQuery(new Term("select", "a")),
                    percolateSearcher,
                    null,
                    Queries.NO_DOCS_INSTANCE
                )
            );
            TopDocs topDocs = shardSearcher.search(query, numQueries + 10);
            assertThat(topDocs.totalHits.value(), equalTo((long) expectedMatches));
            assertThat(topDocs.scoreDocs.length, equalTo(expectedMatches));

            // Verify all matched docs are even-numbered (the "fox" queries)
            for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                assertThat("matched doc should be an even-numbered query", topDocs.scoreDocs[i].doc % 2, equalTo(0));
            }

            // Verify explain works for a matching doc
            Explanation explanation = shardSearcher.explain(query, 0);
            assertThat(explanation.isMatch(), is(true));

            // Verify explain works for a non-matching doc
            Explanation noMatchExplanation = shardSearcher.explain(query, 1);
            assertThat(noMatchExplanation.isMatch(), is(false));

            // Scoring path (no ConstantScoreQuery wrapper)
            Query scoringQuery = new PercolateQuery(
                "_name",
                queryStore,
                Collections.singletonList(new BytesArray("{}")),
                new TermQuery(new Term("select", "a")),
                percolateSearcher,
                null,
                Queries.NO_DOCS_INSTANCE
            );
            topDocs = shardSearcher.search(scoringQuery, numQueries + 10);
            assertThat(topDocs.totalHits.value(), equalTo((long) expectedMatches));

            // Verify scoring explain returns match with detail
            Explanation scoringExplanation = shardSearcher.explain(scoringQuery, 0);
            assertThat(scoringExplanation.isMatch(), is(true));
            assertThat(scoringExplanation.getDetails(), arrayWithSize(1));
        } finally {
            terminate(executor);
        }
    }

}
