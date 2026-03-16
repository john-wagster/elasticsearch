/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.percolator;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TwoPhaseIterator;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.lucene.Lucene;
import org.elasticsearch.core.CheckedFunction;
import org.elasticsearch.search.internal.ContextIndexSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;

final class PercolateQuery extends Query implements Accountable {

    // cost of matching the query against the document, arbitrary as it would be really complex to estimate
    private static final float MATCH_COST = 1000;

    private final String name;
    private final QueryStore queryStore;
    private final List<BytesReference> documents;
    private final Query candidateMatchesQuery;
    private final Query verifiedMatchesQuery;
    private final IndexSearcher percolatorIndexSearcher;
    private final Query nonNestedDocsFilter;

    PercolateQuery(
        String name,
        QueryStore queryStore,
        List<BytesReference> documents,
        Query candidateMatchesQuery,
        IndexSearcher percolatorIndexSearcher,
        Query nonNestedDocsFilter,
        Query verifiedMatchesQuery
    ) {
        this.name = name;
        this.documents = Objects.requireNonNull(documents);
        this.candidateMatchesQuery = Objects.requireNonNull(candidateMatchesQuery);
        this.queryStore = Objects.requireNonNull(queryStore);
        this.percolatorIndexSearcher = Objects.requireNonNull(percolatorIndexSearcher);
        this.nonNestedDocsFilter = nonNestedDocsFilter;
        this.verifiedMatchesQuery = Objects.requireNonNull(verifiedMatchesQuery);
    }

    @Override
    public Query rewrite(IndexSearcher searcher) throws IOException {
        Query rewritten = candidateMatchesQuery.rewrite(searcher);
        if (rewritten != candidateMatchesQuery) {
            return new PercolateQuery(
                name,
                queryStore,
                documents,
                rewritten,
                percolatorIndexSearcher,
                nonNestedDocsFilter,
                verifiedMatchesQuery
            );
        } else {
            return this;
        }
    }

    // Minimum number of candidate documents required to activate parallel verification.
    // Below this threshold sequential verification is faster because thread dispatch overhead dominates.
    static final int PARALLEL_MIN_CANDIDATES = 128;

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        final Weight verifiedMatchesWeight = verifiedMatchesQuery.createWeight(searcher, ScoreMode.COMPLETE_NO_SCORES, boost);
        final Weight candidateMatchesWeight = candidateMatchesQuery.createWeight(searcher, ScoreMode.COMPLETE_NO_SCORES, boost);
        final TaskExecutor taskExecutor;
        final int maxSlices;
        if (searcher instanceof ContextIndexSearcher cis && cis.hasExecutor()) {
            taskExecutor = searcher.getTaskExecutor();
            maxSlices = cis.getMaximumNumberOfSlices();
        } else {
            taskExecutor = null;
            maxSlices = 1;
        }
        return new Weight(this) {
            @Override
            public Explanation explain(LeafReaderContext leafReaderContext, int docId) throws IOException {
                if (taskExecutor != null) {
                    // Direct single-doc verification avoids re-running full parallel verification.
                    // First check if docId is a candidate match.
                    ScorerSupplier candidateSupplier = candidateMatchesWeight.scorerSupplier(leafReaderContext);
                    if (candidateSupplier == null) {
                        return Explanation.noMatch("PercolateQuery");
                    }
                    Scorer candidateScorer = candidateSupplier.get(0L);
                    if (candidateScorer.iterator().advance(docId) != docId) {
                        return Explanation.noMatch("PercolateQuery");
                    }
                    // Verify this single doc against the MemoryIndex
                    CheckedFunction<Integer, Query, IOException> percolatorQueries = queryStore.getQueries(leafReaderContext);
                    Query query = percolatorQueries.apply(docId);
                    if (query == null) {
                        return Explanation.noMatch("PercolateQuery");
                    }
                    if (nonNestedDocsFilter != null) {
                        query = new BooleanQuery.Builder().add(query, Occur.MUST).add(nonNestedDocsFilter, Occur.FILTER).build();
                    }
                    if (scoreMode.needsScores()) {
                        TopDocs topDocs = percolatorIndexSearcher.search(query, 1);
                        if (topDocs.scoreDocs.length > 0) {
                            Explanation detail = percolatorIndexSearcher.explain(query, 0);
                            return Explanation.match(topDocs.scoreDocs[0].score, "PercolateQuery", detail);
                        }
                    } else {
                        if (Lucene.exists(percolatorIndexSearcher, query)) {
                            return Explanation.match(0f, "PercolateQuery");
                        }
                    }
                    return Explanation.noMatch("PercolateQuery");
                }
                // Sequential path: use scorer with TwoPhaseIterator
                Scorer scorer = scorer(leafReaderContext);
                if (scorer != null) {
                    TwoPhaseIterator twoPhaseIterator = scorer.twoPhaseIterator();
                    if (twoPhaseIterator != null) {
                        int result = twoPhaseIterator.approximation().advance(docId);
                        if (result == docId) {
                            if (twoPhaseIterator.matches()) {
                                if (scoreMode.needsScores()) {
                                    CheckedFunction<Integer, Query, IOException> percolatorQueries = queryStore.getQueries(
                                        leafReaderContext
                                    );
                                    Query query = percolatorQueries.apply(docId);
                                    Explanation detail = percolatorIndexSearcher.explain(query, 0);
                                    return Explanation.match(scorer.score(), "PercolateQuery", detail);
                                } else {
                                    return Explanation.match(scorer.score(), "PercolateQuery");
                                }
                            }
                        }
                    }
                }
                return Explanation.noMatch("PercolateQuery");
            }

            @Override
            public ScorerSupplier scorerSupplier(LeafReaderContext leafReaderContext) throws IOException {
                final ScorerSupplier approximationSupplier = candidateMatchesWeight.scorerSupplier(leafReaderContext);
                if (approximationSupplier == null) {
                    return null;
                }

                ScorerSupplier verifiedDocsScorer;
                if (scoreMode.needsScores()) {
                    verifiedDocsScorer = null;
                } else {
                    verifiedDocsScorer = verifiedMatchesWeight.scorerSupplier(leafReaderContext);
                }

                return new ScorerSupplier() {
                    @Override
                    public Scorer get(long leadCost) throws IOException {
                        final Scorer approximation = approximationSupplier.get(leadCost);

                        if (taskExecutor != null) {
                            return parallelScorer(leafReaderContext, approximation, verifiedDocsScorer);
                        }

                        // Sequential path (original behavior)
                        final CheckedFunction<Integer, Query, IOException> percolatorQueries = queryStore.getQueries(leafReaderContext);
                        if (scoreMode.needsScores()) {
                            return new BaseScorer(approximation) {

                                float score;

                                @Override
                                boolean matchDocId(int docId) throws IOException {
                                    Query query = percolatorQueries.apply(docId);
                                    if (query != null) {
                                        if (nonNestedDocsFilter != null) {
                                            query = new BooleanQuery.Builder().add(query, Occur.MUST)
                                                .add(nonNestedDocsFilter, Occur.FILTER)
                                                .build();
                                        }
                                        TopDocs topDocs = percolatorIndexSearcher.search(query, 1);
                                        if (topDocs.scoreDocs.length > 0) {
                                            score = topDocs.scoreDocs[0].score;
                                            return true;
                                        } else {
                                            return false;
                                        }
                                    } else {
                                        return false;
                                    }
                                }

                                @Override
                                public float score() {
                                    return score;
                                }
                            };
                        } else {
                            Bits verifiedDocsBits = Lucene.asSequentialAccessBits(leafReaderContext.reader().maxDoc(), verifiedDocsScorer);
                            return new BaseScorer(approximation) {

                                @Override
                                public float score() throws IOException {
                                    return 0f;
                                }

                                boolean matchDocId(int docId) throws IOException {
                                    // We use the verifiedDocsBits to skip the expensive MemoryIndex verification.
                                    // If docId also appears in the verifiedDocsBits then that means during indexing
                                    // we were able to extract all query terms and for this candidate match
                                    // and we determined based on the nature of the query that it is safe to skip
                                    // the MemoryIndex verification.
                                    if (verifiedDocsBits.get(docId)) {
                                        return true;
                                    }
                                    Query query = percolatorQueries.apply(docId);
                                    if (query == null) {
                                        return false;
                                    }
                                    if (nonNestedDocsFilter != null) {
                                        query = new BooleanQuery.Builder().add(query, Occur.MUST)
                                            .add(nonNestedDocsFilter, Occur.FILTER)
                                            .build();
                                    }
                                    return Lucene.exists(percolatorIndexSearcher, query);
                                }
                            };
                        }
                    }

                    @Override
                    public long cost() {
                        return approximationSupplier.cost();
                    }
                };
            }

            /**
             * Collect all candidate doc IDs from the approximation, then verify them against
             * the MemoryIndex in parallel batches using the search thread pool.
             */
            private Scorer parallelScorer(LeafReaderContext leafReaderContext, Scorer approximation, ScorerSupplier verifiedDocsScorer)
                throws IOException {
                int maxDoc = leafReaderContext.reader().maxDoc();

                // Collect all candidate doc IDs from the approximation
                List<Integer> candidates = new ArrayList<>();
                DocIdSetIterator approxIter = approximation.iterator();
                for (int docId = approxIter.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = approxIter.nextDoc()) {
                    candidates.add(docId);
                }

                if (candidates.isEmpty()) {
                    return new Scorer() {
                        @Override
                        public int docID() {
                            return DocIdSetIterator.NO_MORE_DOCS;
                        }

                        @Override
                        public DocIdSetIterator iterator() {
                            return DocIdSetIterator.empty();
                        }

                        @Override
                        public float getMaxScore(int upTo) {
                            return 0f;
                        }

                        @Override
                        public float score() {
                            return 0f;
                        }
                    };
                }

                // For small candidate sets, fall back to sequential verification
                if (candidates.size() < PARALLEL_MIN_CANDIDATES) {
                    return sequentialVerify(leafReaderContext, candidates, maxDoc, verifiedDocsScorer);
                }

                // Determine number of threads: use configured parallelism but cap at candidate count
                int numThreads = Math.min(maxSlices, candidates.size());
                numThreads = Math.max(numThreads, 1);

                // Split candidates into batches
                int batchSize = (candidates.size() + numThreads - 1) / numThreads;

                if (scoreMode.needsScores()) {
                    // Scoring path: each thread collects matches and scores into per-thread maps
                    // to avoid allocating a potentially large float[maxDoc] array.
                    List<Callable<Map.Entry<FixedBitSet, Map<Integer, Float>>>> scoringTasks = new ArrayList<>(numThreads);
                    for (int t = 0; t < numThreads; t++) {
                        int from = t * batchSize;
                        int to = Math.min(from + batchSize, candidates.size());
                        if (from >= to) break;
                        List<Integer> batch = candidates.subList(from, to);
                        scoringTasks.add(() -> {
                            CheckedFunction<Integer, Query, IOException> queries = queryStore.getQueries(leafReaderContext);
                            FixedBitSet bits = new FixedBitSet(maxDoc);
                            Map<Integer, Float> batchScores = new HashMap<>();
                            for (int docId : batch) {
                                Query query = queries.apply(docId);
                                if (query != null) {
                                    if (nonNestedDocsFilter != null) {
                                        query = new BooleanQuery.Builder().add(query, Occur.MUST)
                                            .add(nonNestedDocsFilter, Occur.FILTER)
                                            .build();
                                    }
                                    TopDocs topDocs = percolatorIndexSearcher.search(query, 1);
                                    if (topDocs.scoreDocs.length > 0) {
                                        bits.set(docId);
                                        batchScores.put(docId, topDocs.scoreDocs[0].score);
                                    }
                                }
                            }
                            return Map.entry(bits, batchScores);
                        });
                    }

                    List<Map.Entry<FixedBitSet, Map<Integer, Float>>> scoringResults = taskExecutor.invokeAll(scoringTasks);

                    // Merge results
                    FixedBitSet merged = new FixedBitSet(maxDoc);
                    Map<Integer, Float> scores = new HashMap<>();
                    for (Map.Entry<FixedBitSet, Map<Integer, Float>> entry : scoringResults) {
                        merged.or(entry.getKey());
                        scores.putAll(entry.getValue());
                    }

                    return scorerFromBitSet(merged, scores);
                } else {
                    // Non-scoring path: each thread collects matches only.
                    // Pre-compute verified bits into a FixedBitSet for thread-safe random access,
                    // since Lucene.asSequentialAccessBits() must be consumed in order.
                    FixedBitSet verifiedBits = precomputeVerifiedBits(maxDoc, verifiedDocsScorer);
                    List<Callable<FixedBitSet>> tasks = new ArrayList<>(numThreads);
                    for (int t = 0; t < numThreads; t++) {
                        int from = t * batchSize;
                        int to = Math.min(from + batchSize, candidates.size());
                        if (from >= to) break;
                        List<Integer> batch = candidates.subList(from, to);
                        tasks.add(() -> {
                            CheckedFunction<Integer, Query, IOException> queries = queryStore.getQueries(leafReaderContext);
                            FixedBitSet bits = new FixedBitSet(maxDoc);
                            for (int docId : batch) {
                                if (verifiedBits.get(docId)) {
                                    bits.set(docId);
                                    continue;
                                }
                                Query query = queries.apply(docId);
                                if (query == null) {
                                    continue;
                                }
                                if (nonNestedDocsFilter != null) {
                                    query = new BooleanQuery.Builder().add(query, Occur.MUST)
                                        .add(nonNestedDocsFilter, Occur.FILTER)
                                        .build();
                                }
                                if (Lucene.exists(percolatorIndexSearcher, query)) {
                                    bits.set(docId);
                                }
                            }
                            return bits;
                        });
                    }

                    List<FixedBitSet> results = taskExecutor.invokeAll(tasks);

                    // Merge results
                    FixedBitSet merged = new FixedBitSet(maxDoc);
                    for (FixedBitSet bits : results) {
                        merged.or(bits);
                    }

                    return scorerFromBitSet(merged, null);
                }
            }

            /**
             * Sequential fallback for small candidate sets. Avoids thread dispatch overhead.
             */
            private Scorer sequentialVerify(
                LeafReaderContext leafReaderContext,
                List<Integer> candidates,
                int maxDoc,
                ScorerSupplier verifiedDocsScorer
            ) throws IOException {
                CheckedFunction<Integer, Query, IOException> queries = queryStore.getQueries(leafReaderContext);
                FixedBitSet matchBits = new FixedBitSet(maxDoc);

                if (scoreMode.needsScores()) {
                    Map<Integer, Float> scores = new HashMap<>();
                    for (int docId : candidates) {
                        Query query = queries.apply(docId);
                        if (query != null) {
                            if (nonNestedDocsFilter != null) {
                                query = new BooleanQuery.Builder().add(query, Occur.MUST).add(nonNestedDocsFilter, Occur.FILTER).build();
                            }
                            TopDocs topDocs = percolatorIndexSearcher.search(query, 1);
                            if (topDocs.scoreDocs.length > 0) {
                                matchBits.set(docId);
                                scores.put(docId, topDocs.scoreDocs[0].score);
                            }
                        }
                    }
                    return scorerFromBitSet(matchBits, scores);
                } else {
                    Bits verifiedDocsBits = Lucene.asSequentialAccessBits(maxDoc, verifiedDocsScorer);
                    for (int docId : candidates) {
                        if (verifiedDocsBits.get(docId)) {
                            matchBits.set(docId);
                            continue;
                        }
                        Query query = queries.apply(docId);
                        if (query == null) {
                            continue;
                        }
                        if (nonNestedDocsFilter != null) {
                            query = new BooleanQuery.Builder().add(query, Occur.MUST).add(nonNestedDocsFilter, Occur.FILTER).build();
                        }
                        if (Lucene.exists(percolatorIndexSearcher, query)) {
                            matchBits.set(docId);
                        }
                    }
                    return scorerFromBitSet(matchBits, null);
                }
            }

            private Scorer scorerFromBitSet(FixedBitSet matchBits, Map<Integer, Float> scores) {
                DocIdSetIterator iter = new BitSetIterator(matchBits, matchBits.cardinality());
                return new Scorer() {
                    @Override
                    public int docID() {
                        return iter.docID();
                    }

                    @Override
                    public DocIdSetIterator iterator() {
                        return iter;
                    }

                    @Override
                    public float getMaxScore(int upTo) {
                        return scores != null ? Float.MAX_VALUE : 0f;
                    }

                    @Override
                    public float score() {
                        return scores != null ? scores.getOrDefault(iter.docID(), 0f) : 0f;
                    }
                };
            }

            /**
             * Pre-compute verified match bits into a FixedBitSet so that multiple threads
             * can perform random-access lookups concurrently. The sequential-access Bits
             * returned by {@link Lucene#asSequentialAccessBits} cannot be shared across threads.
             */
            private FixedBitSet precomputeVerifiedBits(int maxDoc, ScorerSupplier verifiedDocsScorer) throws IOException {
                FixedBitSet bits = new FixedBitSet(maxDoc);
                if (verifiedDocsScorer == null) {
                    return bits;
                }
                Scorer scorer = verifiedDocsScorer.get(0L);
                if (scorer != null) {
                    DocIdSetIterator iter = scorer.iterator();
                    for (int docId = iter.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = iter.nextDoc()) {
                        bits.set(docId);
                    }
                }
                return bits;
            }

            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                // This query uses a significant amount of memory, let's never
                // cache it or compound queries that wrap it.
                return false;
            }
        };
    }

    String getName() {
        return name;
    }

    IndexSearcher getPercolatorIndexSearcher() {
        return percolatorIndexSearcher;
    }

    boolean excludesNestedDocs() {
        return nonNestedDocsFilter != null;
    }

    List<BytesReference> getDocuments() {
        return documents;
    }

    QueryStore getQueryStore() {
        return queryStore;
    }

    Query getCandidateMatchesQuery() {
        return candidateMatchesQuery;
    }

    Query getVerifiedMatchesQuery() {
        return verifiedMatchesQuery;
    }

    // Comparing identity here to avoid being cached
    // Note that in theory if the same instance gets used multiple times it could still get cached,
    // however since we create a new query instance each time we this query this shouldn't happen and thus
    // this risk neglectable.
    @Override
    public boolean equals(Object o) {
        return this == o;
    }

    // Computing hashcode based on identity to avoid caching.
    @Override
    public int hashCode() {
        return System.identityHashCode(this);
    }

    @Override
    public String toString(String s) {
        StringBuilder sources = new StringBuilder();
        for (BytesReference document : documents) {
            sources.append(document.utf8ToString());
            sources.append('\n');
        }
        return "PercolateQuery{document_sources={" + sources + "},inner={" + candidateMatchesQuery.toString(s) + "}}";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public long ramBytesUsed() {
        long ramUsed = 0L;
        for (BytesReference document : documents) {
            ramUsed += document.ramBytesUsed();
        }
        return ramUsed;
    }

    @FunctionalInterface
    interface QueryStore {
        CheckedFunction<Integer, Query, IOException> getQueries(LeafReaderContext ctx) throws IOException;
    }

    abstract static class BaseScorer extends Scorer {

        final Scorer approximation;

        BaseScorer(Scorer approximation) {
            this.approximation = approximation;
        }

        @Override
        public final DocIdSetIterator iterator() {
            return TwoPhaseIterator.asDocIdSetIterator(twoPhaseIterator());
        }

        @Override
        public final TwoPhaseIterator twoPhaseIterator() {
            return new TwoPhaseIterator(approximation.iterator()) {
                @Override
                public boolean matches() throws IOException {
                    return matchDocId(approximation.docID());
                }

                @Override
                public float matchCost() {
                    return MATCH_COST;
                }
            };
        }

        @Override
        public final int docID() {
            return approximation.docID();
        }

        abstract boolean matchDocId(int docId) throws IOException;

        @Override
        public float getMaxScore(int upTo) throws IOException {
            return Float.MAX_VALUE;
        }
    }

}
