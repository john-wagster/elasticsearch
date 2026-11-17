/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.TestUtil;
import org.elasticsearch.index.codec.vectors.diskbbq.AshConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfFlushConfigSource;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfMergeConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfSegmentConfig;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;
import org.elasticsearch.index.store.ShardCodecStateDirectory;
import org.elasticsearch.search.vectors.CountingQueryTransformCache;
import org.elasticsearch.search.vectors.IVFKnnSearchStrategy;
import org.elasticsearch.search.vectors.QueryTransformCache;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;

import static org.hamcrest.Matchers.hasSize;

/**
 * Sharing of the ASH projection matrix W across the segments of a shard, through the {@link ShardCodecStateDirectory} the
 * shard's {@code Store} wraps its directory with, and reuse of a query's projection across the segments that share W.
 */
public class ESNextDiskASHSharedProjectionTests extends ESTestCase {

    private static final String FIELD = "f";
    // ASH defaults project 128 dims to 64, and share a matrix learned from a full sample of 10 * 128 vectors
    private static final int DIMS = 128;
    private static final int FULL_SAMPLE = 10 * DIMS;

    private static Codec ashCodec() {
        return TestUtil.alwaysKnnVectorsFormat(
            new ESNextDiskASHVectorsFormat(
                ESNextDiskASHVectorsFormat.MIN_VECTORS_PER_CLUSTER,
                ESNextDiskASHVectorsFormat.DEFAULT_CENTROIDS_PER_PARENT_CLUSTER,
                null
            )
        );
    }

    private static IndexWriterConfig writerConfig() {
        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setCodec(ashCodec());
        // keep each flush its own segment until a forced merge
        LogDocMergePolicy mergePolicy = new LogDocMergePolicy();
        mergePolicy.setMergeFactor(100);
        iwc.setMergePolicy(mergePolicy);
        iwc.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
        iwc.setRAMBufferSizeMB(1024);
        return iwc;
    }

    /** A low-rank dataset, so a projection learned from one segment's sample also fits the others. */
    private static float[][] basis() {
        float[][] basis = new float[16][DIMS];
        for (float[] b : basis) {
            for (int d = 0; d < DIMS; d++) {
                b[d] = randomFloat() * 2 - 1;
            }
        }
        return basis;
    }

    private static float[] vector(float[][] basis) {
        float[] v = new float[DIMS];
        for (float[] b : basis) {
            float w = randomFloat() * 2 - 1;
            for (int d = 0; d < DIMS; d++) {
                v[d] += w * b[d];
            }
        }
        for (int d = 0; d < DIMS; d++) {
            v[d] += (randomFloat() - 0.5f) * 0.1f;
        }
        return v;
    }

    private static void flush(IndexWriter w, float[][] basis, int docs) throws IOException {
        for (int i = 0; i < docs; i++) {
            Document doc = new Document();
            doc.add(new KnnFloatVectorField(FIELD, vector(basis), VectorSimilarityFunction.EUCLIDEAN));
            w.addDocument(doc);
        }
        w.commit();
    }

    /** The projection matrix ids of the segments, in segment order. */
    private static List<UUID> segmentMatrixIds(IndexWriter w) throws IOException {
        List<UUID> ids = new ArrayList<>();
        try (DirectoryReader reader = DirectoryReader.open(w)) {
            for (LeafReaderContext ctx : reader.leaves()) {
                FieldInfo fieldInfo = ctx.reader().getFieldInfos().fieldInfo(FIELD);
                ids.add(ESNextDiskASHVectorsFormatTests.ashReader(ctx.reader(), FIELD).getProjectionMatrix(fieldInfo).id());
            }
        }
        return ids;
    }

    public void testFlushedSegmentsShareTheFirstFullSampleProjection() throws IOException {
        float[][] basis = basis();
        try (Directory dir = new ShardCodecStateDirectory(newDirectory()); IndexWriter w = new IndexWriter(dir, writerConfig())) {
            // too small to learn a shareable projection: trains its own
            flush(w, basis, FULL_SAMPLE / 2);
            // learns from a full sample: becomes the shard's shared projection
            flush(w, basis, FULL_SAMPLE + randomIntBetween(0, 200));
            // reuse the shared projection, whatever their size
            flush(w, basis, FULL_SAMPLE / 2);
            flush(w, basis, FULL_SAMPLE * 2);

            List<UUID> ids = segmentMatrixIds(w);
            assertThat(ids, hasSize(4));
            UUID shared = ids.get(1);
            assertNotEquals(shared, ids.get(0));
            assertEquals(shared, ids.get(2));
            assertEquals(shared, ids.get(3));

            w.forceMerge(1);
            assertEquals(List.of(shared), segmentMatrixIds(w));
        }
    }

    public void testRotationWithoutReductionIsSharedWhateverTheSampleSize() throws IOException {
        // below 512 dims the dims-derived configuration keeps every dimension: W is a rotation learned from no sample, and
        // any rotation is as good as another, so the first segment's is shared however few vectors it has
        int dims = 64;
        float[][] basis = new float[4][dims];
        for (float[] b : basis) {
            for (int d = 0; d < dims; d++) {
                b[d] = randomFloat() * 2 - 1;
            }
        }
        IndexWriterConfig iwc = writerConfig();
        iwc.setCodec(
            TestUtil.alwaysKnnVectorsFormat(
                new ESNextDiskASHVectorsFormat(
                    AshConfigResolver.dimsDerived(null, IvfSegmentConfig.AshConfig.DEFAULT_QUERY_BITS_PER_DIM),
                    ESNextDiskASHVectorsFormat.MIN_VECTORS_PER_CLUSTER,
                    ESNextDiskASHVectorsFormat.DEFAULT_CENTROIDS_PER_PARENT_CLUSTER,
                    DenseVectorFieldMapper.ElementType.FLOAT,
                    false,
                    null,
                    1,
                    null,
                    -1,
                    null,
                    IvfFlushConfigSource.empty(),
                    IvfMergeConfigResolver.useCodecDefault(),
                    false
                )
            )
        );
        try (Directory dir = new ShardCodecStateDirectory(newDirectory()); IndexWriter w = new IndexWriter(dir, iwc)) {
            int segments = randomIntBetween(2, 4);
            for (int s = 0; s < segments; s++) {
                // at least twice as many vectors as dimensions, so that the rotation counts as learned
                int docs = randomIntBetween(2 * dims, 4 * dims);
                for (int i = 0; i < docs; i++) {
                    float[] v = new float[dims];
                    for (float[] b : basis) {
                        float c = randomFloat() * 2 - 1;
                        for (int d = 0; d < dims; d++) {
                            v[d] += c * b[d];
                        }
                    }
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD, v, VectorSimilarityFunction.EUCLIDEAN));
                    w.addDocument(doc);
                }
                w.commit();
            }
            List<UUID> ids = segmentMatrixIds(w);
            assertThat(ids, hasSize(segments));
            assertEquals("every segment shares the first segment's rotation", 1, new HashSet<>(ids).size());
        }
    }

    public void testWithoutShardStateMergeReusesAFullSampleInputProjection() throws IOException {
        float[][] basis = basis();
        try (Directory dir = newDirectory(); IndexWriter w = new IndexWriter(dir, writerConfig())) {
            flush(w, basis, FULL_SAMPLE);
            flush(w, basis, FULL_SAMPLE + 100);
            List<UUID> ids = segmentMatrixIds(w);
            assertEquals("without per-shard state every flush learns its own projection", 2, new HashSet<>(ids).size());

            w.forceMerge(1);
            List<UUID> merged = segmentMatrixIds(w);
            assertThat(merged, hasSize(1));
            assertEquals("the merge keeps the projection of the input with the most vectors", ids.get(1), merged.get(0));
        }
    }

    public void testQueryIsProjectedOncePerSharedProjection() throws IOException {
        float[][] basis = basis();
        try (Directory dir = new ShardCodecStateDirectory(newDirectory()); IndexWriter w = new IndexWriter(dir, writerConfig())) {
            int segments = randomIntBetween(2, 4);
            for (int s = 0; s < segments; s++) {
                flush(w, basis, FULL_SAMPLE + randomIntBetween(0, 100));
            }
            assertEquals(1, new HashSet<>(segmentMatrixIds(w)).size());

            try (DirectoryReader reader = DirectoryReader.open(w)) {
                assertThat(reader.leaves(), hasSize(segments));
                for (int q = 0; q < 5; q++) {
                    float[] query = vector(basis);
                    CountingQueryTransformCache transforms = new CountingQueryTransformCache();
                    for (LeafReaderContext ctx : reader.leaves()) {
                        TopDocs withoutSharing = search(ctx.reader(), query, QueryTransformCache.NONE);
                        TopDocs withSharing = search(ctx.reader(), query, transforms);
                        assertSameResults(withoutSharing, withSharing);
                    }
                    assertEquals("one projection for all segments sharing the projection matrix", 1, transforms.computed());
                }
            }
        }
    }

    public void testQueryIsProjectedPerDistinctProjection() throws IOException {
        float[][] basis = basis();
        try (Directory dir = newDirectory(); IndexWriter w = new IndexWriter(dir, writerConfig())) {
            int segments = randomIntBetween(2, 4);
            for (int s = 0; s < segments; s++) {
                flush(w, basis, FULL_SAMPLE);
            }
            Set<UUID> distinct = new HashSet<>(segmentMatrixIds(w));
            try (DirectoryReader reader = DirectoryReader.open(w)) {
                float[] query = vector(basis);
                CountingQueryTransformCache transforms = new CountingQueryTransformCache();
                for (LeafReaderContext ctx : reader.leaves()) {
                    assertSameResults(search(ctx.reader(), query, QueryTransformCache.NONE), search(ctx.reader(), query, transforms));
                }
                assertEquals(distinct.size(), transforms.computed());
            }
        }
    }

    private static TopDocs search(LeafReader leafReader, float[] query, QueryTransformCache transforms) throws IOException {
        int k = 10;
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(1.0f, k, k, null, transforms);
        TopKnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE, strategy);
        leafReader.searchNearestVectors(FIELD, query, collector, AcceptDocs.fromLiveDocs(leafReader.getLiveDocs(), leafReader.maxDoc()));
        return collector.topDocs();
    }

    private static void assertSameResults(TopDocs expected, TopDocs actual) {
        assertEquals(expected.scoreDocs.length, actual.scoreDocs.length);
        for (int i = 0; i < expected.scoreDocs.length; i++) {
            assertEquals(expected.scoreDocs[i].doc, actual.scoreDocs[i].doc);
            assertEquals(expected.scoreDocs[i].score, actual.scoreDocs[i].score, 0f);
        }
    }
}
