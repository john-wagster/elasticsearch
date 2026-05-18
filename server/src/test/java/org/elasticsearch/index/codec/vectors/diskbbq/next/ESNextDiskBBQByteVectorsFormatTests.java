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
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.cluster.KMeansFloatVectorValues;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.WriterVectorValues;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;
import org.junit.AssumptionViolatedException;
import org.junit.Before;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION;
import static org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat.MAX_CENTROIDS_PER_PARENT_CLUSTER;
import static org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat.MAX_PRECONDITIONING_BLOCK_DIMS;
import static org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat.MAX_VECTORS_PER_CLUSTER;
import static org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat.MIN_CENTROIDS_PER_PARENT_CLUSTER;
import static org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat.MIN_PRECONDITIONING_BLOCK_DIMS;
import static org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat.MIN_VECTORS_PER_CLUSTER;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;

/**
 * Tests for byte vector support with the ESNext IVF disk BBQ vectors format.
 * Byte vectors are converted to float at the codec boundary and processed through the IVF pipeline.
 */
public class ESNextDiskBBQByteVectorsFormatTests extends BaseKnnVectorsFormatTestCase {

    static {
        LogConfigurator.configureESLogging();
    }

    private KnnVectorsFormat format;

    @Before
    @Override
    public void setUp() throws Exception {
        ESNextDiskBBQVectorsFormat.QuantEncoding encoding = ESNextDiskBBQVectorsFormat.QuantEncoding.values()[random().nextInt(
            ESNextDiskBBQVectorsFormat.QuantEncoding.values().length
        )];
        if (rarely()) {
            format = new ESNextDiskBBQVectorsFormat(
                encoding,
                random().nextInt(2 * MIN_VECTORS_PER_CLUSTER, MAX_VECTORS_PER_CLUSTER),
                random().nextInt(8, MAX_CENTROIDS_PER_PARENT_CLUSTER),
                DenseVectorFieldMapper.ElementType.BYTE,
                false,
                null,
                1,
                false,
                DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
                null
            );
        } else if (rarely()) {
            format = new ESNextDiskBBQVectorsFormat(
                encoding,
                random().nextInt(MIN_VECTORS_PER_CLUSTER, MAX_VECTORS_PER_CLUSTER),
                random().nextInt(MIN_CENTROIDS_PER_PARENT_CLUSTER, MAX_CENTROIDS_PER_PARENT_CLUSTER),
                DenseVectorFieldMapper.ElementType.BYTE,
                false,
                null,
                1,
                true,
                random().nextInt(MIN_PRECONDITIONING_BLOCK_DIMS, MAX_PRECONDITIONING_BLOCK_DIMS),
                null
            );
        } else {
            // run with low numbers to force many clusters with parents
            format = new ESNextDiskBBQVectorsFormat(
                encoding,
                random().nextInt(MIN_VECTORS_PER_CLUSTER, 2 * MIN_VECTORS_PER_CLUSTER),
                random().nextInt(MIN_CENTROIDS_PER_PARENT_CLUSTER, 8),
                DenseVectorFieldMapper.ElementType.BYTE,
                false,
                null,
                1,
                false,
                DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
                null
            );
        }
        super.setUp();
    }

    @Override
    protected boolean supportsFloatVectorFallback() {
        return false;
    }

    @Override
    protected Codec getCodec() {
        return TestUtil.alwaysKnnVectorsFormat(format);
    }

    @Override
    protected VectorEncoding randomVectorEncoding() {
        return VectorEncoding.BYTE;
    }

    @Override
    protected VectorSimilarityFunction randomSimilarity() {
        return switch (random().nextInt(4)) {
            case 0 -> VectorSimilarityFunction.DOT_PRODUCT;
            case 1 -> VectorSimilarityFunction.EUCLIDEAN;
            case 2 -> VectorSimilarityFunction.COSINE;
            case 3 -> VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
            default -> throw new IllegalStateException("Unexpected value for similarity");
        };
    }

    @Override
    public void testSearchWithVisitedLimit() {
        throw new AssumptionViolatedException("ivf doesn't enforce visitation limit");
    }

    @Override
    public void testAdvance() throws Exception {
        // TODO re-enable with hierarchical IVF, clustering as it is is flaky
    }

    @Override
    protected void assertOffHeapByteSize(LeafReader r, String fieldName) throws IOException {
        var fieldInfo = r.getFieldInfos().fieldInfo(fieldName);
        if (r instanceof CodecReader codecReader) {
            KnnVectorsReader knnVectorsReader = codecReader.getVectorReader();
            if (knnVectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
                knnVectorsReader = fieldsReader.getFieldReader(fieldName);
            }
            var offHeap = knnVectorsReader.getOffHeapByteSize(fieldInfo);
            long totalByteSize = offHeap.values().stream().mapToLong(Long::longValue).sum();
            assertThat(offHeap.size(), equalTo(3));
            assertThat(totalByteSize, equalTo(offHeap.values().stream().mapToLong(Long::longValue).sum()));
        } else {
            throw new AssertionError("unexpected:" + r.getClass());
        }
    }

    public void testByteVectorIndexAndSearch() throws IOException {
        int dimensions = random().nextInt(12, 500);
        int numDocs = random().nextInt(100, 1_000);
        try (Directory dir = newDirectory(); IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
            for (int i = 0; i < numDocs; i++) {
                byte[] vector = randomByteVector(dimensions);
                Document doc = new Document();
                doc.add(new KnnByteVectorField("f", vector, VectorSimilarityFunction.EUCLIDEAN));
                w.addDocument(doc);
            }
            w.commit();
            if (random().nextBoolean()) {
                w.forceMerge(1);
            }
            try (IndexReader reader = DirectoryReader.open(w)) {
                List<LeafReaderContext> subReaders = reader.leaves();
                for (LeafReaderContext r : subReaders) {
                    LeafReader leafReader = r.reader();
                    byte[] queryVector = randomByteVector(dimensions);
                    TopDocs topDocs = leafReader.searchNearestVectors(
                        "f",
                        queryVector,
                        10,
                        AcceptDocs.fromLiveDocs(leafReader.getLiveDocs(), leafReader.maxDoc()),
                        Integer.MAX_VALUE
                    );
                    assertEquals(Math.min(leafReader.maxDoc(), 10), topDocs.scoreDocs.length);
                }
            }
        }
    }

    /**
     * Deterministic test for byte vectors with COSINE similarity and preconditioning enabled.
     * This combination exercises the most complex path: byte→float→normalize→precondition→quantize.
     */
    public void testByteVectorCosineWithPreconditioning() throws IOException {
        int dimensions = random().nextInt(32, 256);
        int numDocs = random().nextInt(200, 600);
        ESNextDiskBBQVectorsFormat.QuantEncoding encoding = ESNextDiskBBQVectorsFormat.QuantEncoding.values()[random().nextInt(
            ESNextDiskBBQVectorsFormat.QuantEncoding.values().length
        )];
        KnnVectorsFormat cosineFormat = new ESNextDiskBBQVectorsFormat(
            encoding,
            random().nextInt(MIN_VECTORS_PER_CLUSTER, MAX_VECTORS_PER_CLUSTER),
            random().nextInt(MIN_CENTROIDS_PER_PARENT_CLUSTER, MAX_CENTROIDS_PER_PARENT_CLUSTER),
            DenseVectorFieldMapper.ElementType.BYTE,
            false,
            null,
            1,
            true,  // doPrecondition = true (deterministic)
            random().nextInt(MIN_PRECONDITIONING_BLOCK_DIMS, MAX_PRECONDITIONING_BLOCK_DIMS),
            null
        );
        Codec codec = TestUtil.alwaysKnnVectorsFormat(cosineFormat);
        try (Directory dir = newDirectory(); IndexWriter w = new IndexWriter(dir, newIndexWriterConfig().setCodec(codec))) {
            for (int i = 0; i < numDocs; i++) {
                byte[] vector = randomNonZeroByteVector(dimensions);
                Document doc = new Document();
                doc.add(new KnnByteVectorField("f", vector, VectorSimilarityFunction.COSINE));
                w.addDocument(doc);
            }
            w.commit();
            w.forceMerge(1);
            try (IndexReader reader = DirectoryReader.open(w)) {
                for (LeafReaderContext r : reader.leaves()) {
                    LeafReader leafReader = r.reader();
                    byte[] queryVector = randomNonZeroByteVector(dimensions);
                    TopDocs topDocs = leafReader.searchNearestVectors(
                        "f",
                        queryVector,
                        10,
                        AcceptDocs.fromLiveDocs(leafReader.getLiveDocs(), leafReader.maxDoc()),
                        Integer.MAX_VALUE
                    );
                    assertEquals(Math.min(leafReader.maxDoc(), 10), topDocs.scoreDocs.length);
                }
            }
        }
    }

    /**
     * Tests that {@link IVFVectorsWriter#createWriterVectorValues} returns {@link WriterVectorValues.FloatValues}
     * for COSINE similarity (centroids are L2-normalized floats, rounding to bytes destroys precision),
     * returns FloatValues for float-backed values, and returns {@link WriterVectorValues.ByteValues}
     * only for non-COSINE byte-backed non-preconditioned values.
     */
    public void testCreateWriterVectorValuesCosineExclusion() {
        int dims = 32;
        List<byte[]> byteVectors = new ArrayList<>();
        List<float[]> floatVectors = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            byteVectors.add(randomByteVector(dims));
            floatVectors.add(new float[dims]);
            for (int j = 0; j < dims; j++) {
                floatVectors.get(i)[j] = byteVectors.get(i)[j];
            }
        }

        // Byte-backed, non-preconditioned — should return ByteValues for non-COSINE similarities
        KMeansFloatVectorValues byteBacked = KMeansFloatVectorValues.buildFromBytes(byteVectors, null, dims, false);
        assertTrue(byteBacked.isByteBacked());
        assertFalse(byteBacked.isPreconditioned());

        for (VectorSimilarityFunction sim : new VectorSimilarityFunction[] {
            VectorSimilarityFunction.EUCLIDEAN,
            VectorSimilarityFunction.DOT_PRODUCT,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT }) {
            FieldInfo fi = makeFieldInfo("test", dims, sim);
            assertThat(IVFVectorsWriter.createWriterVectorValues(fi, byteBacked), instanceOf(WriterVectorValues.ByteValues.class));
        }

        FieldInfo cosineFi = makeFieldInfo("test", dims, VectorSimilarityFunction.COSINE);
        assertThat(IVFVectorsWriter.createWriterVectorValues(cosineFi, byteBacked), instanceOf(WriterVectorValues.FloatValues.class));

        // Float-backed — should always return FloatValues
        KMeansFloatVectorValues floatBacked = KMeansFloatVectorValues.build(floatVectors, null, dims);
        assertFalse(floatBacked.isByteBacked());
        for (VectorSimilarityFunction sim : VectorSimilarityFunction.values()) {
            FieldInfo fi = makeFieldInfo("test", dims, sim);
            assertThat(IVFVectorsWriter.createWriterVectorValues(fi, floatBacked), instanceOf(WriterVectorValues.FloatValues.class));
        }
    }

    /**
     * Tests that {@link ESNextDiskBBQVectorsWriter#roundCentroidsToBytes} correctly rounds
     * float centroids to byte values clamped to [-128, 127].
     */
    public void testRoundCentroidsToBytes() {
        float[][] centroids = new float[][] {
            { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.4f, -0.4f },
            { 127.0f, 128.0f, 200.0f, -128.0f, -129.0f, -200.0f, 127.4f },
            { 127.6f, -128.6f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f }, };

        byte[][] result = ESNextDiskBBQVectorsWriter.roundCentroidsToBytes(centroids);

        assertEquals(3, result.length);

        // Row 0: normal rounding
        assertArrayEquals(new byte[] { 0, 1, -1, 1, 0, 0, 0 }, result[0]);

        // Row 1: clamping at boundaries
        assertArrayEquals(new byte[] { 127, 127, 127, -128, -128, -128, 127 }, result[1]);

        // Row 2: rounding .5 — Math.round rounds .5 up (toward positive infinity)
        assertArrayEquals(new byte[] { 127, -128, 1, 0, 2, -1, 3 }, result[2]);
    }

    private static byte[] randomByteVector(int dimensions) {
        byte[] vector = new byte[dimensions];
        random().nextBytes(vector);
        return vector;
    }

    /** Returns a random byte vector guaranteed to have non-zero L2 norm (required for COSINE). */
    private static byte[] randomNonZeroByteVector(int dimensions) {
        byte[] vector = new byte[dimensions];
        do {
            random().nextBytes(vector);
        } while (isZeroVector(vector));
        return vector;
    }

    private static boolean isZeroVector(byte[] v) {
        for (byte b : v) {
            if (b != 0) return false;
        }
        return true;
    }

    private static FieldInfo makeFieldInfo(String name, int dims, VectorSimilarityFunction similarity) {
        return new FieldInfo(
            name,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(),
            0,
            0,
            0,
            dims,
            VectorEncoding.BYTE,
            similarity,
            false,
            false
        );
    }
}
