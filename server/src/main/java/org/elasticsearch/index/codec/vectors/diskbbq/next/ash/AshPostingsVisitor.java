/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq.next.ash;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.elasticsearch.index.codec.vectors.diskbbq.DocIdsWriter;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;
import org.elasticsearch.search.vectors.BulkKnnCollector;

import static org.elasticsearch.simdvec.ES940OSQVectorsScorer.BULK_SIZE;

/**
 * PostingVisitor for ASH-encoded posting lists.
 * <p>
 * Reads bit-packed codes with float16 scale/offset per vector and scores them
 * asymmetrically using the precomputed query transforms. Each vector stores a
 * 1-byte ASH cluster ID indicating which centering centroid was used during
 * encoding, allowing the scorer to select the correct transformed query vector.
 * <p>
 * The on-disk per-vector format is:
 * {@code [byte ashClusterId][short scale_f16][short offset_f16][byte[packedCodeBytes] packed_codes]}
 */
public class AshPostingsVisitor implements IVFVectorsReader.PostingVisitor {
    private final float[][] w;
    private final float[][] ashCentroids;
    private final float[] query;
    private final IndexInput parentsSlice;
    private final float[] globalCentroid;
    private final FieldInfo fieldInfo;
    private final IndexInput indexInput;
    private final Bits acceptDocs;
    private final int nDims;
    private final int bitsPerDim;
    private final int packedCodeBytes;
    private final DocIdsWriter idsWriter = new DocIdsWriter();
    private final int[] docIdsScratch = new int[BULK_SIZE];
    private final int[] offsetsScratch = new int[BULK_SIZE];
    private final float[] scores = new float[BULK_SIZE];

    // Per-ASH-cluster precomputed query transforms (lazily populated)
    private final float[] queryTransformed;
    private final float[] queryDotCentroidByCluster;
    private boolean clusterTransformsReady;
    private float currentQueryDotCentroid;

    // Per-posting-list state
    private int vectors;
    private byte docEncoding;
    private int docBase;
    private long slicePos;
    private float centroidDistance;
    private final org.apache.lucene.index.VectorSimilarityFunction similarityFunction;
    private final float[] currentCentroid;

    public AshPostingsVisitor(
        float[][] w,
        float[][] ashCentroids,
        float[] query,
        IndexInput parentsSlice,
        float[] globalCentroid,
        FieldInfo fieldInfo,
        IndexInput indexInput,
        Bits acceptDocs,
        int bitsPerDim
    ) {
        this.w = w;
        this.ashCentroids = ashCentroids;
        this.query = query;
        this.parentsSlice = parentsSlice;
        this.globalCentroid = globalCentroid;
        this.fieldInfo = fieldInfo;
        this.indexInput = indexInput;
        this.acceptDocs = acceptDocs;
        this.nDims = w[0].length;
        this.bitsPerDim = bitsPerDim;
        this.packedCodeBytes = AsymmetricHashingScorer.packedByteLength(nDims, bitsPerDim);
        this.similarityFunction = fieldInfo.getVectorSimilarityFunction();
        this.currentCentroid = new float[fieldInfo.getVectorDimension()];

        // Pre-allocate per-ASH-cluster arrays
        int nAshClusters = ashCentroids != null ? ashCentroids.length : 0;
        this.queryTransformed = new float[nDims];
        this.queryDotCentroidByCluster = new float[nAshClusters];
        this.clusterTransformsReady = false;

        for (int j = 0; j < nDims; j++) {
            double sum = 0;
            for (int d = 0; d < query.length; d++) {
                sum += (double) query[d] * w[d][j];
            }
            queryTransformed[j] = (float) sum;
        }
    }

    @Override
    public int resetPostingsScorer(PostingMetadata metadata) throws java.io.IOException {
        float score = metadata.documentCentroidScore();
        indexInput.seek(metadata.offset());
        float centroidToParentSqDist = Float.intBitsToFloat(indexInput.readInt());
        indexInput.readFloats(currentCentroid, 0, currentCentroid.length);
        vectors = indexInput.readVInt();
        docEncoding = indexInput.readByte();
        docBase = 0;
        slicePos = indexInput.getFilePointer();

        centroidDistance = switch (similarityFunction) {
            case EUCLIDEAN -> ((1 / score) - 1) - centroidToParentSqDist;
            case COSINE, DOT_PRODUCT -> 2 * score - 1;
            case MAXIMUM_INNER_PRODUCT -> score - 1;
        };

        double dot = 0;
        for (int d = 0; d < query.length; d++) {
            dot += (double) query[d] * currentCentroid[d];
        }
        currentQueryDotCentroid = (float) dot;

        return vectors;
    }

    @Override
    public int visit(KnnCollector knnCollector) throws java.io.IOException {
        indexInput.seek(slicePos);
        int scoredDocs = 0;
        int perVectorBytes = Byte.BYTES + Short.BYTES + Short.BYTES + packedCodeBytes;
        byte[] codeBuf = new byte[packedCodeBytes];

        int limit = vectors - BULK_SIZE + 1;
        int i = 0;
        for (; i < limit; i += BULK_SIZE) {
            readDocIds(BULK_SIZE);
            int docsToBulkScore = docToBulkScore(BULK_SIZE);
            if (docsToBulkScore == 0) {
                indexInput.skipBytes((long) perVectorBytes * BULK_SIZE);
                continue;
            }
            float maxScore = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < BULK_SIZE; j++) {
                if (docIdsScratch[j] != -1) {
                    float scale = Float.float16ToFloat(indexInput.readShort());
                    float offset = Float.float16ToFloat(indexInput.readShort());
                    indexInput.readBytes(codeBuf, 0, packedCodeBytes);
                    float s = scoreVector(codeBuf, scale, offset);
                    scores[j] = convertScore(s);
                    if (scores[j] > maxScore) {
                        maxScore = scores[j];
                    }
                } else {
                    indexInput.skipBytes(perVectorBytes);
                }
            }
            if (knnCollector.minCompetitiveSimilarity() < maxScore) {
                collectBulk(knnCollector, BULK_SIZE, docsToBulkScore, maxScore);
            }
            scoredDocs += docsToBulkScore;
        }
        // Tail
        if (i < vectors) {
            int tailSize = vectors - i;
            readDocIds(tailSize);
            int docsToBulkScore = docToBulkScore(tailSize);
            if (docsToBulkScore > 0) {
                float maxScore = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < tailSize; j++) {
                    if (docIdsScratch[j] != -1) {
                        float scale = Float.float16ToFloat(indexInput.readShort());
                        float offset = Float.float16ToFloat(indexInput.readShort());
                        indexInput.readBytes(codeBuf, 0, packedCodeBytes);
                        float s = scoreVector(codeBuf, scale, offset);
                        scores[j] = convertScore(s);
                        if (scores[j] > maxScore) {
                            maxScore = scores[j];
                        }
                    } else {
                        indexInput.skipBytes(perVectorBytes);
                    }
                }
                if (knnCollector.minCompetitiveSimilarity() < maxScore) {
                    collectBulk(knnCollector, tailSize, docsToBulkScore, maxScore);
                }
                scoredDocs += docsToBulkScore;
            }
        }
        if (scoredDocs > 0) {
            knnCollector.incVisitedCount(scoredDocs);
        }
        return scoredDocs;
    }

    private float convertScore(float rawDotProduct) {
        return switch (similarityFunction) {
            case EUCLIDEAN -> 1 / (1 + rawDotProduct);
            case COSINE, DOT_PRODUCT -> (1 + rawDotProduct) / 2;
            case MAXIMUM_INNER_PRODUCT -> rawDotProduct >= 0 ? rawDotProduct + 1 : 1 / (1 - rawDotProduct);
        };
    }

    private float scoreVector(byte[] codeBuf, float scale, float offset) {
        if (bitsPerDim == 1) {
            return AsymmetricHashingScorer.scoreOneVectorBinary(queryTransformed, currentQueryDotCentroid, codeBuf, nDims, scale, offset);
        } else {
            return AsymmetricHashingScorer.scoreOneVectorMultiBit(
                queryTransformed,
                currentQueryDotCentroid,
                codeBuf,
                nDims,
                bitsPerDim,
                scale,
                offset
            );
        }
    }

    private void readDocIds(int count) throws java.io.IOException {
        idsWriter.readInts(indexInput, count, docEncoding, docIdsScratch);
        for (int j = 0; j < count; j++) {
            docBase += docIdsScratch[j];
            docIdsScratch[j] = docBase;
        }
    }

    private int docToBulkScore(int bulkSize) {
        if (acceptDocs == null) {
            return bulkSize;
        }
        int docToScore = 0;
        for (int ii = 0; ii < bulkSize; ii++) {
            if (docIdsScratch[ii] == -1 || acceptDocs.get(docIdsScratch[ii]) == false) {
                docIdsScratch[ii] = -1;
            } else {
                offsetsScratch[docToScore] = ii;
                docToScore++;
            }
        }
        return docToScore;
    }

    private void collectBulk(KnnCollector knnCollector, int bulkSize, int docsToBulkScore, float maxScore) {
        if (knnCollector instanceof BulkKnnCollector bulkCollector) {
            if (docsToBulkScore == bulkSize) {
                bulkCollector.bulkCollect(docIdsScratch, scores, bulkSize, maxScore);
                return;
            }
            for (int ii = 0; ii < docsToBulkScore; ii++) {
                int offset = offsetsScratch[ii];
                docIdsScratch[ii] = docIdsScratch[offset];
                scores[ii] = scores[offset];
            }
            bulkCollector.bulkCollect(docIdsScratch, scores, docsToBulkScore, maxScore);
            return;
        }
        for (int ii = 0; ii < bulkSize; ii++) {
            final int doc = docIdsScratch[ii];
            if (doc != -1) {
                knnCollector.collect(doc, scores[ii]);
            }
        }
    }
}
