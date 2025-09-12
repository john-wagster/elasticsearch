/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */
package org.elasticsearch.xpack.searchablesnapshots.store.input;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.Version;
import org.elasticsearch.common.blobstore.BlobContainer;
import org.elasticsearch.common.blobstore.OperationPurpose;
import org.elasticsearch.common.lucene.store.ESIndexInputTestCase;
import org.elasticsearch.common.unit.ByteSizeUnit;
import org.elasticsearch.common.unit.ByteSizeValue;
import org.elasticsearch.index.codec.vectors.diskbbq.DocIdsWriter;
import org.elasticsearch.index.snapshots.blobstore.BlobStoreIndexShardSnapshot.FileInfo;
import org.elasticsearch.index.store.StoreFileMetadata;
import org.elasticsearch.threadpool.ThreadPool;
import org.elasticsearch.xpack.searchablesnapshots.store.IndexInputStats;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

import static org.elasticsearch.xpack.searchablesnapshots.AbstractSearchableSnapshotsTestCase.randomIOContext;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class DirectBlobContainerIndexInputTests extends ESIndexInputTestCase {

    private DirectBlobContainerIndexInput createIndexInput(
        long minimumReadSize,
        Runnable onReadBlob
    ) throws IOException {
        final String fileName = randomAlphaOfLength(5) + randomFileExtension();

        // write some data to a tmp file
        final long len;
        byte[] pretendBlobBytes;
        int blockSize = 16;
//        int[] docIDs = new int[250 + random().nextInt(5000000)];
        int[] docIDs = new int[5000000];
//        final int bpv = TestUtil.nextInt(random(), 1, 16);
        final int bpv = 16;
        for (int i = 0; i < docIDs.length; ++i) {
            docIDs[i] = TestUtil.nextInt(random(), 0, (1 << bpv) - 1);
        }
        Arrays.sort(docIDs);
        int[] deltaEncodedIds = new int[docIDs.length];
        deltaEncodedIds[0] = docIDs[0];
        for (int i = 1; i < docIDs.length; i++) {
            deltaEncodedIds[i] = docIDs[i] - docIDs[i - 1];
        }
        try (Directory dir = newDirectory()) {
            DocIdsWriter docIdsWriter = new DocIdsWriter();
            try (IndexOutput out = dir.createOutput("tmp", IOContext.DEFAULT)) {

                byte encoding = docIdsWriter.calculateBlockEncoding(i -> deltaEncodedIds[i], deltaEncodedIds.length, blockSize);
                assert encoding == (byte) 16;
                out.writeByte(encoding);
                int limit = deltaEncodedIds.length - blockSize + 1;
                int i = 0;
                for (; i < limit; i += blockSize) {
                    int offset = i;
                    docIdsWriter.writeDocIds(d -> deltaEncodedIds[d + offset], blockSize, encoding, out);
                }
                // handle tail
                if (i < deltaEncodedIds.length) {
                    int offset = i;
                    docIdsWriter.writeDocIds(d -> deltaEncodedIds[d + offset], deltaEncodedIds.length - i, encoding, out);
                }
                len = out.getFilePointer();
//                if (random().nextBoolean()) {
//                    out.writeLong(0); // garbage
//                }
            }

            // overwrite input
            try (IndexInput in = dir.openInput("tmp", IOContext.READONCE)) {
                pretendBlobBytes = new byte[(int) len];
                in.readBytes(pretendBlobBytes, 0, (int) len);
            }

            dir.deleteFile("tmp");
        }

        long partSize = len;

        // form up a fake blob so we can read back the data in the tmp file with a BlobCacheBufferedIndexInput and it's readVInt method
        final BlobContainer blobContainer = mock(BlobContainer.class);
        when(blobContainer.readBlob(any(OperationPurpose.class), anyString(), anyLong(), anyLong())).thenAnswer(invocationOnMock -> {
            String name = (String) invocationOnMock.getArguments()[1];
            long position = (long) invocationOnMock.getArguments()[2];
            long length = (long) invocationOnMock.getArguments()[3];
            assertThat(
                "Reading [" + length + "] bytes from [" + name + "] at [" + position + "] exceeds part size [" + partSize + "]",
                position + length,
                lessThanOrEqualTo(partSize)
            );

            onReadBlob.run();

            return new ByteArrayInputStream(pretendBlobBytes, (int) position, (int) length);
        });

//        final FileInfo fileInfo = new FileInfo(
//            randomAlphaOfLength(5),
//            new StoreFileMetadata(fileName, len, "NO CHECKSUM", Version.LATEST.toString()),
//            randomFrom(ByteSizeValue.of(partSize, ByteSizeUnit.BYTES),
//            ByteSizeValue.of(randomLongBetween(partSize, Long.MAX_VALUE), ByteSizeUnit.BYTES),
//            ByteSizeValue.ZERO,
//            ByteSizeValue.of(-1, ByteSizeUnit.BYTES),
//            null
//        )
//        );

        final FileInfo fileInfo = new FileInfo(
            randomAlphaOfLength(5),
            new StoreFileMetadata(fileName, len, "NO CHECKSUM", Version.LATEST.toString()),
            ByteSizeValue.of(partSize, ByteSizeUnit.BYTES)
        );

        final DirectBlobContainerIndexInput indexInput = new DirectBlobContainerIndexInput(
            fileName,
            blobContainer,
            fileInfo,
            randomIOContext(),
            new IndexInputStats(1L, len, len, len, () -> 0L),
            minimumReadSize
        );

        DocIdsWriter docIdsWriter = new DocIdsWriter();
        int[] read = new int[deltaEncodedIds.length];
        int[] block = new int[blockSize];
        int limit = deltaEncodedIds.length - blockSize + 1;
        byte encoding = indexInput.readByte();
        int i = 0;
        for (; i < limit; i += blockSize) {
            int offset = i;
            docIdsWriter.readInts(indexInput, blockSize, encoding, block);
            System.arraycopy(block, 0, read, offset, blockSize);
        }
        // handle tail
        if (i < deltaEncodedIds.length) {
            int offset = i;
            docIdsWriter.readInts(indexInput, deltaEncodedIds.length - i, encoding, block);
            System.arraycopy(block, 0, read, offset, deltaEncodedIds.length - i);
        }
        assertArrayEquals(deltaEncodedIds, read);
        assertEquals(len, indexInput.getFilePointer());

//        assertEquals(input.length, indexInput.length());
        return indexInput;
    }

    public void testRandomReadsMod() throws IOException {

        try(ExecutorService es = Executors.newFixedThreadPool(32)) {
            for (int i = 0; i < 10000; i++) {
                es.submit(() -> {
                    try {
                        createIndexInput(
                            randomIntBetween(1, 1000),
                            () -> {
                            }
                        );
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });
            }
        }
    }
//
//    public void testRandomReads() throws IOException {
//        for (int i = 0; i < 100; i++) {
//            final Tuple<String, byte[]> bytes = randomChecksumBytes(randomIntBetween(1, 1000));
//            final byte[] input = bytes.v2();
//
//            final DirectBlobContainerIndexInput indexInput = createIndexInput(bytes);
//            assertEquals(input.length, indexInput.length());
//            assertEquals(0, indexInput.getFilePointer());
//            byte[] output = randomReadAndSlice(indexInput, input.length);
//            assertArrayEquals(input, output);
//        }
//    }
//
//    public void testCloneAndLargeRead() throws IOException {
//        final Tuple<String, byte[]> bytes = randomChecksumBytes(between(ByteSizeUnit.KB.toIntBytes(2), ByteSizeUnit.KB.toIntBytes(10)));
//        try (var indexInput = createIndexInput(bytes)) {
//            indexInput.readLong();
//
//            final var clone = indexInput.clone();
//
//            // do a read which is large enough to exercise the path which bypasses the buffer and fills the output directly
//
//            final var originalBytes = new byte[2048];
//            indexInput.readBytes(originalBytes, 0, originalBytes.length);
//
//            final var cloneBytes = new byte[originalBytes.length];
//            clone.readBytes(cloneBytes, 0, cloneBytes.length);
//
//            assertArrayEquals(originalBytes, cloneBytes);
//        }
//    }
//
//    public void testRandomOverflow() throws IOException {
//        for (int i = 0; i < 100; i++) {
//            final Tuple<String, byte[]> bytes = randomChecksumBytes(randomIntBetween(1, 1000));
//            final byte[] input = bytes.v2();
//
//            final DirectBlobContainerIndexInput indexInput = createIndexInput(bytes);
//            int firstReadLen = randomIntBetween(0, input.length - 1);
//            randomReadAndSlice(indexInput, firstReadLen);
//            int bytesLeft = input.length - firstReadLen;
//            int secondReadLen = bytesLeft + randomIntBetween(1, 100);
//            expectThrows(EOFException.class, () -> indexInput.readBytes(new byte[secondReadLen], 0, secondReadLen));
//        }
//    }
//
//    public void testSeekOverflow() throws IOException {
//        for (int i = 0; i < 100; i++) {
//            final Tuple<String, byte[]> bytes = randomChecksumBytes(randomIntBetween(1, 1000));
//            final byte[] input = bytes.v2();
//
//            final DirectBlobContainerIndexInput indexInput = createIndexInput(bytes);
//            int firstReadLen = randomIntBetween(0, input.length - 1);
//            randomReadAndSlice(indexInput, firstReadLen);
//            expectThrows(IOException.class, () -> {
//                switch (randomIntBetween(0, 2)) {
//                    case 0 -> indexInput.seek(Integer.MAX_VALUE + 4L);
//                    case 1 -> indexInput.seek(-randomIntBetween(1, 10));
//                    default -> {
//                        int seek = input.length + randomIntBetween(1, 100);
//                        indexInput.seek(seek);
//                    }
//                }
//            });
//        }
//    }
//
//    public void testSequentialReadsShareInputStreamFromBlobStore() throws IOException {
//        for (int i = 0; i < 100; i++) {
//            final Tuple<String, byte[]> bytes = randomChecksumBytes(randomIntBetween(1, 1000));
//            final byte[] input = bytes.v2();
//
//            final int minimumReadSize = randomIntBetween(1, 1000);
//            final int partSize = randomBoolean() ? input.length : randomIntBetween(1, input.length);
//            final String checksum = bytes.v1();
//
//            final AtomicInteger readBlobCount = new AtomicInteger();
//            final DirectBlobContainerIndexInput indexInput = createIndexInput(
//                input,
//                partSize,
//                minimumReadSize,
//                checksum,
//                readBlobCount::incrementAndGet
//            );
//
//            assertEquals(input.length, indexInput.length());
//
//            final int readStart = randomIntBetween(0, input.length);
//            final int readEnd = randomIntBetween(readStart, input.length);
//            final int readLen = readEnd - readStart;
//
//            indexInput.seek(readStart);
//
//            // Straightforward sequential reading from `indexInput` (no cloning, slicing or seeking)
//            final byte[] output = new byte[readLen];
//            int readPos = readStart;
//            while (readPos < readEnd) {
//                if (randomBoolean()) {
//                    output[readPos++ - readStart] = indexInput.readByte();
//                } else {
//                    int len = randomIntBetween(1, readEnd - readPos);
//                    indexInput.readBytes(output, readPos - readStart, len);
//                    readPos += len;
//                }
//            }
//            assertEquals(readEnd, readPos);
//            assertEquals(readEnd, indexInput.getFilePointer());
//
//            final byte[] expected = new byte[readLen];
//            System.arraycopy(input, readStart, expected, 0, readLen);
//            assertArrayEquals(expected, output);
//
//            // compute the maximum expected number of ranges read from the blob store
//            final int firstPart = readStart / partSize;
//            final int bufferedEnd = readEnd + indexInput.getBufferSize() - 1;
//            final int lastPart = (bufferedEnd - 1) / partSize; // may overshoot a part due to buffering but not due to readahead
//
//            final int expectedRanges;
//            if (firstPart == lastPart) {
//                final int bufferedBytes = bufferedEnd - readStart;
//                expectedRanges = (bufferedBytes + minimumReadSize - 1) / minimumReadSize; // ceil(bufferedBytes/minimumReadSize)
//            } else {
//                // read was split across parts; each part involves at least one range
//
//                final int bytesInFirstPart = (firstPart + 1) * partSize - readStart;
//                // ceil(bytesInFirstPart/minimumReadSize)
//                final int rangesInFirstPart = (bytesInFirstPart + minimumReadSize - 1) / minimumReadSize;
//
//                final int bytesInLastPart = bufferedEnd - lastPart * partSize;
//                // ceil(bytesInLastPart/minimumReadSize)
//                final int rangesInLastPart = (bytesInLastPart + minimumReadSize - 1) / minimumReadSize;
//
//                // ceil(partSize/minimumReadSize);
//                final int rangesInMiddleParts = (partSize + minimumReadSize - 1) / minimumReadSize;
//                final int middlePartCount = lastPart - firstPart - 1;
//
//                expectedRanges = rangesInFirstPart + rangesInLastPart + rangesInMiddleParts * middlePartCount;
//            }
//
//            assertThat(
//                "data was read in ranges of no less than " + minimumReadSize + " where possible",
//                readBlobCount.get(),
//                lessThanOrEqualTo(expectedRanges)
//            );
//        }
//    }

}
