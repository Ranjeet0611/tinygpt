package com.tinygpt.data;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GPTDataLoaderTest {

    private static final int[] TOKENS = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    @Test
    void targetIsInputShiftedByOne() {
        GPTDataset dataset = new GPTDataset(TOKENS, 4, 4);
        assertEquals(2, dataset.size());
        assertArrayEquals(new int[]{10, 11, 12, 13}, dataset.input(0));
        assertArrayEquals(new int[]{11, 12, 13, 14}, dataset.target(0));
        assertArrayEquals(new int[]{14, 15, 16, 17}, dataset.input(1));
    }

    @Test
    void strideOfOneMakesOverlappingWindows() {
        GPTDataset dataset = new GPTDataset(TOKENS, 4, 1);
        assertEquals(6, dataset.size());
        assertArrayEquals(new int[]{11, 12, 13, 14}, dataset.input(1));
    }

    @Test
    void batchesAndDropLast() {
        GPTDataset dataset = new GPTDataset(TOKENS, 2, 1);   // 8 examples
        GPTDataLoader dropLast = new GPTDataLoader(dataset, 3, false, true);
        GPTDataLoader keepLast = new GPTDataLoader(dataset, 3, false, false);
        assertEquals(2, count(dropLast));
        assertEquals(3, count(keepLast));
        assertEquals(dropLast.numberOfBatches(), count(dropLast));
        assertEquals(keepLast.numberOfBatches(), count(keepLast));
    }

    private static int count(GPTDataLoader loader) {
        int batches = 0;
        for (Batch ignored : loader) {
            batches++;
        }
        return batches;
    }
}
