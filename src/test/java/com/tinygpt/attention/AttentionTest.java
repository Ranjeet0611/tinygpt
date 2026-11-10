package com.tinygpt.attention;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class AttentionTest {

    private NDManager manager;

    @BeforeEach
    void setUp() {
        Engine.getInstance().setRandomSeed(123);
        manager = NDManager.newBaseManager();
    }

    @AfterEach
    void tearDown() {
        manager.close();
    }

    @Test
    void simpleAttentionWeightsSumToOne() {
        NDArray inputs = manager.randomNormal(new Shape(6, 3));
        float[] rowSums = SimpleAttention.weights(SimpleAttention.scores(inputs)).sum(new int[]{-1}).toFloatArray();
        for (float sum : rowSums) {
            assertEquals(1f, sum, 1e-5);
        }
    }

    @Test
    void causalMaskHidesTheFuture() {
        NDArray scores = manager.randomNormal(new Shape(5, 5));
        NDArray weights = scores.add(CausalAttention.causalMask(manager, 5)).softmax(-1);
        for (int row = 0; row < 5; row++) {
            for (int col = row + 1; col < 5; col++) {
                assertEquals(0f, weights.getFloat(row, col));
            }
        }
    }

    @Test
    void attentionModulesKeepBatchAndTokens() {
        NDArray batch = manager.randomNormal(new Shape(2, 6, 3));
        assertEquals(new Shape(2, 6, 2), new CausalAttention(manager, 3, 2, 0f).forward(batch).getShape());
        assertEquals(new Shape(2, 6, 4), new MultiHeadAttention(manager, 3, 4, 2, 0f, false).forward(batch).getShape());
    }

    @Test
    void earlierTokensAreNotAffectedByLaterTokens() {
        MultiHeadAttention attention = new MultiHeadAttention(manager, 4, 4, 2, 0f, false);
        NDArray original = manager.randomNormal(new Shape(1, 5, 4));
        NDArray changedLast = original.duplicate();
        changedLast.set(new NDIndex("0, 4"), 99f);

        float[] before = attention.forward(original).get("0, 0:4").toFloatArray();
        float[] after = attention.forward(changedLast).get("0, 0:4").toFloatArray();
        assertArrayEquals(before, after, 1e-6f);
    }
}
