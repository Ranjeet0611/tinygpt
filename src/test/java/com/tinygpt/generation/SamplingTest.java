package com.tinygpt.generation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SamplingTest {

    private static final float[] SCORES = {4.51f, 0.89f, -1.90f, 6.75f, 1.63f, -1.62f, -1.89f, 6.28f, 1.79f};

    @Test
    void softmaxAddsUpToOne() {
        float sum = 0;
        for (float p : TextGenerator.softmax(SCORES, 1f)) {
            sum += p;
        }
        assertEquals(1f, sum, 1e-5);
    }

    @Test
    void lowTemperatureMakesTheTopChoiceAlmostCertain() {
        float[] cold = TextGenerator.softmax(SCORES, 0.1f);
        assertTrue(cold[3] > 0.99f);
    }

    @Test
    void topKKeepsOnlyTheBestScores() {
        float[] probabilities = TextGenerator.softmax(TextGenerator.keepTopK(SCORES, 3), 1f);
        int nonZero = 0;
        for (float p : probabilities) {
            if (p > 0) {
                nonZero++;
            }
        }
        assertEquals(3, nonZero);
        assertEquals(3, TextGenerator.argMax(SCORES));
    }
}
