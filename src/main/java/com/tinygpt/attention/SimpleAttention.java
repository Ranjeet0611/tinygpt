package com.tinygpt.attention;

import ai.djl.ndarray.NDArray;

/**
 * The simplest attention: no learnable weights, just "how similar are two words?".
 *
 * <pre>
 *   1. scores   = inputs × inputsᵀ      how much does each word match every other word (dot product)
 *   2. weights  = softmax(scores)       turn scores into percentages that add up to 100%
 *   3. context  = weights × inputs      each word becomes a mix of all words, by those percentages
 * </pre>
 *
 * Example for the word "journey" in "Your journey starts with one step":
 * <pre>
 *   Your 14% | journey 24% | starts 23% | with 12% | one 11% | step 16%
 *   new "journey" = 0.14·Your + 0.24·journey + 0.23·starts + ...
 * </pre>
 *
 * So after attention, each word "knows" a bit about the words around it.
 *
 * Book: Chapter 3.3
 *
 * Picture guide: docs/classes/4-attention.md#simpleattention · line by line: docs/walkthroughs/step05-simple-attention.md
 */
public final class SimpleAttention {

    private SimpleAttention() {}

    /** inputs [tokens, dim] -> scores [tokens, tokens] */
    public static NDArray scores(NDArray inputs) {
        return inputs.matMul(inputs.transpose());
    }

    /** Each row becomes percentages that sum to 1. */
    public static NDArray weights(NDArray scores) {
        return scores.softmax(-1);
    }

    /** inputs [tokens, dim] -> context vectors [tokens, dim] */
    public static NDArray contextVectors(NDArray inputs) {
        return weights(scores(inputs)).matMul(inputs);
    }
}
