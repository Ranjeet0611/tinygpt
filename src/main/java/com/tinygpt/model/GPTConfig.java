package com.tinygpt.model;

/**
 * All the "size settings" of a GPT model in one place.
 *
 * @param vocabularySize  how many different tokens exist (GPT-2: 50,257)
 * @param contextLength   the longest text the model can read at once, in tokens
 * @param embeddingDim    how many numbers describe each token
 * @param numberOfHeads   attention heads per transformer block
 * @param numberOfLayers  how many transformer blocks are stacked
 * @param dropoutRate     share of numbers switched off during training (0.1 = 10%)
 * @param qkvBias         whether the Query/Key/Value layers have a bias
 *
 * Book: Chapter 4.1
 *
 * Picture guide: docs/classes/5-model.md#gptconfig · line by line: docs/walkthroughs/step09-gpt-model.md
 */
public record GPTConfig(
        int vocabularySize,
        int contextLength,
        int embeddingDim,
        int numberOfHeads,
        int numberOfLayers,
        float dropoutRate,
        boolean qkvBias) {

    /** The real GPT-2 "small" (124 million parameters), as used in the book. Too slow to train on a laptop. */
    public static final GPTConfig GPT2_SMALL = new GPTConfig(50257, 1024, 768, 12, 12, 0.1f, false);

    /** Our tiny version (about 13 million parameters). Trains in a few minutes on a laptop CPU. */
    public static final GPTConfig TINY = new GPTConfig(50257, 64, 128, 4, 4, 0.1f, false);
}
