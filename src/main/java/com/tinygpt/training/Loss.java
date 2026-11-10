package com.tinygpt.training;

import ai.djl.ndarray.NDArray;

/**
 * Cross-entropy loss: one number that says "how wrong was the model?". Lower is better.
 *
 * <pre>
 *   The right next word was "you".
 *   The model gave "you" a probability of ...
 *       0.90  ─►  loss = -ln(0.90) = 0.11   (good guess, small loss)
 *       0.01  ─►  loss = -ln(0.01) = 4.61   (bad guess, big loss)
 * </pre>
 *
 * We do this for every position in every example and take the average.
 * Training is just: change the weights a tiny bit so this number goes down. Repeat.
 *
 * A useful way to read it: e^loss is the "perplexity", roughly how many words
 * the model is still unsure between. A model that knows nothing starts near
 * ln(50,257) ≈ 10.8, i.e. every token is equally likely.
 *
 * Book: Chapter 5.1
 *
 * Picture guide: docs/classes/6-training.md#loss
 */
public final class Loss {

    private Loss() {}

    /**
     * @param logits  [batch, tokens, vocabularySize] scores from the model
     * @param targets [batch, tokens] the correct next-token IDs
     */
    public static NDArray crossEntropy(NDArray logits, NDArray targets) {
        NDArray logProbabilities = logits.logSoftmax(-1);

        // A "one-hot" row is all zeros except a 1 at the correct token:
        //   target 2, vocabulary of 5  ->  [0, 0, 1, 0, 0]
        // Multiplying by it and summing picks out the log-probability of the right answer.
        // (NDArray.gather does the same, but its backward pass gave random NaNs on multi-threaded CPU.)
        NDArray oneHot = targets.oneHot((int) logits.getShape().getLastDimension());
        NDArray correctLogProbability = logProbabilities.mul(oneHot).sum(new int[]{-1});

        return correctLogProbability.mean().neg();
    }
}
