package com.tinygpt.attention;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.tinygpt.nn.Dropout;
import com.tinygpt.nn.Linear;
import com.tinygpt.nn.Module;

/**
 * Causal attention = self-attention that is <b>not allowed to look into the future</b>.
 *
 * When the model guesses the next word, peeking at it would be cheating.
 * So we hide every word to the right with a mask:
 *
 * <pre>
 *                Your journey starts with
 *   Your       [  ✓      ✗       ✗     ✗  ]
 *   journey    [  ✓      ✓       ✗     ✗  ]
 *   starts     [  ✓      ✓       ✓     ✗  ]
 *   with       [  ✓      ✓       ✓     ✓  ]
 * </pre>
 *
 * We set the ✗ scores to -infinity. After softmax, e^(-infinity) = 0, so they get 0% attention.
 * Dropout then randomly hides some of the remaining attention weights during training.
 *
 * Book: Chapter 3.5
 *
 * Picture guide: docs/classes/4-attention.md#causalattention
 */
public class CausalAttention extends Module {

    private final Linear query;
    private final Linear key;
    private final Linear value;
    private final Dropout dropout;

    public CausalAttention(NDManager manager, int inputDim, int outputDim, float dropoutRate) {
        query = child("query", new Linear(manager, inputDim, outputDim, false));
        key = child("key", new Linear(manager, inputDim, outputDim, false));
        value = child("value", new Linear(manager, inputDim, outputDim, false));
        dropout = child("dropout", new Dropout(dropoutRate));
    }

    /** inputs [batch, tokens, inputDim] -> context vectors [batch, tokens, outputDim] */
    public NDArray forward(NDArray inputs) {
        NDArray queries = query.forward(inputs);
        NDArray keys = key.forward(inputs);
        NDArray values = value.forward(inputs);

        int tokens = (int) inputs.getShape().get(1);
        NDArray scores = queries.matMul(keys.transpose(0, 2, 1));
        NDArray masked = scores.add(causalMask(inputs.getManager(), tokens));

        long keyDim = keys.getShape().getLastDimension();
        NDArray weights = dropout.forward(masked.div(Math.sqrt(keyDim)).softmax(-1));
        return weights.matMul(values);
    }

    /**
     * A [tokens, tokens] grid: 0 where looking is allowed, -infinity above the diagonal (the future).
     * <pre>
     *   [ 0  -∞  -∞ ]
     *   [ 0   0  -∞ ]
     *   [ 0   0   0 ]
     * </pre>
     */
    public static NDArray causalMask(NDManager manager, int tokens) {
        float[][] mask = new float[tokens][tokens];
        for (int row = 0; row < tokens; row++) {
            for (int col = row + 1; col < tokens; col++) {
                mask[row][col] = Float.NEGATIVE_INFINITY;
            }
        }
        return manager.create(mask);
    }
}
