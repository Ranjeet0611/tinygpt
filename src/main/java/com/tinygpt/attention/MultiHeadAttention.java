package com.tinygpt.attention;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.tinygpt.nn.Dropout;
import com.tinygpt.nn.Linear;
import com.tinygpt.nn.Module;

/**
 * Several causal attentions ("heads") running side by side, each looking for something different.
 *
 * <pre>
 *   head 1: "which word is the subject?"
 *   head 2: "what was the last adjective?"
 *   head 3: "is there a name nearby?"
 *   ...
 * </pre>
 *
 * Instead of building N separate attention modules, we do one big Q, K, V and cut it into N slices:
 *
 * <pre>
 *   x [batch, tokens, dim]
 *     ─► Q, K, V          [batch, tokens, dim]
 *     ─► split into heads [batch, heads, tokens, headDim]    (headDim = dim / heads)
 *     ─► causal attention inside every head at the same time
 *     ─► glue heads back  [batch, tokens, dim]
 *     ─► output Linear    [batch, tokens, dim]                (lets the heads share what they found)
 * </pre>
 *
 * This is the attention used inside the GPT model.
 *
 * Book: Chapter 3.6
 *
 * Picture guide: docs/classes/4-attention.md#multiheadattention · line by line: docs/walkthroughs/step08-multi-head-attention.md
 */
public class MultiHeadAttention extends Module {

    private final int numberOfHeads;
    private final int headDim;
    private final Linear query;
    private final Linear key;
    private final Linear value;
    private final Linear outputProjection;
    private final Dropout dropout;

    public MultiHeadAttention(NDManager manager, int inputDim, int outputDim, int numberOfHeads,
                              float dropoutRate, boolean qkvBias) {
        if (outputDim % numberOfHeads != 0) {
            throw new IllegalArgumentException("outputDim must be divisible by numberOfHeads");
        }
        this.numberOfHeads = numberOfHeads;
        this.headDim = outputDim / numberOfHeads;
        query = child("query", new Linear(manager, inputDim, outputDim, qkvBias));
        key = child("key", new Linear(manager, inputDim, outputDim, qkvBias));
        value = child("value", new Linear(manager, inputDim, outputDim, qkvBias));
        outputProjection = child("outputProjection", new Linear(manager, outputDim, outputDim, true));
        dropout = child("dropout", new Dropout(dropoutRate));
    }

    /** x [batch, tokens, inputDim] -> [batch, tokens, outputDim] */
    public NDArray forward(NDArray x) {
        long batch = x.getShape().get(0);
        int tokens = (int) x.getShape().get(1);

        NDArray queries = splitHeads(query.forward(x), batch, tokens);
        NDArray keys = splitHeads(key.forward(x), batch, tokens);
        NDArray values = splitHeads(value.forward(x), batch, tokens);

        // [batch, heads, tokens, headDim] × [batch, heads, headDim, tokens] = [batch, heads, tokens, tokens]
        NDArray scores = queries.matMul(keys.transpose(0, 1, 3, 2));
        NDArray masked = scores.add(CausalAttention.causalMask(x.getManager(), tokens));
        NDArray weights = dropout.forward(masked.div(Math.sqrt(headDim)).softmax(-1));

        NDArray context = weights.matMul(values);                 // [batch, heads, tokens, headDim]
        NDArray merged = context.transpose(0, 2, 1, 3)            // [batch, tokens, heads, headDim]
                .reshape(batch, tokens, (long) numberOfHeads * headDim);
        return outputProjection.forward(merged);
    }

    /** [batch, tokens, dim] -> [batch, heads, tokens, headDim] */
    private NDArray splitHeads(NDArray x, long batch, int tokens) {
        return x.reshape(batch, tokens, numberOfHeads, headDim).transpose(0, 2, 1, 3);
    }
}
