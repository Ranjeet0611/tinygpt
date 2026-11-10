package com.tinygpt.attention;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.tinygpt.nn.Linear;
import com.tinygpt.nn.Module;

/**
 * Self-attention with learnable weights: Query, Key and Value.
 *
 * Think of a library:
 * <pre>
 *   Query  "what am I looking for?"        (the question a word asks)
 *   Key    "what do I contain?"            (the label on each book)
 *   Value  "what do I actually hand over?" (the content of the book)
 * </pre>
 *
 * <pre>
 *   inputs ──×Wq──► queries ─┐
 *          ──×Wk──► keys ────┴─► scores = Q × Kᵀ ─► ÷ √dim ─► softmax ─► weights ─┐
 *          ──×Wv──► values ───────────────────────────────────────────────────────┴─► weights × V = context
 * </pre>
 *
 * Dividing by √dim keeps scores from getting huge, so softmax does not become all-or-nothing.
 * Wq, Wk and Wv are learned, so the model decides for itself what "similar" should mean.
 *
 * Book: Chapter 3.4
 *
 * Picture guide: docs/classes/4-attention.md#selfattention
 */
public class SelfAttention extends Module {

    private final Linear query;
    private final Linear key;
    private final Linear value;

    public SelfAttention(NDManager manager, int inputDim, int outputDim) {
        query = child(new Linear(manager, inputDim, outputDim, false));
        key = child(new Linear(manager, inputDim, outputDim, false));
        value = child(new Linear(manager, inputDim, outputDim, false));
    }

    /** inputs [tokens, inputDim] -> context vectors [tokens, outputDim] */
    public NDArray forward(NDArray inputs) {
        return attend(query.forward(inputs), key.forward(inputs), value.forward(inputs));
    }

    /** The attention recipe once queries, keys and values are known. Kept public so examples can use fixed weights. */
    public static NDArray attend(NDArray queries, NDArray keys, NDArray values) {
        NDArray scores = queries.matMul(keys.transpose());
        long keyDim = keys.getShape().getLastDimension();
        NDArray weights = scores.div(Math.sqrt(keyDim)).softmax(-1);
        return weights.matMul(values);
    }
}
