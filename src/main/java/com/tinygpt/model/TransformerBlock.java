package com.tinygpt.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.tinygpt.attention.MultiHeadAttention;
import com.tinygpt.nn.Dropout;
import com.tinygpt.nn.FeedForward;
import com.tinygpt.nn.LayerNorm;
import com.tinygpt.nn.Module;

/**
 * One transformer block: the main building block of GPT. GPT stacks many of them.
 *
 * <pre>
 *        x ─────────────────────────────┐   (shortcut)
 *        │                              │
 *        ▼                              │
 *   LayerNorm ─► MultiHeadAttention ─► Dropout ─► (+) ──► x
 *                                                         │
 *        ┌────────────────────────────────────────────────┤   (shortcut)
 *        ▼                                                │
 *   LayerNorm ─► FeedForward ─► Dropout ───────────────► (+) ──► output
 * </pre>
 *
 * Step 1: tokens talk to each other (attention).
 * Step 2: each token thinks on its own (feed forward).
 *
 * The <b>shortcut</b> (also called residual connection) adds the input back to the output.
 * It is like keeping your original notes and only writing corrections in the margin,
 * so nothing important gets lost even after many blocks.
 *
 * Book: Chapter 4.4 (shortcuts) and 4.5 (transformer block)
 *
 * Picture guide: docs/classes/5-model.md#transformerblock
 */
public class TransformerBlock extends Module {

    private final LayerNorm norm1;
    private final MultiHeadAttention attention;
    private final LayerNorm norm2;
    private final FeedForward feedForward;
    private final Dropout dropShortcut;

    public TransformerBlock(NDManager manager, GPTConfig config) {
        norm1 = child(new LayerNorm(manager, config.embeddingDim()));
        attention = child(new MultiHeadAttention(manager, config.embeddingDim(), config.embeddingDim(),
                config.numberOfHeads(), config.dropoutRate(), config.qkvBias()));
        norm2 = child(new LayerNorm(manager, config.embeddingDim()));
        feedForward = child(new FeedForward(manager, config.embeddingDim()));
        dropShortcut = child(new Dropout(config.dropoutRate()));
    }

    /** x [batch, tokens, embeddingDim] -> same shape */
    public NDArray forward(NDArray x) {
        x = x.add(dropShortcut.forward(attention.forward(norm1.forward(x))));
        x = x.add(dropShortcut.forward(feedForward.forward(norm2.forward(x))));
        return x;
    }
}
