package com.tinygpt.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * A small two-layer network that each token goes through on its own.
 *
 * <pre>
 *   [embeddingDim]  --Linear-->  [4 × embeddingDim]  --GELU-->  --Linear-->  [embeddingDim]
 *       768                           3072                                      768
 *     (expand: room to "think")                                     (shrink back)
 * </pre>
 *
 * Attention lets tokens talk to each other. The feed forward part lets each token
 * think about what it just heard.
 *
 * Book: Chapter 4.3
 *
 * Picture guide: docs/classes/3-nn.md#feedforward
 */
public class FeedForward extends Module {

    private final Linear expand;
    private final Linear shrink;

    public FeedForward(NDManager manager, int embeddingDim) {
        expand = child(new Linear(manager, embeddingDim, 4 * embeddingDim, true));
        shrink = child(new Linear(manager, 4 * embeddingDim, embeddingDim, true));
    }

    public NDArray forward(NDArray x) {
        return shrink.forward(GELU.apply(expand.forward(x)));
    }
}
