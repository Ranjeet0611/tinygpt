package com.tinygpt.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Layer normalization: rescales each vector so its numbers have average 0 and spread 1.
 *
 * <pre>
 *   before:  [ 0.2, 9.0, -4.0, 30.0 ]   (wild sizes)
 *   after:   [-0.66, 0.02, -0.98, 1.62]   (mean 0, variance 1)
 * </pre>
 *
 * Like adjusting the volume so no instrument is too loud or too quiet.
 * This keeps training stable. {@code scale} and {@code shift} are learnable, so the
 * model can still pick a different volume if that helps.
 *
 * Book: Chapter 4.2
 *
 * Picture guide: docs/classes/3-nn.md#layernorm · line by line: docs/walkthroughs/step09-gpt-model.md
 */
public class LayerNorm extends Module {

    private static final float EPSILON = 1e-5f; // avoids dividing by zero

    private final NDArray scale;
    private final NDArray shift;

    public LayerNorm(NDManager manager, int embeddingDim) {
        scale = parameter("scale", manager.ones(new Shape(embeddingDim)));
        shift = parameter("shift", manager.zeros(new Shape(embeddingDim)));
    }

    public NDArray forward(NDArray x) {
        int[] lastAxis = {-1};
        NDArray mean = x.mean(lastAxis, true);
        NDArray variance = x.sub(mean).square().mean(lastAxis, true);
        NDArray normalized = x.sub(mean).div(variance.add(EPSILON).sqrt());
        return normalized.mul(scale).add(shift);
    }
}
