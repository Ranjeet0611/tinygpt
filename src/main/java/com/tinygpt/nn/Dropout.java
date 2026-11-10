package com.tinygpt.nn;

import ai.djl.ndarray.NDArray;

/**
 * Dropout: during training, randomly switch off some numbers (set them to 0).
 *
 * <pre>
 *   rate = 0.5
 *   in:   [ 1.0, 2.0, 3.0, 4.0 ]
 *   out:  [ 2.0, 0.0, 0.0, 8.0 ]   (survivors are scaled by 1 / (1 - rate) to keep the total similar)
 * </pre>
 *
 * Like a team practising with random players sitting out: everyone has to learn,
 * nobody can rely on one star. This stops the model memorising too much.
 * When we are not training, dropout does nothing.
 *
 * Book: Chapter 3.5.2
 *
 * Picture guide: docs/classes/3-nn.md#dropout
 */
public class Dropout extends Module {

    private final float rate;

    public Dropout(float rate) {
        this.rate = rate;
    }

    public NDArray forward(NDArray x) {
        if (!isTraining() || rate == 0f) {
            return x;
        }
        NDArray keep = x.getManager().randomUniform(0f, 1f, x.getShape()).gte(rate).toType(x.getDataType(), false);
        return x.mul(keep).div(1f - rate);
    }
}
