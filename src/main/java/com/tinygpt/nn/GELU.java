package com.tinygpt.nn;

import ai.djl.ndarray.NDArray;

/**
 * GELU activation: a smooth "mostly let positives through, mostly block negatives" curve.
 *
 * <pre>
 *        out
 *         |          /
 *         |        /
 *         |      /
 *   ------+--.-'------- in
 *       ''  (small dip below 0, then flat)
 * </pre>
 *
 * Without a bend like this, stacking layers would just be one big multiplication,
 * and the model could only learn straight-line patterns.
 *
 * Formula (GPT-2's approximation):
 * {@code 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))}
 *
 * Book: Chapter 4.3
 *
 * Picture guide: docs/classes/3-nn.md#gelu
 */
public final class GELU {

    private static final double SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);

    private GELU() {}

    public static NDArray apply(NDArray x) {
        NDArray inner = x.add(x.pow(3).mul(0.044715f)).mul(SQRT_2_OVER_PI);
        return x.mul(0.5f).mul(inner.tanh().add(1f));
    }
}
