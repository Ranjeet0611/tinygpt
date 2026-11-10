package com.tinygpt.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * A fully connected layer: {@code output = input × weight + bias}.
 *
 * <pre>
 *   input  [.., inFeatures]
 *      × weight [inFeatures, outFeatures]
 *      + bias   [outFeatures]
 *   output [.., outFeatures]
 * </pre>
 *
 * Think of it as a machine with knobs (the weights). Each output number is a
 * weighted mix of all the input numbers. Training turns the knobs.
 *
 * Same as {@code torch.nn.Linear}.
 *
 * Picture guide: docs/classes/3-nn.md#linear
 */
public class Linear extends Module {

    private final NDArray weight;
    private final NDArray bias;

    public Linear(NDManager manager, int inFeatures, int outFeatures, boolean useBias) {
        // Random starting values between -1/√in and +1/√in, like PyTorch does by default
        float limit = (float) (1.0 / Math.sqrt(inFeatures));
        weight = parameter("weight", manager.randomUniform(-limit, limit, new Shape(inFeatures, outFeatures), DataType.FLOAT32));
        bias = useBias ? parameter("bias", manager.zeros(new Shape(outFeatures))) : null;
    }

    public NDArray forward(NDArray input) {
        NDArray output = input.matMul(weight);
        return bias == null ? output : output.add(bias);
    }
}
