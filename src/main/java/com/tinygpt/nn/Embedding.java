package com.tinygpt.nn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * A lookup table: token ID -> list of numbers (a "vector").
 *
 * <pre>
 *            dim0   dim1   dim2
 *   ID 0  [  0.3,  -1.2,   0.8 ]
 *   ID 1  [ -0.5,   0.1,   2.0 ]
 *   ID 2  [  1.1,   0.4,  -0.3 ]   &lt;- embedding(2) just returns this row
 *   ...
 * </pre>
 *
 * The rows start random. During training, words used in similar ways
 * slowly get similar rows, so "cat" ends up close to "dog".
 *
 * Same as {@code torch.nn.Embedding}.
 * Book: Chapter 2.7
 *
 * Picture guide: docs/classes/3-nn.md#embedding
 */
public class Embedding extends Module {

    private final NDArray weight;

    public Embedding(NDManager manager, int numberOfIds, int embeddingDim) {
        weight = parameter(manager.randomNormal(0f, 1f, new Shape(numberOfIds, embeddingDim), DataType.FLOAT32));
    }

    /** ids of any shape [...] -> vectors of shape [..., embeddingDim] */
    public NDArray forward(NDArray ids) {
        return weight.get(ids);
    }

    /** The whole table, so examples can print it. */
    public NDArray weight() {
        return weight;
    }
}
