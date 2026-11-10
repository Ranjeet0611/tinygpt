package com.tinygpt.nn;

import ai.djl.ndarray.NDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * The parent of every layer, just like {@code torch.nn.Module} in the book.
 *
 * A module does two jobs:
 * <ol>
 *   <li>It keeps its <b>parameters</b>: the numbers the model learns during training.</li>
 *   <li>It keeps its <b>children</b>: smaller modules inside it (a model is modules inside modules).</li>
 * </ol>
 *
 * <pre>
 *   GPTModel
 *   ├── Embedding (tokens)        &lt;- parameters
 *   ├── Embedding (positions)     &lt;- parameters
 *   ├── TransformerBlock x N
 *   │   ├── LayerNorm, MultiHeadAttention, LayerNorm, FeedForward
 *   ├── LayerNorm
 *   └── Linear (output head)      &lt;- parameters
 * </pre>
 *
 * Calling {@link #parameters()} on the top module collects every number in the whole tree,
 * so the optimizer can update all of them.
 *
 * Picture guide: docs/classes/3-nn.md#module
 */
public abstract class Module {

    private final List<NDArray> ownParameters = new ArrayList<>();
    private final List<Module> children = new ArrayList<>();
    private boolean training = true;

    /** Register a learnable array. Autograd will compute a gradient for it. */
    protected NDArray parameter(NDArray values) {
        values.setRequiresGradient(true);
        ownParameters.add(values);
        return values;
    }

    /** Register a smaller module that lives inside this one. */
    protected <M extends Module> M child(M module) {
        children.add(module);
        return module;
    }

    /** All learnable arrays of this module and every module inside it. */
    public List<NDArray> parameters() {
        List<NDArray> all = new ArrayList<>(ownParameters);
        for (Module child : children) {
            all.addAll(child.parameters());
        }
        return all;
    }

    /** Total count of learnable numbers. */
    public long parameterCount() {
        return parameters().stream().mapToLong(NDArray::size).sum();
    }

    /** Training mode turns dropout on. Evaluation mode (false) turns it off. */
    public void setTraining(boolean training) {
        this.training = training;
        for (Module child : children) {
            child.setTraining(training);
        }
    }

    public boolean isTraining() {
        return training;
    }
}
