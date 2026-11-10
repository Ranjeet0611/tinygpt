package com.tinygpt.nn;

import ai.djl.ndarray.NDArray;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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
 * Picture guide: docs/classes/3-nn.md#module · line by line: docs/walkthroughs/step06-self-attention.md
 */
public abstract class Module {

    private final Map<String, NDArray> ownParameters = new LinkedHashMap<>();
    private final Map<String, Module> children = new LinkedHashMap<>();
    private boolean training = true;

    /** Register a learnable array under a name, e.g. "weight". Autograd will compute a gradient for it. */
    protected NDArray parameter(String name, NDArray values) {
        values.setRequiresGradient(true);
        ownParameters.put(name, values);
        return values;
    }

    /** Register a smaller module that lives inside this one, under a name, e.g. "attention". */
    protected <M extends Module> M child(String name, M module) {
        children.put(name, module);
        return module;
    }

    /**
     * Every learnable array in this module and the modules inside it, with its full name:
     * <pre>
     *   "tokenEmbedding.weight"
     *   "blocks.0.attention.query.weight"
     *   "blocks.0.feedForward.expand.bias"
     *   ...
     * </pre>
     * Same idea as PyTorch's {@code named_parameters()}. We use the names to load GPT-2's weights.
     */
    public Map<String, NDArray> namedParameters() {
        Map<String, NDArray> all = new LinkedHashMap<>(ownParameters);
        children.forEach((childName, child) ->
                child.namedParameters().forEach((name, array) -> all.put(childName + "." + name, array)));
        return all;
    }

    /** All learnable arrays of this module and every module inside it. */
    public List<NDArray> parameters() {
        return new ArrayList<>(namedParameters().values());
    }

    /** Total count of learnable numbers. */
    public long parameterCount() {
        return parameters().stream().mapToLong(NDArray::size).sum();
    }

    /** Training mode turns dropout on. Evaluation mode (false) turns it off. */
    public void setTraining(boolean training) {
        this.training = training;
        for (Module child : children.values()) {
            child.setTraining(training);
        }
    }

    public boolean isTraining() {
        return training;
    }
}
