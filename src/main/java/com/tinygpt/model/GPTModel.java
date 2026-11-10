package com.tinygpt.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.tinygpt.nn.Dropout;
import com.tinygpt.nn.Embedding;
import com.tinygpt.nn.LayerNorm;
import com.tinygpt.nn.Linear;
import com.tinygpt.nn.Module;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * The full GPT model: token IDs in, "how likely is each next token" scores out.
 *
 * <pre>
 *   token IDs           [batch, tokens]
 *     ─► token embedding     "what is this word?"         [batch, tokens, dim]
 *     +  position embedding  "where is this word?"        [tokens, dim]
 *     ─► dropout
 *     ─► TransformerBlock × numberOfLayers
 *     ─► final LayerNorm
 *     ─► output head (Linear)                             [batch, tokens, vocabularySize]
 *   logits: one score per possible next token, at every position
 * </pre>
 *
 * The biggest score in the last position is the model's best guess for the next token.
 *
 * Book: Chapter 4.6
 *
 * Picture guide: docs/classes/5-model.md#gptmodel
 */
public class GPTModel extends Module {

    private final GPTConfig config;
    private final Embedding tokenEmbedding;
    private final Embedding positionEmbedding;
    private final Dropout dropout;
    private final List<TransformerBlock> blocks = new ArrayList<>();
    private final LayerNorm finalNorm;
    private final Linear outputHead;

    public GPTModel(NDManager manager, GPTConfig config) {
        this.config = config;
        tokenEmbedding = child(new Embedding(manager, config.vocabularySize(), config.embeddingDim()));
        positionEmbedding = child(new Embedding(manager, config.contextLength(), config.embeddingDim()));
        dropout = child(new Dropout(config.dropoutRate()));
        for (int i = 0; i < config.numberOfLayers(); i++) {
            blocks.add(child(new TransformerBlock(manager, config)));
        }
        finalNorm = child(new LayerNorm(manager, config.embeddingDim()));
        outputHead = child(new Linear(manager, config.embeddingDim(), config.vocabularySize(), false));
    }

    /** tokenIds [batch, tokens] -> logits [batch, tokens, vocabularySize] */
    public NDArray forward(NDArray tokenIds) {
        int tokens = (int) tokenIds.getShape().get(1);
        if (tokens > config.contextLength()) {
            throw new IllegalArgumentException("Input has " + tokens + " tokens but contextLength is " + config.contextLength());
        }

        NDArray positions = tokenIds.getManager().arange(tokens);
        NDArray x = tokenEmbedding.forward(tokenIds).add(positionEmbedding.forward(positions));
        x = dropout.forward(x);
        for (TransformerBlock block : blocks) {
            x = block.forward(x);
        }
        return outputHead.forward(finalNorm.forward(x));
    }

    public GPTConfig config() {
        return config;
    }

    /** Save every learned number to a file, so we do not have to train again. Book: Chapter 5.4 */
    public void save(Path file) throws IOException {
        Files.createDirectories(file.toAbsolutePath().getParent());
        try (OutputStream out = Files.newOutputStream(file)) {
            new NDList(parameters()).encode(out);
        }
    }

    /** Load numbers saved by {@link #save(Path)} into this model. The config must match. */
    public void load(Path file) throws IOException {
        List<NDArray> parameters = parameters();
        try (InputStream in = Files.newInputStream(file);
             NDManager temp = parameters.getFirst().getManager().newSubManager()) {
            NDList saved = NDList.decode(temp, in);
            if (saved.size() != parameters.size()) {
                throw new IOException("Saved model does not match this config");
            }
            for (int i = 0; i < saved.size(); i++) {
                parameters.get(i).set(saved.get(i).toByteBuffer());
            }
        }
    }
}
