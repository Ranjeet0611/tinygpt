package com.tinygpt.pretrained;

import ai.djl.ndarray.NDArray;
import com.tinygpt.model.GPTModel;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Copies OpenAI's trained GPT-2 numbers into <b>our own</b> {@link GPTModel}.
 *
 * OpenAI spent a lot of computer time training GPT-2 on 8 million web pages. Our model has exactly the same
 * design, so we can borrow those numbers instead of training from scratch, like moving into a furnished house.
 * Each OpenAI name just has to be matched to the name of the same part in our model:
 *
 * <pre>
 *   OpenAI's name                    our name
 *   wte.weight                  ──►  tokenEmbedding.weight
 *   wpe.weight                  ──►  positionEmbedding.weight
 *   h.0.ln_1.weight / .bias     ──►  blocks.0.norm1.scale / .shift
 *   h.0.attn.c_attn.weight      ──►  blocks.0.attention.query / key / value .weight   (cut into 3 equal parts)
 *   h.0.attn.c_proj.weight      ──►  blocks.0.attention.outputProjection.weight
 *   h.0.ln_2.weight / .bias     ──►  blocks.0.norm2.scale / .shift
 *   h.0.mlp.c_fc.weight         ──►  blocks.0.feedForward.expand.weight
 *   h.0.mlp.c_proj.weight       ──►  blocks.0.feedForward.shrink.weight
 *   ln_f.weight / .bias         ──►  finalNorm.scale / .shift
 *   wte.weight (turned sideways) ──► outputHead.weight      (GPT-2 reuses the token table here)
 * </pre>
 *
 * Book: Chapter 5.5 ({@code load_weights_into_gpt})
 *
 * Picture guide: docs/classes/9-chat.md#gpt2weights
 */
public final class GPT2Weights {

    private GPT2Weights() {}

    public static void load(GPTModel model, Path safetensorsFile) throws IOException {
        SafeTensors file = SafeTensors.open(safetensorsFile);
        Map<String, NDArray> ours = model.namedParameters();
        Set<String> filled = new HashSet<>();
        int dim = model.config().embeddingDim();

        copy(ours, filled, "tokenEmbedding.weight", file.get("wte.weight").values());
        copy(ours, filled, "positionEmbedding.weight", file.get("wpe.weight").values());

        for (int i = 0; i < model.config().numberOfLayers(); i++) {
            String h = "h." + i + ".";
            String b = "blocks." + i + ".";

            copy(ours, filled, b + "norm1.scale", file.get(h + "ln_1.weight").values());
            copy(ours, filled, b + "norm1.shift", file.get(h + "ln_1.bias").values());

            // OpenAI keeps Query, Key and Value side by side in one [768, 2304] grid. Cut it into 3 grids of [768, 768].
            float[][] qkvWeights = splitColumns(file.get(h + "attn.c_attn.weight").values(), dim, 3 * dim, 3);
            float[][] qkvBiases = splitColumns(file.get(h + "attn.c_attn.bias").values(), 1, 3 * dim, 3);
            String[] parts = {"query", "key", "value"};
            for (int p = 0; p < 3; p++) {
                copy(ours, filled, b + "attention." + parts[p] + ".weight", qkvWeights[p]);
                copy(ours, filled, b + "attention." + parts[p] + ".bias", qkvBiases[p]);
            }
            copy(ours, filled, b + "attention.outputProjection.weight", file.get(h + "attn.c_proj.weight").values());
            copy(ours, filled, b + "attention.outputProjection.bias", file.get(h + "attn.c_proj.bias").values());

            copy(ours, filled, b + "norm2.scale", file.get(h + "ln_2.weight").values());
            copy(ours, filled, b + "norm2.shift", file.get(h + "ln_2.bias").values());

            copy(ours, filled, b + "feedForward.expand.weight", file.get(h + "mlp.c_fc.weight").values());
            copy(ours, filled, b + "feedForward.expand.bias", file.get(h + "mlp.c_fc.bias").values());
            copy(ours, filled, b + "feedForward.shrink.weight", file.get(h + "mlp.c_proj.weight").values());
            copy(ours, filled, b + "feedForward.shrink.bias", file.get(h + "mlp.c_proj.bias").values());
        }

        copy(ours, filled, "finalNorm.scale", file.get("ln_f.weight").values());
        copy(ours, filled, "finalNorm.shift", file.get("ln_f.bias").values());

        // Our output head is [768, 50257]; OpenAI's token table is [50257, 768]. Same numbers, turned sideways.
        copy(ours, filled, "outputHead.weight", transpose(file.get("wte.weight").values(), model.config().vocabularySize(), dim));

        Set<String> missing = new HashSet<>(ours.keySet());
        missing.removeAll(filled);
        if (!missing.isEmpty()) {
            throw new IllegalStateException("These parts got no GPT-2 numbers: " + missing);
        }
    }

    private static void copy(Map<String, NDArray> ours, Set<String> filled, String name, float[] values) {
        NDArray target = ours.get(name);
        if (target == null) {
            throw new IllegalStateException("Our model has no part named " + name + ". Use the config from GPT2Size.");
        }
        if (target.size() != values.length) {
            throw new IllegalStateException(name + ": expected " + target.size() + " numbers, got " + values.length);
        }
        target.set(FloatBuffer.wrap(values));
        filled.add(name);
    }

    /** Cut a [rows, columns] grid into {@code pieces} grids side by side, each [rows, columns / pieces]. */
    static float[][] splitColumns(float[] grid, int rows, int columns, int pieces) {
        int width = columns / pieces;
        float[][] result = new float[pieces][rows * width];
        for (int r = 0; r < rows; r++) {
            for (int p = 0; p < pieces; p++) {
                System.arraycopy(grid, r * columns + p * width, result[p], r * width, width);
            }
        }
        return result;
    }

    /** [rows, columns] → [columns, rows] */
    static float[] transpose(float[] grid, int rows, int columns) {
        float[] result = new float[grid.length];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                result[c * rows + r] = grid[r * columns + c];
            }
        }
        return result;
    }
}
