package com.tinygpt.generation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import com.tinygpt.model.GPTModel;
import com.tinygpt.tokenizer.Tokenizer;

import java.util.Arrays;
import java.util.Random;

/**
 * Makes the model write text, one token at a time.
 *
 * <pre>
 *   "Every effort moves"          ─► model ─► best next token: " you"
 *   "Every effort moves you"      ─► model ─► best next token: " forward"
 *   "Every effort moves you forward" ...      (repeat maxNewTokens times)
 * </pre>
 *
 * How we choose the next token from the model's scores (logits):
 * <ul>
 *   <li><b>Greedy</b> (temperature = 0): always take the highest score. Safe but repetitive.</li>
 *   <li><b>Temperature</b>: below 1 = more careful, above 1 = more creative/random.</li>
 *   <li><b>Top-k</b>: only pick among the k best tokens, so we never choose something silly.</li>
 * </ul>
 *
 * This part is written in plain Java (no tensors) so every step is easy to follow.
 *
 * Book: Chapter 4.7 (greedy) and 5.3 (temperature, top-k)
 *
 * Picture guide: docs/classes/7-generation.md#textgenerator
 */
public class TextGenerator {

    /** Pass as stopTokenId to never stop early. */
    public static final int NO_STOP_TOKEN = -1;

    private final GPTModel model;
    private final Tokenizer tokenizer;
    private final NDManager manager;
    private final Random random;

    public TextGenerator(GPTModel model, Tokenizer tokenizer, NDManager manager, long seed) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.manager = manager;
        this.random = new Random(seed);
    }

    /** Greedy: always pick the most likely token. */
    public String generateGreedy(String prompt, int maxNewTokens) {
        return generate(prompt, maxNewTokens, 0f, 0);
    }

    /**
     * @param temperature 0 = greedy; 0.5 = careful; 1.0 = normal; 1.5 = wild
     * @param topK        only consider the k most likely tokens (0 = consider all)
     * @return the prompt followed by the new text
     */
    public String generate(String prompt, int maxNewTokens, float temperature, int topK) {
        return prompt + continuation(prompt, maxNewTokens, temperature, topK, NO_STOP_TOKEN);
    }

    /**
     * Only the <b>new</b> text, stopping early if the model writes {@code stopTokenId}.
     * The chatbot uses this with stopTokenId = 50256 ({@code <|endoftext|>}): "I've finished my answer".
     * Book: Chapter 5.3.3 ({@code eos_id})
     */
    public String continuation(String prompt, int maxNewTokens, float temperature, int topK, int stopTokenId) {
        boolean wasTraining = model.isTraining();
        model.setTraining(false);

        int[] ids = tokenizer.encode(prompt);
        int promptLength = ids.length;
        for (int step = 0; step < maxNewTokens; step++) {
            float[] scores = nextTokenScores(ids);
            int next = temperature <= 0f ? argMax(scores) : sample(scores, temperature, topK);
            if (next == stopTokenId) {
                break;
            }
            ids = Arrays.copyOf(ids, ids.length + 1);
            ids[ids.length - 1] = next;
        }

        model.setTraining(wasTraining);
        return tokenizer.decode(Arrays.copyOfRange(ids, promptLength, ids.length));
    }

    /** Run the model and return the scores for the token that comes after the last one. */
    private float[] nextTokenScores(int[] ids) {
        // The model can only read contextLength tokens, so keep only the most recent ones
        int contextLength = model.config().contextLength();
        int[] window = ids.length > contextLength ? Arrays.copyOfRange(ids, ids.length - contextLength, ids.length) : ids;

        try (NDScope ignored = new NDScope()) {
            NDArray input = manager.create(new int[][]{window});        // [1, tokens]
            NDArray logits = model.forward(input);                       // [1, tokens, vocab]
            return logits.get(0, window.length - 1).toFloatArray();      // last position only
        }
    }

    public static int argMax(float[] scores) {
        int best = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > scores[best]) {
                best = i;
            }
        }
        return best;
    }

    private int sample(float[] scores, float temperature, int topK) {
        float[] probabilities = softmax(keepTopK(scores, topK), temperature);
        double r = random.nextDouble();
        double cumulative = 0;
        for (int i = 0; i < probabilities.length; i++) {
            cumulative += probabilities[i];
            if (r < cumulative) {
                return i;
            }
        }
        return argMax(probabilities);
    }

    /** Set every score outside the top k to -infinity, so it gets probability 0. */
    public static float[] keepTopK(float[] scores, int topK) {
        if (topK <= 0 || topK >= scores.length) {
            return scores;
        }
        float[] sorted = scores.clone();
        Arrays.sort(sorted);
        float threshold = sorted[sorted.length - topK];
        float[] result = scores.clone();
        for (int i = 0; i < result.length; i++) {
            if (result[i] < threshold) {
                result[i] = Float.NEGATIVE_INFINITY;
            }
        }
        return result;
    }

    /** Turn scores into probabilities that add up to 1. Temperature divides the scores first. */
    public static float[] softmax(float[] scores, float temperature) {
        float max = Float.NEGATIVE_INFINITY;
        for (float s : scores) {
            max = Math.max(max, s);
        }
        double sum = 0;
        double[] exp = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            exp[i] = Math.exp((scores[i] - max) / temperature);  // subtract max so exp() never overflows
            sum += exp[i];
        }
        float[] probabilities = new float[scores.length];
        for (int i = 0; i < scores.length; i++) {
            probabilities[i] = (float) (exp[i] / sum);
        }
        return probabilities;
    }
}
