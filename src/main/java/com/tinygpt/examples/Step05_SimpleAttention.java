package com.tinygpt.examples;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.tinygpt.attention.SimpleAttention;

/**
 * STEP 5 - Attention without any learning: each word looks at every word and mixes them.
 *
 * Run: ./run.sh 5
 * Book: Chapter 3.3
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs · line by line: docs/walkthroughs/step05-simple-attention.md
 */
public class Step05_SimpleAttention {

    static final String[] WORDS = {"Your", "journey", "starts", "with", "one", "step"};

    /** The book's tiny 3-number embeddings for "Your journey starts with one step". */
    static NDArray sentence(NDManager manager) {
        return manager.create(new float[][]{
                {0.43f, 0.15f, 0.89f}, // Your
                {0.55f, 0.87f, 0.66f}, // journey
                {0.57f, 0.85f, 0.64f}, // starts
                {0.22f, 0.58f, 0.33f}, // with
                {0.77f, 0.25f, 0.10f}, // one
                {0.05f, 0.80f, 0.55f}  // step
        });
    }

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray inputs = sentence(manager);

            // Focus on one word: "journey"
            NDArray journey = inputs.get(1);
            NDArray scores = inputs.matMul(journey);
            System.out.println("1) How much does \"journey\" match each word? (dot product)");
            print(scores.toFloatArray(), "%.4f");

            NDArray weights = scores.softmax(0);
            System.out.println("2) Softmax turns those into percentages that add up to 100%:");
            print(weights.mul(100).toFloatArray(), "%.1f%%");

            NDArray context = weights.matMul(inputs);
            System.out.println("3) New \"journey\" vector = mix of all words using those percentages:");
            System.out.println("   " + context);

            // Same thing for all six words at once, using matrix multiplication
            System.out.println("4) All six words at once:");
            System.out.println(SimpleAttention.contextVectors(inputs));
        }
    }

    private static void print(float[] values, String format) {
        for (int i = 0; i < values.length; i++) {
            System.out.printf("   %-8s " + format + "%n", WORDS[i], values[i]);
        }
        System.out.println();
    }
}
