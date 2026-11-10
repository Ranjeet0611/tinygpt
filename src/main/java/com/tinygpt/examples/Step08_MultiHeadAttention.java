package com.tinygpt.examples;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.tinygpt.attention.MultiHeadAttention;

/**
 * STEP 8 - Multi-head attention: several attention "heads" look for different things at the same time.
 *
 * Run: ./run.sh 8
 * Book: Chapter 3.6
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs · line by line: docs/walkthroughs/step08-multi-head-attention.md
 */
public class Step08_MultiHeadAttention {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(123);
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray inputs = Step05_SimpleAttention.sentence(manager);
            NDArray batch = NDArrays.stack(new NDList(inputs, inputs));      // [2 sentences, 6 words, 3 numbers]

            MultiHeadAttention attention = new MultiHeadAttention(manager, 3, 2, 2, 0.0f, false);
            attention.setTraining(false);
            NDArray context = attention.forward(batch);

            System.out.println("Input  " + batch.getShape() + "  (2 sentences, 6 words, 3 numbers each)");
            System.out.println("2 heads, each 1 number wide, glued back together:");
            System.out.println("Output " + context.getShape());
            System.out.println(context);
            System.out.println("Learnable numbers in this module: " + attention.parameterCount());
            System.out.println();

            // GPT-2 size: 12 heads of 64 numbers = 768
            MultiHeadAttention gpt2Size = new MultiHeadAttention(manager, 768, 768, 12, 0.1f, false);
            System.out.println("A GPT-2-sized attention (768 dims, 12 heads) has " + String.format("%,d", gpt2Size.parameterCount()) + " learnable numbers.");
        }
    }
}
