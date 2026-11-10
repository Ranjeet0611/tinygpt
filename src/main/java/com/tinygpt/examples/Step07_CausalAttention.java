package com.tinygpt.examples;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import com.tinygpt.attention.CausalAttention;
import com.tinygpt.attention.SimpleAttention;
import com.tinygpt.nn.Dropout;

/**
 * STEP 7 - Causal attention: no peeking at future words. Plus dropout.
 *
 * Run: ./run.sh 7
 * Book: Chapter 3.5
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs
 */
public class Step07_CausalAttention {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(123);
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray inputs = Step05_SimpleAttention.sentence(manager);

            NDArray mask = CausalAttention.causalMask(manager, 6);
            System.out.println("1) The mask: 0 = allowed, -Infinity = future (hidden)");
            System.out.println(mask);

            NDArray weights = SimpleAttention.scores(inputs).add(mask).softmax(-1);
            System.out.println("2) Attention weights after masking. Everything above the diagonal is 0,");
            System.out.println("   and every row still adds up to 1:");
            System.out.println(weights);

            Dropout dropout = new Dropout(0.5f);
            System.out.println("3) Dropout(0.5) on a grid of ones: about half become 0, the rest become 2");
            System.out.println(dropout.forward(manager.ones(new Shape(6, 6))));

            // A batch of 2 copies of the sentence: [2, 6, 3]
            NDArray batch = NDArrays.stack(new NDList(inputs, inputs));
            CausalAttention attention = new CausalAttention(manager, 3, 2, 0.0f);
            NDArray context = attention.forward(batch);
            System.out.println("4) CausalAttention on a batch " + batch.getShape() + " -> " + context.getShape());
        }
    }
}
