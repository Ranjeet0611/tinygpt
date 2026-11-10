package com.tinygpt.examples;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.tinygpt.attention.SelfAttention;

/**
 * STEP 6 - Self-attention with learnable Query, Key and Value weights.
 *
 * We use the exact weights printed in the book, so you get the same numbers as the book.
 *
 * Run: ./run.sh 6
 * Book: Chapter 3.4
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs
 */
public class Step06_SelfAttention {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(123);
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray inputs = Step05_SimpleAttention.sentence(manager);   // 6 words x 3 numbers

            // Book weights (torch.manual_seed(123)), each 3 inputs -> 2 outputs
            NDArray wQuery = manager.create(new float[][]{
                    {0.29611194f, 0.51656228f}, {0.25167072f, 0.68855679f}, {0.07397246f, 0.86652195f}});
            NDArray wKey = manager.create(new float[][]{
                    {0.13657987f, 0.10247904f}, {0.18405646f, 0.72644675f}, {0.31525391f, 0.68710667f}});
            NDArray wValue = manager.create(new float[][]{
                    {0.07563531f, 0.19663817f}, {0.31641197f, 0.40174013f}, {0.11856830f, 0.82739538f}});

            NDArray queries = inputs.matMul(wQuery);
            NDArray keys = inputs.matMul(wKey);
            NDArray values = inputs.matMul(wValue);
            System.out.println("1) Every word gets a query, a key and a value (2 numbers each):");
            System.out.println("   queries " + queries.getShape() + ", keys " + keys.getShape() + ", values " + values.getShape());
            System.out.println();

            NDArray scores = queries.matMul(keys.transpose());
            System.out.println("2) Scores = queries x keys (who should listen to whom):");
            System.out.println(scores);

            NDArray context = SelfAttention.attend(queries, keys, values);
            System.out.println("3) Context vectors (book: [[0.2996, 0.8053], [0.3061, 0.8210], ...]):");
            System.out.println(context);

            // The reusable module does the same with random starting weights
            SelfAttention module = new SelfAttention(manager, 3, 2);
            System.out.println("4) SelfAttention module with random weights -> output shape " + module.forward(inputs).getShape());
        }
    }
}
