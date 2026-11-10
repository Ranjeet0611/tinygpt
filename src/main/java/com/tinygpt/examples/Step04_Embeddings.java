package com.tinygpt.examples;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.tinygpt.data.GPTDataLoader;
import com.tinygpt.data.GPTDataset;
import com.tinygpt.data.SampleText;
import com.tinygpt.nn.Embedding;
import com.tinygpt.tokenizer.BpeTokenizer;

/**
 * STEP 4 - Turn token IDs into vectors (lists of numbers), and add "where am I in the sentence".
 *
 * Run: ./run.sh 4
 * Book: Chapter 2.7 (token embeddings) and 2.8 (position embeddings)
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs · line by line: docs/walkthroughs/step04-embeddings.md
 */
public class Step04_Embeddings {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(123);
        try (NDManager manager = NDManager.newBaseManager()) {

            // 1. A tiny lookup table: 6 possible IDs, 3 numbers each
            Embedding small = new Embedding(manager, 6, 3);
            System.out.println("1) Embedding table (6 IDs x 3 numbers). It starts random; training will improve it.");
            System.out.println(small.weight());

            NDArray ids = manager.create(new int[]{2, 3, 5, 1});
            System.out.println("Looking up IDs [2, 3, 5, 1] just picks those rows:");
            System.out.println(small.forward(ids));
            System.out.println();

            // 2. Real sizes: GPT-2 vocabulary, 256 numbers per token
            int vocabularySize = 50257;
            int outputDim = 256;
            int contextLength = 4;
            int[] storyIds = new BpeTokenizer().encode(SampleText.theVerdict());
            GPTDataLoader loader = new GPTDataLoader(new GPTDataset(storyIds, contextLength, contextLength), 8, false, true);
            NDArray batch = manager.create(loader.iterator().next().inputs());
            System.out.println("2) A batch of token IDs:              " + batch.getShape() + "   (8 examples x 4 tokens)");

            Embedding tokenEmbedding = new Embedding(manager, vocabularySize, outputDim);
            NDArray tokenVectors = tokenEmbedding.forward(batch);
            System.out.println("   Token embeddings  (what):          " + tokenVectors.getShape() + "   (each token -> 256 numbers)");

            // 3. Position embeddings: one vector per position 0, 1, 2, 3
            Embedding positionEmbedding = new Embedding(manager, contextLength, outputDim);
            NDArray positionVectors = positionEmbedding.forward(manager.arange(contextLength));
            System.out.println("   Position embeddings (where):       " + positionVectors.getShape() + "      (one per position)");

            NDArray inputEmbeddings = tokenVectors.add(positionVectors);
            System.out.println("   Token + position = model input:    " + inputEmbeddings.getShape());
            System.out.println();
            System.out.println("Without positions, \"dog bites man\" and \"man bites dog\" would look the same to the model.");
        }
    }
}
