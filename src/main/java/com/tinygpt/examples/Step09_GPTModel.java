package com.tinygpt.examples;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.tinygpt.generation.TextGenerator;
import com.tinygpt.model.GPTConfig;
import com.tinygpt.model.GPTModel;
import com.tinygpt.tokenizer.BpeTokenizer;

/**
 * STEP 9 - Put everything together into a full GPT-2 model (not trained yet).
 *
 * Run: ./run.sh 9   (needs about 1 GB of memory for the 163 million numbers)
 * Book: Chapter 4.1 - 4.7
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs · line by line: docs/walkthroughs/step09-gpt-model.md
 */
public class Step09_GPTModel {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(123);
        BpeTokenizer tokenizer = new BpeTokenizer();

        try (NDManager manager = NDManager.newBaseManager()) {
            GPTConfig config = GPTConfig.GPT2_SMALL;
            GPTModel model = new GPTModel(manager, config);
            model.setTraining(false);

            System.out.println("GPT-2 small: " + config);
            System.out.printf("Total learnable numbers: %,d   (book: 163,009,536)%n", model.parameterCount());
            System.out.println("(OpenAI's GPT-2 reuses the token embedding as the output layer, which is why it is called \"124M\".)");
            System.out.println();

            int[][] batch = {
                    tokenizer.encode("Every effort moves you"),
                    tokenizer.encode("Every day holds a")
            };
            NDArray logits = model.forward(manager.create(batch));
            System.out.println("Input:  2 texts x 4 tokens");
            System.out.println("Output: " + logits.getShape() + "  -> a score for each of the 50,257 possible next tokens, at every position");
            System.out.println();

            TextGenerator generator = new TextGenerator(model, tokenizer, manager, 123);
            System.out.println("Let the untrained model write 6 tokens after \"Hello, I am\":");
            System.out.println("   " + generator.generateGreedy("Hello, I am", 6));
            System.out.println("Nonsense, because all weights are still random. Training (./run.sh train) fixes that.");
        }
    }
}
