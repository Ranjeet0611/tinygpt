package com.tinygpt.examples;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.tinygpt.model.GPTConfig;
import com.tinygpt.model.GPTModel;
import com.tinygpt.tokenizer.BpeTokenizer;
import com.tinygpt.training.Loss;

/**
 * STEP 10 - Measure how wrong the model is, with one number: the loss.
 *
 * Run: ./run.sh 10
 * Book: Chapter 5.1
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs · line by line: docs/walkthroughs/step10-loss.md
 */
public class Step10_Loss {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(123);
        BpeTokenizer tokenizer = new BpeTokenizer();

        try (NDManager manager = NDManager.newBaseManager()) {
            GPTModel model = new GPTModel(manager, GPTConfig.TINY);
            model.setTraining(false);

            String[] inputs = {"every effort moves", "I really like"};
            String[] targets = {" effort moves you", " really like chocolate"};
            int[][] inputIds = {tokenizer.encode(inputs[0]), tokenizer.encode(inputs[1])};
            int[][] targetIds = {tokenizer.encode(targets[0]), tokenizer.encode(targets[1])};

            NDArray logits = model.forward(manager.create(inputIds));
            NDArray probabilities = logits.softmax(-1);

            System.out.println("How much probability does the untrained model give the correct next word?");
            for (int text = 0; text < 2; text++) {
                for (int position = 0; position < 3; position++) {
                    int correct = targetIds[text][position];
                    float probability = probabilities.getFloat(text, position, correct);
                    System.out.printf("   after %-22s correct = %-12s probability = %.7f%n",
                            "\"" + tokenizer.decode(java.util.Arrays.copyOf(inputIds[text], position + 1)) + "\"",
                            "\"" + tokenizer.decodeOne(correct) + "\"", probability);
                }
            }
            System.out.println("   A random guess out of 50,257 tokens would be 1/50257 = 0.0000199");
            System.out.println();

            float loss = Loss.crossEntropy(logits, manager.create(targetIds)).getFloat();
            System.out.printf("Cross-entropy loss: %.4f   (lower is better, perfect = 0)%n", loss);
            System.out.printf("Perplexity e^loss:  %,.0f   (\"the model is unsure between about this many tokens\")%n", Math.exp(loss));
            System.out.println("Training will push this loss down.");
        }
    }
}
