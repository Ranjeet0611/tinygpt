package com.tinygpt.examples;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import com.tinygpt.generation.TextGenerator;
import com.tinygpt.model.GPTModel;
import com.tinygpt.pretrained.Downloads;
import com.tinygpt.pretrained.GPT2Size;
import com.tinygpt.pretrained.GPT2Weights;
import com.tinygpt.tokenizer.BpeTokenizer;

/**
 * STEP 12 - Load OpenAI's trained GPT-2 numbers into OUR model, and let it write.
 *
 * Our GPTModel has exactly GPT-2's design, so OpenAI's numbers fit straight in.
 * Training GPT-2 ourselves would take weeks; downloading its numbers takes a minute.
 *
 * Run: ./run.sh 12   (downloads 548 MB the first time)
 * Book: Chapter 5.5
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs
 */
public class Step12_LoadGPT2 {

    public static void main(String[] args) throws Exception {
        Engine.getInstance().setRandomSeed(123);
        GPT2Size size = GPT2Size.SMALL;
        Downloads.ifMissing(size.url(), size.weightsFile());

        try (NDManager manager = NDManager.newBaseManager()) {
            GPTModel model = new GPTModel(manager, size.config());
            GPT2Weights.load(model, size.weightsFile());
            System.out.printf("Loaded OpenAI's GPT-2 small into our GPTModel: %,d numbers%n%n", model.parameterCount());

            TextGenerator generator = new TextGenerator(model, new BpeTokenizer(), manager, 123);

            // A well-known test: Hugging Face's guide shows GPT-2's greedy answer for this prompt.
            // If our model does exactly the same maths as GPT-2, we get exactly the same words.
            System.out.println("Greedy:");
            System.out.println("  " + generator.generateGreedy("I enjoy walking with my cute dog", 30).replace("\n", " "));
            System.out.println();

            // The book's example (chapter 5.5): temperature 1.5, top-k 50
            System.out.println("Temperature 1.5, top-k 50:");
            System.out.println("  " + generator.generate("Every effort moves you", 25, 1.5f, 50).replace("\n", " "));
            System.out.println();
            System.out.println("It writes real English, but it only continues text. It doesn't answer questions yet.");
            System.out.println("Chapter 7 teaches it to follow instructions: ./run.sh finetune");
        }
    }
}
