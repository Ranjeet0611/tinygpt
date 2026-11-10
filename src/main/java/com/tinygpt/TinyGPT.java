package com.tinygpt;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import com.tinygpt.data.GPTDataLoader;
import com.tinygpt.data.GPTDataset;
import com.tinygpt.data.SampleText;
import com.tinygpt.generation.TextGenerator;
import com.tinygpt.model.GPTConfig;
import com.tinygpt.model.GPTModel;
import com.tinygpt.tokenizer.BpeTokenizer;
import com.tinygpt.training.Trainer;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * The whole journey in one program: read a story, train a tiny GPT on it, and make it write.
 *
 * <pre>
 *   ./run.sh train                          train, save, then write some text
 *   ./run.sh generate "Every effort moves"   load the saved model and write
 * </pre>
 *
 * Book: Chapter 5 brings together everything from chapters 2, 3 and 4.
 *
 * Picture guide: docs/classes/8-main-and-examples.md#tinygpt
 */
public class TinyGPT {

    private static final Path MODEL_FILE = Path.of("models", "tinygpt.params");
    private static final String PROMPT = "My name is";

    // Training settings
    private static final int EPOCHS = 25;
    private static final int BATCH_SIZE = 8;
    private static final float LEARNING_RATE = 1e-3f;
    private static final float WEIGHT_DECAY = 0.1f;
    private static final float TRAIN_SHARE = 0.9f;

    public static void main(String[] args) throws Exception {
        Engine.getInstance().setRandomSeed(123); // same seed -> same results every run

        GPTConfig config = GPTConfig.TINY;
        BpeTokenizer tokenizer = new BpeTokenizer();

        try (NDManager manager = NDManager.newBaseManager()) {
            GPTModel model = new GPTModel(manager, config);
            TextGenerator generator = new TextGenerator(model, tokenizer, manager, 123);

            if (args.length > 0 && args[0].equals("generate")) {
                String prompt = args.length > 1 ? args[1] : PROMPT;
                if (!Files.exists(MODEL_FILE)) {
                    System.out.println("No saved model yet. Run without arguments first to train one.");
                    return;
                }
                model.load(MODEL_FILE);
                showSamples(generator, prompt);
                return;
            }

            System.out.printf("Model: %d layers, %d heads, %d dims, context %d tokens -> %,d parameters%n%n",
                    config.numberOfLayers(), config.numberOfHeads(), config.embeddingDim(),
                    config.contextLength(), model.parameterCount());

            // 1. Read the story and split it: 90% to learn from, 10% to test on
            String text = SampleText.theVerdict();
            int split = (int) (text.length() * TRAIN_SHARE);
            int[] trainTokens = tokenizer.encode(text.substring(0, split));
            int[] validationTokens = tokenizer.encode(text.substring(split));
            System.out.printf("Training tokens: %d, validation tokens: %d%n", trainTokens.length, validationTokens.length);

            // 2. Cut the tokens into windows and group them into batches
            int window = config.contextLength();
            GPTDataLoader trainLoader = new GPTDataLoader(new GPTDataset(trainTokens, window, window), BATCH_SIZE, true, true);
            GPTDataLoader validationLoader = new GPTDataLoader(new GPTDataset(validationTokens, window, window), BATCH_SIZE, false, false);

            // 3. Before training the model knows nothing and writes nonsense
            System.out.println("Before training: " + generator.generateGreedy(PROMPT, 10));
            System.out.println();

            // 4. Train
            Trainer trainer = new Trainer(model, manager, LEARNING_RATE, WEIGHT_DECAY);
            long start = System.currentTimeMillis();
            trainer.train(trainLoader, validationLoader, EPOCHS, 10, generator, PROMPT);
            System.out.printf("%nTraining took %.1f seconds%n", (System.currentTimeMillis() - start) / 1000.0);

            // 5. Save, so `generate` can reuse it without training again
            model.save(MODEL_FILE);
            System.out.println("Saved model to " + MODEL_FILE);
            System.out.println();

            showSamples(generator, PROMPT);
        }
    }

    private static void showSamples(TextGenerator generator, String prompt) {
        System.out.println("Greedy (always the top choice):");
        System.out.println("  " + generator.generateGreedy(prompt, 40).replace("\n", " "));
        System.out.println("Temperature 1.0, top-k 25 (a bit creative):");
        System.out.println("  " + generator.generate(prompt, 40, 1.0f, 25).replace("\n", " "));
    }
}
