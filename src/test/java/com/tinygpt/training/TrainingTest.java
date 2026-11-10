package com.tinygpt.training;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import com.tinygpt.data.GPTDataLoader;
import com.tinygpt.data.GPTDataset;
import com.tinygpt.generation.TextGenerator;
import com.tinygpt.model.GPTConfig;
import com.tinygpt.model.GPTModel;
import com.tinygpt.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrainingTest {

    @Test
    void lossOfAPerfectGuessIsAlmostZero() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray logits = manager.create(new float[]{0f, 100f, 0f}, new Shape(1, 1, 3));   // very sure it is token 1
            NDArray right = manager.create(new int[][]{{1}});
            NDArray wrong = manager.create(new int[][]{{2}});
            assertEquals(0f, Loss.crossEntropy(logits, right).getFloat(), 1e-4);
            assertEquals(100f, Loss.crossEntropy(logits, wrong).getFloat(), 1e-3);
        }
    }

    @Test
    void positionsMarkedIgnoreAreNotGraded() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // position 0 is a perfect guess, position 1 is a terrible guess but marked -100
            NDArray logits = manager.create(new float[]{0f, 100f, 0f, 100f, 0f, 0f}, new Shape(1, 2, 3));
            NDArray targets = manager.create(new int[][]{{1, Loss.IGNORE}});
            assertEquals(0f, Loss.crossEntropy(logits, targets).getFloat(), 1e-4);
        }
    }

    @Test
    void lossOfARandomModelIsAboutLnVocabulary() {
        Engine.getInstance().setRandomSeed(123);
        try (NDManager manager = NDManager.newBaseManager()) {
            GPTModel model = new GPTModel(manager, new GPTConfig(1000, 8, 16, 2, 1, 0f, false));
            NDArray logits = model.forward(manager.create(new int[][]{{1, 2, 3, 4}}));
            float loss = Loss.crossEntropy(logits, manager.create(new int[][]{{2, 3, 4, 5}})).getFloat();
            assertEquals(Math.log(1000), loss, 1.0);
        }
    }

    @Test
    void trainingMakesTheModelLearnACountingPattern() {
        Engine.getInstance().setRandomSeed(123);
        // Training text: 0 1 2 3 4 5 6 7 8 9 0 1 2 ... the model should learn "after n comes n+1"
        int[] tokens = new int[400];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = i % 10;
        }
        GPTConfig config = new GPTConfig(10, 8, 16, 2, 1, 0f, false);

        try (NDManager manager = NDManager.newBaseManager()) {
            GPTModel model = new GPTModel(manager, config);
            GPTDataLoader loader = new GPTDataLoader(new GPTDataset(tokens, 8, 4), 8, true, true);
            Trainer trainer = new Trainer(model, manager, 1e-2f, 0f);

            float before = trainer.averageLoss(loader, 5);
            TextGenerator generator = new TextGenerator(model, new DigitTokenizer(), manager, 1);
            trainer.train(loader, loader, 10, 1000, generator, "0");
            float after = trainer.averageLoss(loader, 5);

            assertTrue(after < before / 4, "loss should drop a lot: before " + before + ", after " + after);
            assertEquals("3456789012", generator.generateGreedy("3", 9));
        }
    }

    /** Each digit 0-9 is one token. */
    private static class DigitTokenizer implements Tokenizer {
        @Override
        public int[] encode(String text) {
            return text.chars().map(c -> c - '0').toArray();
        }

        @Override
        public String decode(int[] tokenIds) {
            StringBuilder text = new StringBuilder();
            for (int id : tokenIds) {
                text.append(id);
            }
            return text.toString();
        }

        @Override
        public int vocabularySize() {
            return 10;
        }
    }
}
