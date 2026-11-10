package com.tinygpt;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import com.tinygpt.generation.TextGenerator;
import com.tinygpt.instruct.Instruction;
import com.tinygpt.instruct.InstructionDataLoader;
import com.tinygpt.model.GPTModel;
import com.tinygpt.pretrained.Downloads;
import com.tinygpt.pretrained.GPT2Size;
import com.tinygpt.pretrained.GPT2Weights;
import com.tinygpt.tokenizer.BpeTokenizer;
import com.tinygpt.training.Trainer;

import java.nio.file.Path;
import java.util.List;

/**
 * Teach GPT-2 to follow instructions, so we can chat with it.
 *
 * <pre>
 *   1. download OpenAI's GPT-2 (small: 548 MB, medium: 1.5 GB) and the book's 1,100 instructions (200 KB)
 *   2. load GPT-2's numbers into our GPTModel                   (it can write English, but not answer)
 *   3. split the instructions: 935 to practise on, 55 to check progress, 110 to test at the end
 *   4. practise for 2 rounds on "instruction → good answer" examples
 *   5. save the result to models/gpt2-chat.params             (then: ./run.sh chat)
 * </pre>
 *
 *   ./run.sh finetune           GPT-2 small (124M), about 6 minutes on an M4 Pro
 *   ./run.sh finetune medium    GPT-2 medium (355M), the book's choice: better answers, about 3× slower
 *
 * Book: Chapter 7
 *
 * Picture guide: docs/classes/9-chat.md#finetune
 */
public class FineTune {

    public static final Path DATA_FILE = Path.of("data", "instruction-data.json");

    // The book's settings (chapter 7.6)
    private static final int EPOCHS = 2;
    private static final int BATCH_SIZE = 8;
    private static final float LEARNING_RATE = 5e-5f;
    private static final float WEIGHT_DECAY = 0.1f;

    public static void main(String[] args) throws Exception {
        Engine.getInstance().setRandomSeed(123);
        GPT2Size size = GPT2Size.fromArgs(args);
        System.out.println("Fine-tuning GPT-2 " + size + "\n");
        Downloads.ifMissing(size.url(), size.weightsFile());
        Downloads.ifMissing(Downloads.INSTRUCTION_DATA_URL, DATA_FILE);

        // 1. The data: 85% to practise on, 10% for the final test, 5% to check progress (book chapter 7.2)
        List<Instruction> all = Instruction.load(DATA_FILE);
        int trainCount = (int) (all.size() * 0.85);
        int testCount = (int) (all.size() * 0.10);
        List<Instruction> train = all.subList(0, trainCount);
        List<Instruction> test = all.subList(trainCount, trainCount + testCount);
        List<Instruction> validation = all.subList(trainCount + testCount, all.size());
        System.out.printf("Instructions: %d to practise on, %d to check progress, %d to test%n%n",
                train.size(), validation.size(), test.size());
        System.out.println("What one practice example looks like:\n");
        System.out.println(train.get(0).fullText());
        System.out.println("\n" + "-".repeat(70));

        BpeTokenizer tokenizer = new BpeTokenizer();
        try (NDManager manager = NDManager.newBaseManager()) {
            // 2. Our model, filled with OpenAI's numbers
            GPTModel model = new GPTModel(manager, size.config());
            GPT2Weights.load(model, size.weightsFile());
            TextGenerator generator = new TextGenerator(model, tokenizer, manager, 123);

            Instruction check = validation.getFirst();
            System.out.println("Before fine-tuning, asked: " + check.instruction() + " " + check.input());
            System.out.println("  it answers: " + answer(generator, check));
            System.out.println();

            // 3. Practise
            int maxLength = model.config().contextLength();
            InstructionDataLoader trainLoader = new InstructionDataLoader(train, tokenizer, BATCH_SIZE, true, true, maxLength);
            InstructionDataLoader validationLoader = new InstructionDataLoader(validation, tokenizer, BATCH_SIZE, false, false, maxLength);

            Trainer trainer = new Trainer(model, manager, LEARNING_RATE, WEIGHT_DECAY);
            long start = System.currentTimeMillis();
            trainer.train(trainLoader, validationLoader, EPOCHS, 20,
                    () -> System.out.println("   asked: " + check.instruction() + " " + check.input()
                            + "\n   answer: " + answer(generator, check)));
            System.out.printf("%nFine-tuning took %.1f minutes%n", (System.currentTimeMillis() - start) / 60000.0);

            // 4. Save, so ./run.sh chat can use it
            model.save(size.chatModelFile());
            System.out.println("Saved to " + size.chatModelFile() + ". Now try: ./run.sh chat");
            System.out.println();

            // 5. Questions it never practised on
            System.out.println("Test questions it has never seen:");
            for (Instruction example : test.subList(0, 3)) {
                System.out.println("\n> " + example.instruction() + (example.input().isBlank() ? "" : "  [" + example.input() + "]"));
                System.out.println("  correct answer: " + example.output());
                System.out.println("  model answer:   " + answer(generator, example));
            }
        }
    }

    /** Ask the model, greedily, and stop when it says <|endoftext|> ("I'm finished"). */
    public static String answer(TextGenerator generator, Instruction instruction) {
        return generator.continuation(instruction.prompt(), 120, 0f, 0, BpeTokenizer.END_OF_TEXT_ID).strip().replace("\n", " ");
    }
}
