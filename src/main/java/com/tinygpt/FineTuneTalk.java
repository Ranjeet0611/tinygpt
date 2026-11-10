package com.tinygpt;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import com.tinygpt.dialog.Dialogue;
import com.tinygpt.generation.TextGenerator;
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
 * Teach GPT-2 to hold an everyday conversation, using the DailyDialog dataset.
 *
 * <pre>
 *   1. download OpenAI's GPT-2 and DailyDialog (11,118 practice conversations, 1,000 to check progress)
 *   2. load GPT-2's numbers into our GPTModel                (it can write English, but not take turns)
 *   3. write each conversation as a script ("A: ...\nB: ...\n") and keep the first 256 tokens of it
 *   4. practise for 1 round on all of them
 *   5. save the result to models/gpt2-talk.params          (then: ./run.sh talk)
 * </pre>
 *
 *   ./run.sh finetune-talk           GPT-2 small (124M), about 70 minutes on an M4 Pro
 *   ./run.sh finetune-talk medium    GPT-2 medium (355M), better replies, roughly 3× slower
 *
 * This is the same recipe as {@link FineTune} (book chapter 7): only the practice data is different.
 *
 * Picture guide: docs/classes/9-chat.md#finetunetalk · line by line: docs/walkthroughs/finetune-talk-and-talk.md
 */
public class FineTuneTalk {

    public static final Path DATA_FILE = Path.of("data", "dailydialog.zip");

    private static final int EPOCHS = 1;
    private static final int BATCH_SIZE = 8;
    private static final int MAX_TOKENS = 256;      // longer conversations are cut (at a whole turn) to save time and memory
    private static final float LEARNING_RATE = 5e-5f;
    private static final float WEIGHT_DECAY = 0.1f;

    public static void main(String[] args) throws Exception {
        Engine.getInstance().setRandomSeed(123);
        GPT2Size size = GPT2Size.fromArgs(args);
        System.out.println("Teaching GPT-2 " + size + " to chat, with DailyDialog\n");
        Downloads.ifMissing(size.url(), size.weightsFile());
        Downloads.ifMissing(Downloads.DAILY_DIALOG_URL, DATA_FILE);

        BpeTokenizer tokenizer = new BpeTokenizer();
        List<String> train = texts(Dialogue.load(DATA_FILE, "train"), tokenizer);
        List<String> validation = texts(Dialogue.load(DATA_FILE, "validation"), tokenizer);
        System.out.printf("Conversations: %,d to practise on, %,d to check progress%n%n", train.size(), validation.size());
        System.out.println("What one practice conversation looks like:\n");
        System.out.println(train.getFirst());
        System.out.println("-".repeat(70));

        List<String> check = List.of("Hi! How are you today?");

        try (NDManager manager = NDManager.newBaseManager()) {
            GPTModel model = new GPTModel(manager, size.config());
            GPT2Weights.load(model, size.weightsFile());
            TextGenerator generator = new TextGenerator(model, tokenizer, manager, 123);
            System.out.println("Before practice, to \"" + check.getFirst() + "\" it replies: " + reply(generator, check));
            System.out.println();

            int maxLength = model.config().contextLength();
            InstructionDataLoader trainLoader = new InstructionDataLoader(train, tokenizer, BATCH_SIZE, true, true, maxLength);
            InstructionDataLoader validationLoader = new InstructionDataLoader(validation, tokenizer, BATCH_SIZE, false, false, maxLength);

            Trainer trainer = new Trainer(model, manager, LEARNING_RATE, WEIGHT_DECAY);
            long start = System.currentTimeMillis();
            trainer.train(trainLoader, validationLoader, EPOCHS, 100,
                    () -> System.out.println("   A: " + check.getFirst() + "\n   B:" + reply(generator, check)));
            System.out.printf("%nPractice took %.1f minutes%n", (System.currentTimeMillis() - start) / 60000.0);

            model.save(size.talkModelFile());
            System.out.println("Saved to " + size.talkModelFile() + ". Now try: ./run.sh talk");
        }
    }

    /** Each conversation as a script, cut to MAX_TOKENS. */
    private static List<String> texts(List<Dialogue> dialogues, BpeTokenizer tokenizer) {
        return dialogues.stream().map(d -> d.firstTokens(tokenizer, MAX_TOKENS).text()).toList();
    }

    /**
     * The model's next line, given the conversation so far.
     * A little randomness (temperature 0.7, top 40) keeps it from repeating the same safe reply every time.
     */
    public static String reply(TextGenerator generator, List<String> history) {
        return generator.continuation(Dialogue.prompt(history), 60, 0.7f, 40, Dialogue.NEW_LINE_ID);
    }
}
