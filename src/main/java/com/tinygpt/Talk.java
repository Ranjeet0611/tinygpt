package com.tinygpt;

import ai.djl.ndarray.NDManager;
import com.tinygpt.generation.TextGenerator;
import com.tinygpt.model.GPTModel;
import com.tinygpt.pretrained.GPT2Size;
import com.tinygpt.tokenizer.BpeTokenizer;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

/**
 * Have a conversation with GPT-2 fine-tuned on DailyDialog. Unlike {@link Chat}, it <b>remembers</b> what was said.
 *
 * <pre>
 *   You: Hi! How are you?                  history = ["Hi! How are you?"]
 *        │
 *        ▼  the model reads the conversation as a script and finishes B's line:
 *   "A: Hi! How are you?\nB:"
 *        │
 *        ▼  TextGenerator writes until the end of the line ("\n")
 *   Bot: I'm fine, thanks. And you?        history = ["Hi! How are you?", "I'm fine, thanks. And you?"]
 * </pre>
 *
 * Only the last 10 turns are kept, so the prompt stays short and fast.
 *
 *   ./run.sh talk          uses GPT-2 medium if you fine-tuned it, otherwise small
 *   ./run.sh talk small    choose the size yourself (run ./run.sh finetune-talk [medium] once first)
 *
 * Picture guide: docs/classes/9-chat.md#talk · line by line: docs/walkthroughs/finetune-talk-and-talk.md
 */
public class Talk {

    private static final int REMEMBERED_TURNS = 10;

    public static void main(String[] args) throws Exception {
        GPT2Size size = args.length > 0 ? GPT2Size.fromArgs(args)
                : Files.exists(GPT2Size.MEDIUM.talkModelFile()) ? GPT2Size.MEDIUM : GPT2Size.SMALL;
        if (!Files.exists(size.talkModelFile())) {
            System.out.println("No conversation-trained GPT-2 " + size + " yet. Run this once first:  ./run.sh finetune-talk"
                    + (size == GPT2Size.MEDIUM ? " medium" : ""));
            return;
        }

        System.out.println("Loading " + size.talkModelFile() + " ...");
        try (NDManager manager = NDManager.newBaseManager()) {
            GPTModel model = new GPTModel(manager, size.config());
            model.load(size.talkModelFile());
            TextGenerator generator = new TextGenerator(model, new BpeTokenizer(), manager, System.nanoTime());

            System.out.println("TinyGPT talk (GPT-2 " + size + ", fine-tuned on DailyDialog conversations)");
            System.out.println("""
                    Chat like you would with a friend: "Hi! What are you doing this weekend?"
                    Type 'new' to start a new conversation, 'exit' to quit.
                    """);

            List<String> history = new ArrayList<>();
            BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
            while (true) {
                System.out.print("You: ");
                System.out.flush();
                String line = in.readLine();
                if (line == null || line.equalsIgnoreCase("exit") || line.equalsIgnoreCase("quit")) {
                    break;
                }
                if (line.equalsIgnoreCase("new")) {
                    history.clear();
                    System.out.println("(new conversation)\n");
                    continue;
                }
                if (line.isBlank()) {
                    continue;
                }
                history.add(line.strip());
                // Keep an even number of turns, so you are always A and the bot is always B
                while (history.size() > REMEMBERED_TURNS) {
                    history.subList(0, 2).clear();
                }
                String reply = FineTuneTalk.reply(generator, history).strip();
                history.add(reply);
                System.out.println("Bot: " + reply);
                System.out.println();
            }
        }
    }
}
