package com.tinygpt;

import ai.djl.ndarray.NDManager;
import com.tinygpt.generation.TextGenerator;
import com.tinygpt.instruct.Instruction;
import com.tinygpt.model.GPTModel;
import com.tinygpt.pretrained.GPT2Size;
import com.tinygpt.tokenizer.BpeTokenizer;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;

/**
 * Chat with the fine-tuned GPT-2 in the terminal.
 *
 * <pre>
 *   You: What is the opposite of hot?
 *        │
 *        ▼  wrapped in the same format it practised on:
 *   "Below is an instruction ... ### Instruction:\nWhat is the opposite of hot?\n\n### Response:\n"
 *        │
 *        ▼  TextGenerator writes until it says <|endoftext|>
 *   Bot: The opposite of hot is cold.
 * </pre>
 *
 * Each message is answered on its own: the model doesn't remember earlier messages.
 * To give extra input, put it after " || ", e.g.  Rewrite this as a question || The sky is blue.
 *
 *   ./run.sh chat          uses GPT-2 medium if you fine-tuned it, otherwise small
 *   ./run.sh chat small    choose the size yourself (run ./run.sh finetune [medium] once first)
 *
 * Book: Chapter 7.7 (extracting responses)
 *
 * Picture guide: docs/classes/9-chat.md#chat
 */
public class Chat {

    public static void main(String[] args) throws Exception {
        // Use the size asked for; otherwise the biggest one that has been fine-tuned
        GPT2Size size = args.length > 0 ? GPT2Size.fromArgs(args)
                : Files.exists(GPT2Size.MEDIUM.chatModelFile()) ? GPT2Size.MEDIUM : GPT2Size.SMALL;
        if (!Files.exists(size.chatModelFile())) {
            System.out.println("No fine-tuned GPT-2 " + size + " yet. Run this once first:  ./run.sh finetune"
                    + (size == GPT2Size.MEDIUM ? " medium" : ""));
            return;
        }

        try (NDManager manager = NDManager.newBaseManager()) {
            GPTModel model = new GPTModel(manager, size.config());
            model.load(size.chatModelFile());
            TextGenerator generator = new TextGenerator(model, new BpeTokenizer(), manager, 123);

            System.out.println("TinyGPT chat (GPT-2 " + size + ", fine-tuned on 935 instructions)");
            System.out.println("""
                    Ask simple things: "What is the capital of France?", "Give me a synonym for happy".
                    Extra input goes after ||  e.g.  Translate to French || Good morning
                    Type 'exit' to quit.
                    """);

            BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
            while (true) {
                System.out.print("You: ");
                System.out.flush();
                String line = in.readLine();
                if (line == null || line.equalsIgnoreCase("exit") || line.equalsIgnoreCase("quit")) {
                    break;
                }
                if (line.isBlank()) {
                    continue;
                }
                String[] parts = line.split("\\|\\|", 2);
                Instruction request = new Instruction(parts[0].strip(), parts.length > 1 ? parts[1].strip() : "", "");
                System.out.println("Bot: " + FineTune.answer(generator, request));
                System.out.println();
            }
        }
    }
}
