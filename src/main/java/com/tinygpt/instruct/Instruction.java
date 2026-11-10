package com.tinygpt.instruct;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * One example of "a request, and a good answer to it". The book's dataset has 1,100 of these.
 *
 * <pre>
 *   instruction: "Rewrite the sentence using a simile."
 *   input:       "The car is very fast."                  (often empty)
 *   output:      "The car is as fast as lightning."
 * </pre>
 *
 * The model sees it written out in the "Alpaca" style (book chapter 7.2), so it can tell where the request
 * ends and where its own answer should begin:
 *
 * <pre>
 *   Below is an instruction that describes a task. Write a response that appropriately completes the request.
 *
 *   ### Instruction:
 *   Rewrite the sentence using a simile.
 *
 *   ### Input:
 *   The car is very fast.
 *
 *   ### Response:
 *   The car is as fast as lightning.
 * </pre>
 *
 * Book: Chapter 7.2 ({@code format_input})
 *
 * Picture guide: docs/classes/9-chat.md#instruction
 */
public record Instruction(String instruction, String input, String output) {

    private static final String INTRO =
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.";

    /** Everything up to (and including) "### Response:", i.e. what the model reads before it answers. */
    public String prompt() {
        String text = INTRO + "\n\n### Instruction:\n" + instruction;
        if (input != null && !input.isBlank()) {
            text += "\n\n### Input:\n" + input;
        }
        return text + "\n\n### Response:\n";
    }

    /** The prompt plus the correct answer: what the model practises on. */
    public String fullText() {
        return prompt() + output;
    }

    public static List<Instruction> load(Path jsonFile) throws IOException {
        return new Gson().fromJson(Files.readString(jsonFile), new TypeToken<List<Instruction>>() {}.getType());
    }
}
