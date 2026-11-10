package com.tinygpt.dialog;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.tinygpt.tokenizer.Tokenizer;

import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipFile;

/**
 * One everyday conversation between two people, A and B, from the DailyDialog dataset.
 *
 * <pre>
 *   turns: ["Say, Jim, how about going for a few beers after dinner?",
 *           "You know that is tempting but is really not good for our fitness.",
 *           "What do you mean? It will help us to relax.", ...]
 * </pre>
 *
 * The model practises on it written out like a play script, one line per turn:
 *
 * <pre>
 *   A: Say, Jim, how about going for a few beers after dinner?
 *   B: You know that is tempting but is really not good for our fitness.
 *   A: What do you mean? It will help us to relax.
 * </pre>
 *
 * When we chat, <b>you are A and the model is B</b>: we write the conversation so far, end with "B:",
 * and let the model finish the line. The newline at the end of its line means "my turn is over".
 *
 * Picture guide: docs/classes/9-chat.md#dialogue · line by line: docs/walkthroughs/finetune-talk-and-talk.md
 */
public record Dialogue(List<String> turns) {

    /** GPT-2's token for "\n": the model writes it when its line is finished. */
    public static final int NEW_LINE_ID = 198;

    /** A, B, A, B, ... */
    public static String speaker(int turn) {
        return turn % 2 == 0 ? "A" : "B";
    }

    /** The whole conversation, one "A: ..." / "B: ..." line per turn. */
    public String text() {
        return script(turns);
    }

    /** The first turns that fit in {@code maxTokens} tokens (whole turns only, at least one). */
    public Dialogue firstTokens(Tokenizer tokenizer, int maxTokens) {
        int keep = 1;
        while (keep < turns.size() && tokenizer.encode(script(turns.subList(0, keep + 1))).length <= maxTokens) {
            keep++;
        }
        return new Dialogue(turns.subList(0, keep));
    }

    /** What the model reads before it speaks: the conversation so far, then "B:" (or "A:") for its own line. */
    public static String prompt(List<String> history) {
        return script(history) + speaker(history.size()) + ":";
    }

    private static String script(List<String> turns) {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < turns.size(); i++) {
            text.append(speaker(i)).append(": ").append(turns.get(i)).append('\n');
        }
        return text.toString();
    }

    /**
     * Read one part ("train", "validation" or "test") of ConvLab's DailyDialog zip,
     * which holds data/dialogues.json: a list of {"data_split": ..., "turns": [{"utterance": ...}, ...]}.
     */
    public static List<Dialogue> load(Path zip, String split) throws IOException {
        List<Dialogue> dialogues = new ArrayList<>();
        try (ZipFile file = new ZipFile(zip.toFile());
             var reader = new InputStreamReader(file.getInputStream(file.getEntry("data/dialogues.json")), StandardCharsets.UTF_8)) {
            for (JsonElement element : JsonParser.parseReader(reader).getAsJsonArray()) {
                JsonObject dialogue = element.getAsJsonObject();
                if (!dialogue.get("data_split").getAsString().equals(split)) {
                    continue;
                }
                List<String> turns = new ArrayList<>();
                for (JsonElement turn : dialogue.getAsJsonArray("turns")) {
                    turns.add(turn.getAsJsonObject().get("utterance").getAsString().replace("\n", " ").strip());
                }
                dialogues.add(new Dialogue(turns));
            }
        }
        return dialogues;
    }
}
