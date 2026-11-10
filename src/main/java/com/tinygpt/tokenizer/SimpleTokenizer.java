package com.tinygpt.tokenizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Our first, hand-made tokenizer: one ID per word or punctuation mark.
 *
 * <pre>
 *   Step 1  split       "Hello, world."  ->  ["Hello", ",", "world", "."]
 *   Step 2  vocabulary  sort every unique word and number them 0, 1, 2, ...
 *   Step 3  encode      look each word up in the vocabulary
 * </pre>
 *
 * Words it has never seen become the special token {@code <|unk|>} ("unknown").
 * Between two separate documents we put {@code <|endoftext|>}.
 *
 * Book: Chapter 2.2 (splitting), 2.3 (token IDs), 2.4 (special tokens)
 *
 * Picture guide: docs/classes/1-tokenizer.md#simpletokenizer · line by line: docs/walkthroughs/step01-simple-tokenizer.md
 */
public class SimpleTokenizer implements Tokenizer {

    public static final String END_OF_TEXT = "<|endoftext|>";
    public static final String UNKNOWN = "<|unk|>";

    /** Split on punctuation, double dashes and whitespace, keeping the punctuation. */
    private static final Pattern SPLIT_PATTERN = Pattern.compile("([,.:;?_!\"()']|--|\\s)");

    /** Remove the space in front of punctuation when turning tokens back into text: "world ." -> "world." */
    private static final Pattern SPACE_BEFORE_PUNCTUATION = Pattern.compile("\\s+([,.:;?!\"()'])");

    private final Map<String, Integer> wordToId;
    private final Map<Integer, String> idToWord = new HashMap<>();

    public SimpleTokenizer(Map<String, Integer> vocabulary) {
        this.wordToId = Map.copyOf(vocabulary);
        vocabulary.forEach((word, id) -> idToWord.put(id, word));
    }

    /** Build a tokenizer whose vocabulary is every word that appears in {@code text}. */
    public static SimpleTokenizer fromText(String text) {
        return new SimpleTokenizer(buildVocabulary(text));
    }

    /**
     * Give every unique word an ID, in alphabetical order, then add the two special tokens.
     * <pre>
     *   ["the", "cat", "the"]  ->  {"cat": 0, "the": 1, "&lt;|endoftext|&gt;": 2, "&lt;|unk|&gt;": 3}
     * </pre>
     */
    public static Map<String, Integer> buildVocabulary(String text) {
        List<String> uniqueWords = new ArrayList<>(new TreeSet<>(split(text)));
        uniqueWords.add(END_OF_TEXT);
        uniqueWords.add(UNKNOWN);

        Map<String, Integer> vocabulary = new HashMap<>();
        for (int id = 0; id < uniqueWords.size(); id++) {
            vocabulary.put(uniqueWords.get(id), id);
        }
        return vocabulary;
    }

    /** "Hello, world." -> ["Hello", ",", "world", "."] */
    public static List<String> split(String text) {
        List<String> pieces = new ArrayList<>();
        Matcher matcher = SPLIT_PATTERN.matcher(text);
        int start = 0;
        while (matcher.find()) {
            addIfNotBlank(pieces, text.substring(start, matcher.start()));
            addIfNotBlank(pieces, matcher.group());
            start = matcher.end();
        }
        addIfNotBlank(pieces, text.substring(start));
        return pieces;
    }

    private static void addIfNotBlank(List<String> pieces, String piece) {
        String trimmed = piece.strip();
        if (!trimmed.isEmpty()) {
            pieces.add(trimmed);
        }
    }

    @Override
    public int[] encode(String text) {
        return split(text).stream()
                .mapToInt(word -> wordToId.getOrDefault(word, wordToId.get(UNKNOWN)))
                .toArray();
    }

    @Override
    public String decode(int[] tokenIds) {
        List<String> words = new ArrayList<>();
        for (int id : tokenIds) {
            words.add(idToWord.get(id));
        }
        return SPACE_BEFORE_PUNCTUATION.matcher(String.join(" ", words)).replaceAll("$1");
    }

    @Override
    public int vocabularySize() {
        return wordToId.size();
    }
}
