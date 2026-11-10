package com.tinygpt.examples;

import com.tinygpt.data.SampleText;
import com.tinygpt.tokenizer.SimpleTokenizer;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * STEP 1 - Turn words into numbers with our own tokenizer.
 *
 * Run: ./run.sh 1
 * Book: Chapter 2.2, 2.3, 2.4
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs
 */
public class Step01_SimpleTokenizer {

    public static void main(String[] args) {
        String story = SampleText.theVerdict();
        System.out.println("The story has " + story.length() + " characters. It starts like this:");
        System.out.println("  " + story.substring(0, 99) + "...");
        System.out.println();

        // 1. Split text into words and punctuation
        List<String> words = SimpleTokenizer.split(story);
        System.out.println("1) Split into " + words.size() + " pieces. The first 20:");
        System.out.println("   " + words.subList(0, 20));
        System.out.println();

        // 2. Give every unique piece a number
        Map<String, Integer> vocabulary = SimpleTokenizer.buildVocabulary(story);
        System.out.println("2) Vocabulary size: " + vocabulary.size() + " unique tokens (including 2 special tokens)");
        vocabulary.entrySet().stream()
                .sorted(Map.Entry.comparingByValue())
                .limit(8)
                .forEach(entry -> System.out.println("   " + entry.getValue() + " -> " + entry.getKey()));
        System.out.println("   ...");
        System.out.println();

        // 3. Encode and decode
        SimpleTokenizer tokenizer = new SimpleTokenizer(vocabulary);
        String sentence = "\"It's the last he painted, you know,\" Mrs. Gisburn said with pardonable pride.";
        int[] ids = tokenizer.encode(sentence);
        System.out.println("3) Encode: " + sentence);
        System.out.println("   IDs:    " + Arrays.toString(ids));
        System.out.println("   Decode: " + tokenizer.decode(ids));
        System.out.println();

        // 4. Words the tokenizer has never seen become <|unk|>
        String withNewWords = "Hello, do you like tea?" + " " + SimpleTokenizer.END_OF_TEXT + " " + "In the sunlit terraces of the palace.";
        int[] newIds = tokenizer.encode(withNewWords);
        System.out.println("4) Unknown words: " + withNewWords);
        System.out.println("   IDs:    " + Arrays.toString(newIds));
        System.out.println("   Decode: " + tokenizer.decode(newIds));
        System.out.println("   'Hello', 'tea' and 'palace' were never in the story, so they became <|unk|>.");
        System.out.println("   That loses information. Step 2 fixes this with Byte Pair Encoding.");
    }
}
