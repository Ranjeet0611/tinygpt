package com.tinygpt.examples;

import com.tinygpt.data.Batch;
import com.tinygpt.data.GPTDataLoader;
import com.tinygpt.data.GPTDataset;
import com.tinygpt.data.SampleText;
import com.tinygpt.tokenizer.BpeTokenizer;

import java.util.Arrays;

/**
 * STEP 3 - Make "guess the next word" exercises from the story, then group them into batches.
 *
 * Run: ./run.sh 3
 * Book: Chapter 2.6
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs
 */
public class Step03_SlidingWindow {

    public static void main(String[] args) {
        BpeTokenizer tokenizer = new BpeTokenizer();
        int[] ids = tokenizer.encode(SampleText.theVerdict());

        // 1. The idea, one token at a time
        System.out.println("1) Every prefix is a little quiz: given the left side, guess the right side.");
        int[] sample = Arrays.copyOfRange(ids, 50, 55);
        for (int i = 1; i < sample.length; i++) {
            int[] context = Arrays.copyOfRange(sample, 0, i);
            System.out.printf("   %-28s ----> %s%n", "\"" + tokenizer.decode(context) + "\"", "\"" + tokenizer.decodeOne(sample[i]) + "\"");
        }
        System.out.println();

        // 2. The dataset does this for the whole story with a sliding window
        int maxLength = 4;
        int stride = 4;
        GPTDataset dataset = new GPTDataset(ids, maxLength, stride);
        System.out.println("2) Window of " + maxLength + " tokens, moving " + stride + " each time -> " + dataset.size() + " examples");
        System.out.println("   input  " + Arrays.toString(dataset.input(0)) + "  " + tokenizer.decode(dataset.input(0)).replace("\n", " "));
        System.out.println("   target " + Arrays.toString(dataset.target(0)) + "  " + tokenizer.decode(dataset.target(0)).replace("\n", " "));
        System.out.println("   (the target is the input shifted one token to the right)");
        System.out.println();

        // 3. Batches
        GPTDataLoader loader = new GPTDataLoader(dataset, 8, false, true);
        Batch first = loader.iterator().next();
        System.out.println("3) First batch of 8 examples:");
        System.out.println("   inputs  = " + Arrays.deepToString(first.inputs()));
        System.out.println("   targets = " + Arrays.deepToString(first.targets()));
    }
}
