package com.tinygpt.examples;

import com.tinygpt.data.SampleText;
import com.tinygpt.tokenizer.BpeTokenizer;

import java.util.Arrays;

/**
 * STEP 2 - Use GPT-2's real tokenizer, which splits rare words into smaller known pieces.
 *
 * Run: ./run.sh 2
 * Book: Chapter 2.5
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs · line by line: docs/walkthroughs/step02-byte-pair-encoding.md
 */
public class Step02_BytePairEncoding {

    public static void main(String[] args) {
        BpeTokenizer tokenizer = new BpeTokenizer();
        System.out.println("GPT-2 knows " + tokenizer.vocabularySize() + " tokens.");
        System.out.println();

        String text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.";
        int[] ids = tokenizer.encode(text);
        System.out.println("Text:   " + text);
        System.out.println("IDs:    " + Arrays.toString(ids));
        System.out.println("Decode: " + tokenizer.decode(ids));
        System.out.println("No <|unk|> needed! Note <|endoftext|> became ID " + BpeTokenizer.END_OF_TEXT_ID + ".");
        System.out.println();

        // See exactly how a made-up word is cut into pieces
        String madeUp = "Akwirw ier";
        System.out.println("How \"" + madeUp + "\" is split into pieces:");
        for (int id : tokenizer.encode(madeUp)) {
            System.out.printf("   %6d -> \"%s\"%n", id, tokenizer.decodeOne(id));
        }
        System.out.println();

        int[] storyIds = tokenizer.encode(SampleText.theVerdict());
        System.out.println("The whole story becomes " + storyIds.length + " BPE tokens.");
    }
}
