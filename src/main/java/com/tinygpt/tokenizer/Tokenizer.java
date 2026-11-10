package com.tinygpt.tokenizer;

/**
 * A tokenizer turns text into numbers and numbers back into text.
 *
 * <pre>
 *   "Hello world"  --encode-->  [15496, 995]  --decode-->  "Hello world"
 * </pre>
 *
 * Computers cannot do maths on words, but they can on numbers.
 * Every word (or piece of a word) gets its own ID, like a seat number in a cinema.
 *
 * Book: Chapter 2.2 - 2.5
 *
 * Picture guide: docs/classes/1-tokenizer.md#tokenizer · line by line: docs/walkthroughs/step01-simple-tokenizer.md
 */
public interface Tokenizer {

    /** Text to token IDs. */
    int[] encode(String text);

    /** Token IDs back to text. */
    String decode(int[] tokenIds);

    /** How many different tokens this tokenizer knows. */
    int vocabularySize();
}
