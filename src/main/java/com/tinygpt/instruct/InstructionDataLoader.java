package com.tinygpt.instruct;

import com.tinygpt.data.Batch;
import com.tinygpt.tokenizer.BpeTokenizer;
import com.tinygpt.tokenizer.Tokenizer;
import com.tinygpt.training.Loss;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

/**
 * Packs instruction examples into batches. Unlike the story, every example has a <b>different length</b>,
 * so the short ones are padded to match the longest one in the batch.
 *
 * <pre>
 *   example A:  [ 1  2  3  4  5 ]
 *   example B:  [ 6  7  8 ]
 *
 *   1 · add one &lt;|endoftext|&gt; (E) to each:   A = 1 2 3 4 5 E      B = 6 7 8 E
 *   2 · pad with E to the same length:      A = 1 2 3 4 5 E      B = 6 7 8 E E E
 *   3 · input = all but the last, target = all but the first (shifted by one, as always)
 *
 *        input A:  1  2  3  4  5          target A:  2  3  4  5  E
 *        input B:  6  7  8  E  E          target B:  7  8  E  ✗  ✗      (✗ = -100: "don't grade this")
 * </pre>
 *
 * The first E in each target is kept: the model must learn to <b>say "I'm finished"</b> at the end of an answer.
 * The extra padding E's are replaced by -100, so the loss ignores them.
 *
 * Book: Chapter 7.3 ({@code custom_collate_fn})
 *
 * Picture guide: docs/classes/9-chat.md#instructiondataloader
 */
public class InstructionDataLoader implements Iterable<Batch> {

    private final List<int[]> examples = new ArrayList<>();
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;
    private final int maxLength;
    private final Random random = new Random(123);

    public InstructionDataLoader(List<Instruction> data, Tokenizer tokenizer, int batchSize, boolean shuffle,
                                 boolean dropLast, int maxLength) {
        for (Instruction example : data) {
            examples.add(tokenizer.encode(example.fullText()));
        }
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.maxLength = maxLength;
    }

    @Override
    public Iterator<Batch> iterator() {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < examples.size(); i++) {
            order.add(i);
        }
        if (shuffle) {
            Collections.shuffle(order, random);
        }
        return new Iterator<>() {
            private int position = 0;

            @Override
            public boolean hasNext() {
                int remaining = order.size() - position;
                return dropLast ? remaining >= batchSize : remaining > 0;
            }

            @Override
            public Batch next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                List<int[]> chosen = new ArrayList<>();
                for (int i = 0; i < batchSize && position < order.size(); i++) {
                    chosen.add(examples.get(order.get(position++)));
                }
                return collate(chosen, maxLength);
            }
        };
    }

    /** Pad a list of token lists into one rectangular batch, as in the class comment. */
    static Batch collate(List<int[]> examples, int maxLength) {
        int end = BpeTokenizer.END_OF_TEXT_ID;
        int longest = 0;
        for (int[] example : examples) {
            longest = Math.max(longest, example.length + 1);
        }
        int width = Math.min(longest - 1, maxLength);

        int[][] inputs = new int[examples.size()][width];
        int[][] targets = new int[examples.size()][width];
        for (int row = 0; row < examples.size(); row++) {
            int[] padded = new int[longest];
            Arrays.fill(padded, end);
            System.arraycopy(examples.get(row), 0, padded, 0, examples.get(row).length);

            int firstEnd = examples.get(row).length;    // where the real "I'm finished" E sits in the padded row
            for (int col = 0; col < width; col++) {
                inputs[row][col] = padded[col];
                int target = padded[col + 1];
                boolean isExtraPadding = col + 1 > firstEnd;
                targets[row][col] = isExtraPadding ? Loss.IGNORE : target;
            }
        }
        return new Batch(inputs, targets);
    }
}
