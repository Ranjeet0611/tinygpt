package com.tinygpt.data;

import com.tinygpt.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Cuts one long list of token IDs into many small training examples using a "sliding window".
 *
 * A GPT learns one simple game: <b>guess the next token</b>.
 * So for each window, the answer (target) is the same window moved one step to the right.
 *
 * <pre>
 *   tokens:   [ I ][ HAD ][ ALWAYS ][ THOUGHT ][ JACK ][ ... ]
 *
 *   input:    [ I ][ HAD ][ ALWAYS ][ THOUGHT ]
 *   target:        [ HAD ][ ALWAYS ][ THOUGHT ][ JACK ]
 *
 *   "I"                       -> should guess "HAD"
 *   "I HAD"                   -> should guess "ALWAYS"
 *   "I HAD ALWAYS"            -> should guess "THOUGHT"
 *   "I HAD ALWAYS THOUGHT"    -> should guess "JACK"
 * </pre>
 *
 * {@code maxLength} is the window size. {@code stride} is how far the window jumps each time.
 * If stride == maxLength, windows touch but do not overlap.
 *
 * Book: Chapter 2.6
 *
 * Picture guide: docs/classes/2-data.md#gptdataset · line by line: docs/walkthroughs/step03-sliding-window.md
 */
public class GPTDataset {

    private final List<int[]> inputs = new ArrayList<>();
    private final List<int[]> targets = new ArrayList<>();

    public GPTDataset(String text, Tokenizer tokenizer, int maxLength, int stride) {
        this(tokenizer.encode(text), maxLength, stride);
    }

    public GPTDataset(int[] tokenIds, int maxLength, int stride) {
        for (int start = 0; start + maxLength < tokenIds.length; start += stride) {
            inputs.add(Arrays.copyOfRange(tokenIds, start, start + maxLength));
            targets.add(Arrays.copyOfRange(tokenIds, start + 1, start + maxLength + 1));
        }
    }

    public int[] input(int index) {
        return inputs.get(index);
    }

    public int[] target(int index) {
        return targets.get(index);
    }

    public int size() {
        return inputs.size();
    }
}
