package com.tinygpt.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

/**
 * Groups training examples into batches, so the model can learn from several at once.
 *
 * <pre>
 *   dataset:  ex0 ex1 ex2 ex3 ex4 ex5 ex6 ex7
 *   batchSize = 4
 *   batch 1:  [ex0, ex1, ex2, ex3]
 *   batch 2:  [ex4, ex5, ex6, ex7]
 * </pre>
 *
 * With {@code shuffle = true} the order is mixed up every time we loop over it,
 * so the model does not just memorise the order of the examples.
 * With {@code dropLast = true} a final, smaller batch is skipped.
 *
 * Book: Chapter 2.6
 *
 * Picture guide: docs/classes/2-data.md#gptdataloader
 */
public class GPTDataLoader implements Iterable<Batch> {

    private final GPTDataset dataset;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;
    private final Random random;

    public GPTDataLoader(GPTDataset dataset, int batchSize, boolean shuffle, boolean dropLast, long seed) {
        this.dataset = dataset;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.random = new Random(seed);
    }

    public GPTDataLoader(GPTDataset dataset, int batchSize, boolean shuffle, boolean dropLast) {
        this(dataset, batchSize, shuffle, dropLast, 123);
    }

    /** Number of batches in one pass over the data. */
    public int numberOfBatches() {
        return dropLast ? dataset.size() / batchSize : (dataset.size() + batchSize - 1) / batchSize;
    }

    @Override
    public Iterator<Batch> iterator() {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < dataset.size(); i++) {
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
                int count = Math.min(batchSize, order.size() - position);
                int[][] inputs = new int[count][];
                int[][] targets = new int[count][];
                for (int i = 0; i < count; i++) {
                    int index = order.get(position++);
                    inputs[i] = dataset.input(index);
                    targets[i] = dataset.target(index);
                }
                return new Batch(inputs, targets);
            }
        };
    }
}
