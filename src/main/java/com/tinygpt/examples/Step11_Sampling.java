package com.tinygpt.examples;

import com.tinygpt.generation.TextGenerator;

import java.util.Random;

/**
 * STEP 11 - Choosing the next word: greedy, temperature and top-k. No model needed!
 *
 * Imagine the model just read "every effort moves you" and gives these scores for 9 words.
 *
 * Run: ./run.sh 11
 * Book: Chapter 5.3
 *
 * Picture guide: docs/classes/8-main-and-examples.md#the-step-programs
 */
public class Step11_Sampling {

    private static final String[] WORDS = {"closer", "every", "effort", "forward", "inches", "moves", "pizza", "toward", "you"};
    private static final float[] SCORES = {4.51f, 0.89f, -1.90f, 6.75f, 1.63f, -1.62f, -1.89f, 6.28f, 1.79f};

    public static void main(String[] args) {
        System.out.println("Greedy (always pick the highest score): \"" + WORDS[TextGenerator.argMax(SCORES)] + "\"");
        System.out.println();

        for (float temperature : new float[]{1f, 0.1f, 5f}) {
            System.out.println("Temperature " + temperature + " -> pick 1,000 times:");
            printCounts(sampleThousandTimes(TextGenerator.softmax(SCORES, temperature)));
        }
        System.out.println("Low temperature = sure and boring. High temperature = creative, but even \"pizza\" can win.");
        System.out.println();

        System.out.println("Top-k = 3, temperature 1 -> only the 3 best words can ever be picked:");
        printCounts(sampleThousandTimes(TextGenerator.softmax(TextGenerator.keepTopK(SCORES, 3), 1f)));
    }

    private static int[] sampleThousandTimes(float[] probabilities) {
        Random random = new Random(123);
        int[] counts = new int[probabilities.length];
        for (int draw = 0; draw < 1000; draw++) {
            double r = random.nextDouble();
            double cumulative = 0;
            for (int i = 0; i < probabilities.length; i++) {
                cumulative += probabilities[i];
                if (r < cumulative || i == probabilities.length - 1) {
                    counts[i]++;
                    break;
                }
            }
        }
        return counts;
    }

    private static void printCounts(int[] counts) {
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] > 0) {
                System.out.printf("   %-8s %4d  %s%n", WORDS[i], counts[i], "#".repeat(Math.max(1, counts[i] / 20)));
            }
        }
        System.out.println();
    }
}
