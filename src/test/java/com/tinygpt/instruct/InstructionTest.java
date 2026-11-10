package com.tinygpt.instruct;

import com.tinygpt.data.Batch;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class InstructionTest {

    @Test
    void collateMatchesTheBooksExample() {
        // Book chapter 7.3: inputs_1, inputs_2, inputs_3
        Batch batch = InstructionDataLoader.collate(List.of(
                new int[]{0, 1, 2, 3, 4}, new int[]{5, 6}, new int[]{7, 8, 9}), 1024);

        assertArrayEquals(new int[][]{
                {0, 1, 2, 3, 4},
                {5, 6, 50256, 50256, 50256},
                {7, 8, 9, 50256, 50256}}, batch.inputs());
        assertArrayEquals(new int[][]{
                {1, 2, 3, 4, 50256},
                {6, 50256, -100, -100, -100},
                {8, 9, 50256, -100, -100}}, batch.targets());
    }

    @Test
    void collateCutsToTheMaximumLength() {
        Batch batch = InstructionDataLoader.collate(List.of(new int[]{0, 1, 2, 3, 4}), 3);
        assertArrayEquals(new int[][]{{0, 1, 2}}, batch.inputs());
        assertArrayEquals(new int[][]{{1, 2, 3}}, batch.targets());
    }

    @Test
    void promptUsesTheAlpacaFormat() {
        Instruction withInput = new Instruction("Rewrite the sentence using a simile.", "The car is very fast.", "The car is as fast as lightning.");
        assertEquals("""
                Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                Rewrite the sentence using a simile.

                ### Input:
                The car is very fast.

                ### Response:
                The car is as fast as lightning.""", withInput.fullText());

        Instruction noInput = new Instruction("Name a color.", "", "Blue.");
        assertFalse(noInput.prompt().contains("### Input:"));
    }
}
