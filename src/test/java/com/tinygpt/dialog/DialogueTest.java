package com.tinygpt.dialog;

import com.tinygpt.tokenizer.BpeTokenizer;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class DialogueTest {

    private final Dialogue dialogue = new Dialogue(List.of("Hi!", "Hello. How are you?", "Fine, thanks."));

    @Test
    void writesTheConversationAsAScript() {
        assertEquals("A: Hi!\nB: Hello. How are you?\nA: Fine, thanks.\n", dialogue.text());
    }

    @Test
    void promptEndsWithTheNextSpeaker() {
        assertEquals("A: Hi!\nB:", Dialogue.prompt(List.of("Hi!")));
    }

    @Test
    void endOfALineIsTheNewLineToken() {
        BpeTokenizer tokenizer = new BpeTokenizer();
        assertArrayEquals(new int[]{Dialogue.NEW_LINE_ID}, tokenizer.encode("\n"));
    }

    @Test
    void cutsLongConversationsAtAWholeTurn() {
        BpeTokenizer tokenizer = new BpeTokenizer();
        int twoTurns = tokenizer.encode("A: Hi!\nB: Hello. How are you?\n").length;
        assertEquals(2, dialogue.firstTokens(tokenizer, twoTurns).turns().size());
        assertEquals(1, dialogue.firstTokens(tokenizer, 1).turns().size());
    }
}
