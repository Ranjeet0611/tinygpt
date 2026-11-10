package com.tinygpt.tokenizer;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TokenizerTest {

    @Test
    void splitKeepsPunctuationAsSeparateTokens() {
        assertEquals(List.of("Hello", ",", "world", ".", "Is", "this", "--", "a", "test", "?"),
                SimpleTokenizer.split("Hello, world. Is this-- a test?"));
    }

    @Test
    void simpleTokenizerRoundTrip() {
        SimpleTokenizer tokenizer = SimpleTokenizer.fromText("the cat sat on the mat.");
        String text = "the cat sat on the mat.";
        assertEquals(text, tokenizer.decode(tokenizer.encode(text)));
    }

    @Test
    void unknownWordsBecomeUnk() {
        SimpleTokenizer tokenizer = SimpleTokenizer.fromText("the cat sat");
        assertEquals("the <|unk|> sat", tokenizer.decode(tokenizer.encode("the dog sat")));
    }

    @Test
    void bpeMatchesGpt2() {
        BpeTokenizer tokenizer = new BpeTokenizer();
        assertArrayEquals(new int[]{15496, 11, 995}, tokenizer.encode("Hello, world"));
        assertArrayEquals(new int[]{15496, 50256, 995}, tokenizer.encode("Hello<|endoftext|> world"));
        assertEquals("Akwirw ier", tokenizer.decode(tokenizer.encode("Akwirw ier")));
    }
}
