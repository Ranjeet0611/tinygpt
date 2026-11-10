package com.tinygpt.data;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

/**
 * Loads the training text: "The Verdict", a short story by Edith Wharton (1908, public domain).
 * It is the same text the book uses. About 20,000 characters, or 5,145 GPT-2 tokens.
 *
 * Picture guide: docs/classes/2-data.md#sampletext · line by line: docs/walkthroughs/step01-simple-tokenizer.md
 */
public final class SampleText {

    private SampleText() {}

    public static String theVerdict() {
        try (InputStream in = SampleText.class.getResourceAsStream("/the-verdict.txt")) {
            if (in == null) {
                throw new IllegalStateException("the-verdict.txt not found in src/main/resources");
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
