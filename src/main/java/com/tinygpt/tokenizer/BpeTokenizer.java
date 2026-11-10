package com.tinygpt.tokenizer;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingType;
import com.knuddels.jtokkit.api.IntArrayList;

import java.util.regex.Pattern;

/**
 * The real GPT-2 tokenizer, using Byte Pair Encoding (BPE).
 *
 * Instead of whole words, BPE uses common <em>pieces</em> of words.
 * So it never needs an "unknown" token: a new word is just built from smaller pieces.
 *
 * <pre>
 *   "Akwirw ier"  ->  "Ak" "w" "ir" "w" " " "ier"  ->  [33901, 86, 343, 86, 220, 959]
 * </pre>
 *
 * GPT-2 knows 50,257 pieces. ID 50256 is the special {@code <|endoftext|>} token.
 * We use the jtokkit library, which is the Java version of OpenAI's tiktoken.
 *
 * Book: Chapter 2.5
 *
 * Picture guide: docs/classes/1-tokenizer.md#bpetokenizer · line by line: docs/walkthroughs/step02-byte-pair-encoding.md
 */
public class BpeTokenizer implements Tokenizer {

    public static final String END_OF_TEXT = "<|endoftext|>";
    public static final int END_OF_TEXT_ID = 50256;
    public static final int GPT2_VOCABULARY_SIZE = 50257;

    private final Encoding gpt2 = Encodings.newDefaultEncodingRegistry().getEncoding(EncodingType.R50K_BASE);

    @Override
    public int[] encode(String text) {
        IntArrayList result = new IntArrayList();
        // jtokkit would treat "<|endoftext|>" as normal text, so we split it out and add its ID ourselves.
        String[] documents = text.split(Pattern.quote(END_OF_TEXT), -1);
        for (int i = 0; i < documents.length; i++) {
            IntArrayList tokens = gpt2.encodeOrdinary(documents[i]);
            for (int j = 0; j < tokens.size(); j++) {
                result.add(tokens.get(j));
            }
            if (i < documents.length - 1) {
                result.add(END_OF_TEXT_ID);
            }
        }
        return result.toArray();
    }

    @Override
    public String decode(int[] tokenIds) {
        IntArrayList list = new IntArrayList(tokenIds.length);
        for (int id : tokenIds) {
            list.add(id);
        }
        return gpt2.decode(list);
    }

    /** Show how a single token ID looks as text, e.g. 995 -> " world". */
    public String decodeOne(int tokenId) {
        return decode(new int[]{tokenId});
    }

    @Override
    public int vocabularySize() {
        return GPT2_VOCABULARY_SIZE;
    }
}
