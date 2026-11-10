package com.tinygpt.pretrained;

import com.tinygpt.model.GPTConfig;

import java.nio.file.Path;

/**
 * The two GPT-2 sizes we can download, with the settings OpenAI used (Q/K/V with a bias, no dropout).
 *
 * <pre>
 *            numbers   layers  heads  width   download   fine-tune here
 *   SMALL      124M      12     12     768     548 MB     6 minutes on an M4 Pro
 *   MEDIUM     355M      24     16    1024     1.5 GB     roughly 3× longer (the book's choice for chapter 7)
 * </pre>
 *
 * Bigger knows more facts and follows instructions better, but is slower.
 *
 * Picture guide: docs/classes/9-chat.md#gpt2size
 */
public enum GPT2Size {

    SMALL("gpt2", "small (124M)", new GPTConfig(50257, 1024, 768, 12, 12, 0.0f, true)),
    MEDIUM("gpt2-medium", "medium (355M)", new GPTConfig(50257, 1024, 1024, 16, 24, 0.0f, true));

    private final String name;
    private final String label;
    private final GPTConfig config;

    GPT2Size(String name, String label, GPTConfig config) {
        this.name = name;
        this.label = label;
        this.config = config;
    }

    public GPTConfig config() {
        return config;
    }

    /** Where Hugging Face hosts OpenAI's weights. */
    public String url() {
        return "https://huggingface.co/openai-community/" + name + "/resolve/main/model.safetensors";
    }

    /** OpenAI's original weights, e.g. models/gpt2/model.safetensors */
    public Path weightsFile() {
        return Path.of("models", name, "model.safetensors");
    }

    /** Our fine-tuned chat version, e.g. models/gpt2-chat.params */
    public Path chatModelFile() {
        return Path.of("models", name + "-chat.params");
    }

    @Override
    public String toString() {
        return label;
    }

    /** "medium" → MEDIUM, anything else → SMALL */
    public static GPT2Size fromArgs(String[] args) {
        return args.length > 0 && args[0].equalsIgnoreCase("medium") ? MEDIUM : SMALL;
    }
}
