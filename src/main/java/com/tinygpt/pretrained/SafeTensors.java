package com.tinygpt.pretrained;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Reads a {@code .safetensors} file: the format Hugging Face uses to share model weights.
 *
 * The file is simple:
 * <pre>
 *   ┌──────────────┬──────────────────────────────────────────┬──────────────────────────┐
 *   │ 8 bytes      │ JSON header (a "table of contents")      │ all the numbers, packed  │
 *   │ header size  │ "wte.weight": {"dtype": "F32",           │ one after another        │
 *   │              │    "shape": [50257, 768],                │                          │
 *   │              │    "data_offsets": [287209472, ...]}, ...│                          │
 *   └──────────────┴──────────────────────────────────────────┴──────────────────────────┘
 * </pre>
 *
 * To read a tensor, look up its name in the table of contents, then read the bytes between its two offsets.
 *
 * Picture guide: docs/classes/9-chat.md#safetensors
 */
public final class SafeTensors {

    /** One named grid of numbers in the file. */
    public record Tensor(String name, int[] shape, float[] values) {}

    private record Entry(int[] shape, long start, long end) {}

    private final ByteBuffer data;
    private final Map<String, Entry> entries = new LinkedHashMap<>();

    private SafeTensors(ByteBuffer data, JsonObject header) {
        this.data = data;
        for (String name : header.keySet()) {
            if (name.equals("__metadata__")) {
                continue;
            }
            JsonObject info = header.getAsJsonObject(name);
            if (!info.get("dtype").getAsString().equals("F32")) {
                throw new IllegalArgumentException(name + ": only 32-bit float (F32) tensors are supported");
            }
            int[] shape = info.getAsJsonArray("shape").asList().stream().mapToInt(e -> e.getAsInt()).toArray();
            var offsets = info.getAsJsonArray("data_offsets");
            entries.put(name, new Entry(shape, offsets.get(0).getAsLong(), offsets.get(1).getAsLong()));
        }
    }

    public static SafeTensors open(Path file) throws IOException {
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            ByteBuffer sizeBytes = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(sizeBytes, 0);
            long headerSize = sizeBytes.getLong(0);

            ByteBuffer headerBytes = ByteBuffer.allocate((int) headerSize);
            channel.read(headerBytes, 8);
            JsonObject header = JsonParser.parseString(new String(headerBytes.array(), StandardCharsets.UTF_8)).getAsJsonObject();

            // Map the numbers straight from disk, so we don't copy 548 MB into memory up front
            long dataStart = 8 + headerSize;
            ByteBuffer data = channel.map(FileChannel.MapMode.READ_ONLY, dataStart, channel.size() - dataStart)
                    .order(ByteOrder.LITTLE_ENDIAN);
            return new SafeTensors(data, header);
        }
    }

    public Set<String> names() {
        return entries.keySet();
    }

    public Tensor get(String name) {
        Entry entry = entries.get(name);
        if (entry == null) {
            throw new IllegalArgumentException("No tensor named " + name);
        }
        float[] values = new float[(int) ((entry.end - entry.start) / Float.BYTES)];
        data.slice((int) entry.start, (int) (entry.end - entry.start)).order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer().get(values);
        return new Tensor(name, entry.shape, values);
    }
}
