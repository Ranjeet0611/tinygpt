package com.tinygpt.pretrained;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class SafeTensorsTest {

    @Test
    void readsATinyHandMadeFile(@TempDir Path dir) throws Exception {
        // Two tensors: "a" = [[1, 2], [3, 4]] and "b" = [5, 6]
        String header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
                + "\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[16,24]}}";
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer file = ByteBuffer.allocate(8 + headerBytes.length + 24).order(ByteOrder.LITTLE_ENDIAN);
        file.putLong(headerBytes.length).put(headerBytes);
        for (float value : new float[]{1, 2, 3, 4, 5, 6}) {
            file.putFloat(value);
        }
        Path path = dir.resolve("tiny.safetensors");
        Files.write(path, file.array());

        SafeTensors tensors = SafeTensors.open(path);
        assertArrayEquals(new float[]{1, 2, 3, 4}, tensors.get("a").values());
        assertArrayEquals(new int[]{2, 2}, tensors.get("a").shape());
        assertArrayEquals(new float[]{5, 6}, tensors.get("b").values());
    }

    @Test
    void splitsQueryKeyValueSideBySide() {
        // 2 rows × 6 columns, cut into 3 grids of 2 × 2
        float[] grid = {1, 2, 3, 4, 5, 6,
                        7, 8, 9, 10, 11, 12};
        float[][] parts = GPT2Weights.splitColumns(grid, 2, 6, 3);
        assertArrayEquals(new float[]{1, 2, 7, 8}, parts[0]);
        assertArrayEquals(new float[]{3, 4, 9, 10}, parts[1]);
        assertArrayEquals(new float[]{5, 6, 11, 12}, parts[2]);
        assertArrayEquals(new float[]{1, 4, 2, 5, 3, 6}, GPT2Weights.transpose(new float[]{1, 2, 3, 4, 5, 6}, 2, 3));
    }
}
