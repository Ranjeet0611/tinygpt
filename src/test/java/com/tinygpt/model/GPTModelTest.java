package com.tinygpt.model;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.tinygpt.nn.LayerNorm;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GPTModelTest {

    static final GPTConfig SMALL = new GPTConfig(100, 8, 16, 2, 2, 0f, false);

    private NDManager manager;

    @BeforeEach
    void setUp() {
        Engine.getInstance().setRandomSeed(123);
        manager = NDManager.newBaseManager();
    }

    @AfterEach
    void tearDown() {
        manager.close();
    }

    @Test
    void logitsHaveOneScorePerVocabularyToken() {
        GPTModel model = new GPTModel(manager, SMALL);
        NDArray logits = model.forward(manager.create(new int[][]{{1, 2, 3}, {4, 5, 6}}));
        assertEquals(new Shape(2, 3, 100), logits.getShape());
    }

    @Test
    void gpt2SmallParameterCountMatchesBook() {
        // Count without building the 163M-parameter model: formula from chapter 4.6
        GPTConfig c = GPTConfig.GPT2_SMALL;
        long d = c.embeddingDim();
        long perBlock = 2 * 2 * d + 3 * d * d + (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d);
        long total = c.vocabularySize() * d + c.contextLength() * d + c.numberOfLayers() * perBlock + 2 * d + d * c.vocabularySize();
        assertEquals(163_009_536L, total);

        GPTModel small = new GPTModel(manager, SMALL);
        long ds = SMALL.embeddingDim();
        long smallPerBlock = 4 * ds + 3 * ds * ds + (ds * ds + ds) + (4 * ds * ds + 4 * ds) + (4 * ds * ds + ds);
        assertEquals(100 * ds + 8 * ds + 2 * smallPerBlock + 2 * ds + ds * 100, small.parameterCount());
    }

    @Test
    void parametersHaveReadableNames() {
        var names = new GPTModel(manager, SMALL).namedParameters().keySet();
        assertTrue(names.contains("tokenEmbedding.weight"));
        assertTrue(names.contains("blocks.1.attention.query.weight"));
        assertTrue(names.contains("blocks.0.feedForward.expand.bias"));
        assertTrue(names.contains("outputHead.weight"));
    }

    @Test
    void layerNormGivesMeanZeroVarianceOne() {
        NDArray x = manager.randomNormal(5f, 3f, new Shape(2, 10), DataType.FLOAT32);
        NDArray out = new LayerNorm(manager, 10).forward(x);
        for (float mean : out.mean(new int[]{-1}).toFloatArray()) {
            assertEquals(0f, mean, 1e-5);
        }
        for (float variance : out.square().mean(new int[]{-1}).toFloatArray()) {
            assertEquals(1f, variance, 1e-3);
        }
    }

    @Test
    void saveAndLoadGiveTheSameOutput(@TempDir Path dir) throws Exception {
        GPTModel original = new GPTModel(manager, SMALL);
        Path file = dir.resolve("model.params");
        original.save(file);

        GPTModel restored = new GPTModel(manager, SMALL);
        restored.load(file);

        NDArray ids = manager.create(new int[][]{{7, 8, 9}});
        assertArrayEquals(original.forward(ids).toFloatArray(), restored.forward(ids).toFloatArray(), 1e-6f);
    }
}
