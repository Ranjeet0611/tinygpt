package com.tinygpt.training;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.training.GradientCollector;
import com.tinygpt.data.Batch;
import com.tinygpt.generation.TextGenerator;
import com.tinygpt.model.GPTModel;

/**
 * The training loop. Every step is the same 5 moves:
 *
 * <pre>
 *   1. take a batch     inputs + targets
 *   2. forward          model(inputs) ─► logits
 *   3. loss             how wrong were the guesses?
 *   4. backward         gradients: which way should each weight move?
 *   5. optimizer step   move every weight a tiny bit, then reset the gradients to 0
 *   ...repeat for every batch, for several epochs (one epoch = one pass over all the data)
 * </pre>
 *
 * Every few steps we also measure the loss on text the model never trains on (validation).
 * If training loss keeps falling but validation loss goes up, the model is memorising
 * instead of learning ("overfitting").
 *
 * Book: Chapter 5.2 ({@code train_model_simple})
 *
 * Picture guide: docs/classes/6-training.md#trainer · line by line: docs/walkthroughs/step-tinygpt-training.md
 */
public class Trainer {

    private final GPTModel model;
    private final NDManager manager;
    private final AdamW optimizer;
    private int skippedSteps = 0;

    public Trainer(GPTModel model, NDManager manager, float learningRate, float weightDecay) {
        this.model = model;
        this.manager = manager;
        this.optimizer = new AdamW(model.parameters(), learningRate, weightDecay);
    }

    /**
     * @param evalEvery     measure train/validation loss every this many steps
     * @param samplePrompt  after each epoch, show what the model writes for this prompt
     */
    public void train(Iterable<Batch> trainLoader, Iterable<Batch> validationLoader, int epochs, int evalEvery,
                      TextGenerator generator, String samplePrompt) {
        train(trainLoader, validationLoader, epochs, evalEvery, () -> {
            String sample = generator.generateGreedy(samplePrompt, 25);
            System.out.println("   sample: " + sample.replace("\n", " "));
        });
    }

    /**
     * @param evalEvery       measure train/validation loss every this many steps
     * @param afterEachEpoch  runs at the end of every epoch, e.g. to print a sample
     */
    public void train(Iterable<Batch> trainLoader, Iterable<Batch> validationLoader, int epochs, int evalEvery,
                      Runnable afterEachEpoch) {
        int globalStep = 0;
        long tokensSeen = 0;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            model.setTraining(true);
            for (Batch batch : trainLoader) {
                trainStep(batch);
                globalStep++;
                tokensSeen += (long) batch.inputs().length * batch.inputs()[0].length;

                if (globalStep % evalEvery == 0) {
                    float trainLoss = averageLoss(trainLoader, 5);
                    float validationLoss = averageLoss(validationLoader, 5);
                    System.out.printf("Epoch %2d  step %4d  tokens seen %7d  |  train loss %.3f  validation loss %.3f%n",
                            epoch, globalStep, tokensSeen, trainLoss, validationLoss);
                }
            }
            afterEachEpoch.run();
        }
        if (skippedSteps > 0) {
            System.out.printf("(Skipped %d of %d steps because a gradient came back as NaN.)%n", skippedSteps, globalStep);
        }
    }

    /** The 5 moves of one training step. */
    private void trainStep(Batch batch) {
        // NDScope frees every temporary array created inside it when the block ends.
        try (NDScope ignored = new NDScope()) {
            NDArray inputs = manager.create(batch.inputs());
            NDArray targets = manager.create(batch.targets());

            try (GradientCollector gradients = Engine.getInstance().newGradientCollector()) {
                NDArray logits = model.forward(inputs);                      // forward
                NDArray loss = Loss.crossEntropy(logits, targets);           // loss
                gradients.backward(loss);                                    // backward
            }
            if (!optimizer.step()) {                                         // update weights
                skippedSteps++;
            }
            optimizer.zeroGradients();                                       // clean slate for the next step
        }
    }

    /** Average loss over the first {@code maxBatches} batches, without training. */
    public float averageLoss(Iterable<Batch> loader, int maxBatches) {
        boolean wasTraining = model.isTraining();
        model.setTraining(false);
        float total = 0;
        int count = 0;
        for (Batch batch : loader) {
            if (count == maxBatches) {
                break;
            }
            try (NDScope ignored = new NDScope()) {
                NDArray logits = model.forward(manager.create(batch.inputs()));
                total += Loss.crossEntropy(logits, manager.create(batch.targets())).getFloat();
            }
            count++;
        }
        model.setTraining(wasTraining);
        return count == 0 ? Float.NaN : total / count;
    }
}
