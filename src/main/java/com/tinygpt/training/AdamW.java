package com.tinygpt.training;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;

import java.util.ArrayList;
import java.util.List;

/**
 * The optimizer: it uses the gradients to nudge every weight in the right direction.
 *
 * Imagine walking down a foggy hill to reach the bottom (the lowest loss):
 * <ul>
 *   <li>The <b>gradient</b> tells you which way is uphill. So you step the other way.</li>
 *   <li><b>Momentum</b> (m) remembers your recent direction, so you roll smoothly instead of zig-zagging.</li>
 *   <li><b>Scaling</b> (v) takes smaller steps where the ground is bumpy and bigger steps where it is smooth.</li>
 *   <li><b>Weight decay</b> gently pulls every weight towards 0, so no weight grows too big.</li>
 * </ul>
 *
 * <pre>
 *   m = β1·m + (1-β1)·grad
 *   v = β2·v + (1-β2)·grad²
 *   weight -= learningRate · ( m̂ / (√v̂ + ε)  +  weightDecay · weight )
 * </pre>
 *
 * This is the same as {@code torch.optim.AdamW}, used in the book.
 *
 * Book: Chapter 5.2
 *
 * Picture guide: docs/classes/6-training.md#adamw · line by line: docs/walkthroughs/step-tinygpt-training.md
 */
public class AdamW {

    private static final float BETA1 = 0.9f;
    private static final float BETA2 = 0.999f;
    private static final float EPSILON = 1e-8f;

    private final List<NDArray> parameters;
    private final List<NDArray> momentum = new ArrayList<>();
    private final List<NDArray> velocity = new ArrayList<>();
    private final float learningRate;
    private final float weightDecay;
    private int step = 0;

    public AdamW(List<NDArray> parameters, float learningRate, float weightDecay) {
        this.parameters = parameters;
        this.learningRate = learningRate;
        this.weightDecay = weightDecay;
        for (NDArray p : parameters) {
            momentum.add(p.zerosLike());
            velocity.add(p.zerosLike());
        }
    }

    /**
     * Update every parameter once, using the gradients from the last backward pass.
     *
     * @return false if the step was skipped because a gradient was NaN or infinite.
     *         One bad step would otherwise turn every weight into NaN forever.
     *         (PyTorch's mixed-precision GradScaler skips bad steps the same way.)
     */
    public boolean step() {
        if (!gradientsAreFinite()) {
            return false;
        }
        step++;
        float correction1 = 1f - (float) Math.pow(BETA1, step);
        float correction2 = 1f - (float) Math.pow(BETA2, step);

        for (int i = 0; i < parameters.size(); i++) {
            NDArray weight = parameters.get(i);
            NDArray gradient = weight.getGradient();
            NDArray m = momentum.get(i);
            NDArray v = velocity.get(i);

            m.muli(BETA1).addi(gradient.mul(1f - BETA1));
            v.muli(BETA2).addi(gradient.square().mul(1f - BETA2));

            NDArray mHat = m.div(correction1);
            NDArray vHat = v.div(correction2);
            NDArray update = mHat.div(vHat.sqrt().add(EPSILON)).add(weight.mul(weightDecay));
            weight.subi(update.mul(learningRate));
        }
        return true;
    }

    private boolean gradientsAreFinite() {
        for (NDArray weight : parameters) {
            NDArray gradient = weight.getGradient();
            if (gradient.isNaN().any().getBoolean() || gradient.isInfinite().any().getBoolean()) {
                return false;
            }
        }
        return true;
    }

    /** Reset every gradient to 0, ready for the next backward pass (PyTorch: {@code optimizer.zero_grad()}). */
    public void zeroGradients() {
        for (NDArray weight : parameters) {
            weight.getGradient().set(new NDIndex(":"), 0f);   // overwrite with 0 (also clears any NaN)
        }
    }
}
