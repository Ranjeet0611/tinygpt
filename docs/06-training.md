# 6. Training: teaching the model

**Book:** Chapter 5.1, 5.2, 5.4 · **Code:** [`training/`](../src/main/java/com/tinygpt/training), [`TinyGPT.java`](../src/main/java/com/tinygpt/TinyGPT.java) · **Run:** `./run.sh 10`, `./run.sh train` · **Class pictures:** [training classes](classes/6-training.md)

← [5. The GPT model](05-gpt-model.md) · Next: [7. Writing text →](07-generating-text.md)

```mermaid
flowchart LR
    A["0 · What is an LLM?"] --> B["1 · Tokens"] --> C["2 · Training data"] --> D["3 · Embeddings"] --> E["4 · Attention"] --> F["5 · GPT model"] --> G["6 · Training"] --> H["7 · Writing text"]
    classDef here fill:#ffd54f,stroke:#e65100,stroke-width:3px,color:#000
    class G here
```

---

## 🤔 The problem

The model is built, but all of its 13.7 million numbers are **random**, so it writes nonsense.
Nobody could set 13.7 million numbers by hand. We need a way for the model to **find good numbers by itself**.

## 💡 The idea: practise, check, adjust, repeat

> 🎯 Learning to throw darts: throw, see how far you missed, adjust your arm **a little**, throw again.
> You never calculate the perfect throw. You just get a bit less wrong each time, thousands of times.

```mermaid
flowchart LR
    A["1 · 📦 take a batch<br/>of quizzes"] --> B["2 · 🧠 model guesses<br/>(forward)"]
    B --> C["3 · 🎯 how wrong?<br/>(loss)"]
    C --> D["4 · 📐 which way should<br/>each number move?<br/>(backward)"]
    D --> E["5 · ⛰️ move every number<br/>a tiny bit<br/>(optimizer step)"]
    E -->|"next batch"| A
```

To do this we need three tools:
1. a way to **measure** how wrong the model is: the **loss**;
2. a way to find **which direction** to change each number: **gradients** (backward pass);
3. a rule for **how far** to move each number: the **optimizer** (AdamW).

---

## 🔍 Tool 1 · Loss: one number for "how wrong?" (book 5.1)

### The book's example

Two inputs, with the right answers shifted one token to the right (just like [2. Training data](02-training-data.md)):

| Input | Target (correct next tokens) |
|---|---|
| `every effort moves` | ` effort moves you` |
| `I really like` | ` really like chocolate` |

That's **6 quizzes**: after "every" → "effort", after "every effort" → "moves", ... after "I really like" → "chocolate".

### Look at the probability given to the right answer

The model's scores become percentages (softmax). We only care about the percentage given to the **correct** token.
Real numbers from `./run.sh 10` (the untrained TinyGPT):

| After... | Correct token | Probability the model gave it |
|---|---|---:|
| "every" | " effort" | 0.0000226 |
| "every effort" | " moves" | 0.0000124 |
| "every effort moves" | " you" | 0.0000106 |
| "I" | " really" | 0.0000078 |
| "I really" | " like" | 0.0000130 |
| "I really like" | " chocolate" | 0.0000257 |

A blind guess among 50,257 tokens would be 1/50,257 = 0.0000199. The untrained model is no better than guessing. 🎲

### The six steps from scores to loss (book figure 5.7)

```mermaid
flowchart TD
    S1["1 · logits<br/>raw scores [2, 3, 50,257]"] -->|"softmax"| S2["2 · probabilities<br/>every token gets a %"]
    S2 -->|"pick the correct token's %"| S3["3 · target probabilities<br/>[0.0000226, 0.0000124, 0.0000106, ...]"]
    S3 -->|"natural log (ln)"| S4["4 · log probabilities<br/>[-10.70, -11.30, -11.45, -11.76, -11.25, -10.57]"]
    S4 -->|"average"| S5["5 · average: -11.17"]
    S5 -->|"× -1"| S6["6 · loss = 11.17 ✅<br/>(cross-entropy)"]
```

**Why the logarithm?** It turns "tiny probability" into "big penalty" in a smooth way that's easy to learn from:

| Probability of the right answer | Loss (−ln p) | Feels like |
|---:|---:|---|
| 100% | 0.00 | 🎯 bullseye |
| 90% | 0.11 | very close |
| 50% | 0.69 | okay |
| 10% | 2.30 | far off |
| 1% | 4.61 | way off |
| 0.002% (random guess) | 10.82 | blindfolded |

**Training = making this number go down.** A perfect model would score 0.

### Perplexity: the loss in plain words

`perplexity = e^loss` ≈ "**how many tokens is the model still choosing between?**"

```
untrained:  loss 11.17 → perplexity ≈ 71,000   ("could be almost any token")
trained:    loss  0.77 → perplexity ≈ 2.2      ("it's one of about two tokens")
```

### How our code picks the right answer: one-hot

```
correct token is #2 of 5  →  one-hot [0, 0, 1, 0, 0]
log-probabilities          [-3.1, -2.4, -0.9, -4.0, -2.2]
multiply, then add up  →   -0.9   ← the right answer's log-probability
```

```java
// Loss.crossEntropy()
NDArray logProbabilities = logits.logSoftmax(-1);                       // steps 1, 2, 4
NDArray oneHot = targets.oneHot(vocabularySize);
NDArray correct = logProbabilities.mul(oneHot).sum(new int[]{-1});      // step 3
return correct.mean().neg();                                            // steps 5, 6
```

---

## 🔍 Tool 2 · Gradients: which way should each number move?

For **each** of the 13.7 million numbers we ask:

> *"If I nudge this number up a tiny bit, does the loss go up or down? By how much?"*

The answer is that number's **gradient**. Working it out by hand would be impossible, but **autograd** does it automatically.
During the forward pass it remembers every calculation; then `backward()` replays them **in reverse**, passing "blame" from the loss back to every number.

```mermaid
flowchart RL
    L["🎯 loss = 11.17"] -->|"blame"| H["output head"]
    H -->|"blame"| B4["block 4"]
    B4 -->|"blame"| B3["block 3"]
    B3 -->|"blame"| B2["block 2"]
    B2 -->|"blame"| B1["block 1"]
    B1 -->|"blame"| E["embeddings"]
```

> 🏭 Like a factory that made a faulty product: you walk back along the line, and each station learns how much it contributed to the mistake.

This is why **shortcut connections** matter ([5. The GPT model](05-gpt-model.md)): they keep the blame signal strong all the way down to the embeddings.

---

## 🔍 Tool 3 · The optimizer: AdamW

### The simplest version: walk downhill

> ⛰️ You're on a foggy hill and want to reach the valley (lowest loss). You can't see it. You can only feel the slope under your feet (the gradient).
> So you take a small step **downhill**. Then feel again, and step again.

```
number = 5,  gradient = +2 (uphill is "bigger"),  learning rate = 0.1
new number = 5 − 0.1 × 2 = 4.8          ← one small step downhill
```

### AdamW's extra tricks

```mermaid
flowchart TD
    G["📐 gradient: which way is uphill"] --> U["the step"]
    M["🏃 momentum<br/>remember recent directions:<br/>roll smoothly, don't zig-zag"] --> U
    V["👣 adaptive step size<br/>small steps where the slope is jumpy,<br/>bigger where it's steady"] --> U
    W["🧲 weight decay<br/>gently pull every number towards 0,<br/>so none grows too big"] --> U
    LR["learning rate<br/>(0.001 in TinyGPT)"] --> U
    U --> N["number = number − step"]
```

| Learning rate | What happens |
|---|---|
| too small | 🐌 learns very slowly |
| just right | ✅ the loss goes down steadily |
| too big | 🦘 jumps over the valley; the loss bounces or explodes |

### Safety check: skip a broken step

If a gradient ever comes back as NaN ("not a number"), one update would spread it into every number and ruin the model.
So `AdamW.step()` first checks all gradients, and **skips** the step if any is broken. PyTorch's mixed-precision trainer does the same.
(Why this can happen here: see the README's "Known issue".)

---

## 🏋️ The training loop (book 5.2)

### The book's `train_model_simple`, in our `Trainer`

```mermaid
flowchart TD
    START(["for epoch = 1 to 25"]) --> TRAIN["model.setTraining(true)<br/>(dropout on)"]
    TRAIN --> B["📦 next batch from the loader"]
    B --> F["forward: logits = model.forward(inputs)"]
    F --> L["loss = Loss.crossEntropy(logits, targets)"]
    L --> BW["gradients.backward(loss)"]
    BW --> ST["optimizer.step()"]
    ST --> ZG["optimizer.zeroGradients()<br/>(wipe the old blame clean)"]
    ZG --> EV{"every 10 steps?"}
    EV -->|"yes"| E["📝 measure loss on<br/>practice text AND unseen text"]
    EV -->|"no"| NB
    E --> NB{"more batches?"}
    NB -->|"yes"| B
    NB -->|"no"| SAMPLE["✍️ print a sample sentence"]
    SAMPLE --> START
```

**Why wipe the gradients?** Autograd **adds** new gradients on top of old ones. Without wiping, each step would also use the blame from every earlier step.

### 🏭 Watch the data move through one training step

```mermaid
flowchart TD
    B["Batch<br/>inputs int[8][64]<br/>targets int[8][64]"] -->|"manager.create"| N["NDArrays [8, 64]"]
    N -->|"model.forward(inputs)"| LG["logits [8, 64, 50,257]"]
    LG -->|"Loss.crossEntropy(logits, targets)"| LS["loss: one number, e.g. 5.98"]
    LS -->|"gradients.backward(loss)"| GR["a gradient next to every one<br/>of the 13,665,792 numbers"]
    GR -->|"optimizer.step()"| UP["all 13,665,792 numbers<br/>nudged a tiny bit"]
    UP -->|"optimizer.zeroGradients()"| CL["gradients reset to 0"]
    N -.->|"NDScope frees all the temporary<br/>grids when the step ends"| CL
```

```java
// Trainer.trainStep(): the 5 moves
try (NDScope ignored = new NDScope()) {                        // free temporary grids afterwards
    NDArray inputs  = manager.create(batch.inputs());          // 1. batch
    NDArray targets = manager.create(batch.targets());
    try (GradientCollector gradients = ...) {
        NDArray logits = model.forward(inputs);                // 2. forward
        NDArray loss = Loss.crossEntropy(logits, targets);     // 3. loss
        gradients.backward(loss);                              // 4. backward
    }
    optimizer.step();                                          // 5. update
    optimizer.zeroGradients();                                 //    wipe for next time
}
```

### Counting the work

```
4,612 practice tokens ÷ 64 per window   = 72 windows
72 windows ÷ 8 per batch                = 9 steps per epoch
9 steps × 25 epochs                     = 225 steps
time on a laptop                        ≈ 20 seconds
```

---

## 📈 What happened when we trained (real numbers from `./run.sh train`)

```mermaid
xychart-beta
    title "Loss during training (lower is better)"
    x-axis "training step" [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220]
    y-axis "loss" 0 --> 10
    line [9.32, 7.55, 6.42, 5.98, 5.81, 5.51, 5.33, 5.04, 4.74, 4.45, 4.09, 3.82, 3.46, 3.09, 2.71, 2.36, 2.02, 1.72, 1.44, 1.21, 0.97, 0.77]
    line [9.49, 7.83, 6.81, 6.58, 6.53, 6.46, 6.40, 6.33, 6.27, 6.22, 6.19, 6.16, 6.15, 6.14, 6.18, 6.23, 6.32, 6.33, 6.37, 6.45, 6.55, 6.62]
```

*Lower line: loss on practice text. Upper line: loss on unseen text (validation).*

### What it wrote along the way

| After | `Every effort moves you` ... |
|---|---|
| 0 steps | ` Hero farmland tenth harness ATL subsidized Contracts fixes impairment emission` |
| 27 steps (epoch 3) | `,,,,,,,,,,,,,,,,,,,,,,,,,` |
| 54 steps (epoch 6) | `, the, the, the the, the, the.` |
| 108 steps (epoch 12) | ` know the the the that the one of the, and "  "  " of the fact, and I had` |
| 153 steps (epoch 17) | ` know to the Riv of the picture for a smile in a little of the picture.` |
| 225 steps (epoch 25) | ` know; and I had the craft was vindicated by little Claude Nutley, who, in a footstep and my bravura tricks.` |

```mermaid
flowchart LR
    A["random words"] --> B["the most common token<br/>(,)"] --> C["common words<br/>(the, and)"] --> D["short phrases"] --> E["whole sentences<br/>from the story"]
```

Just like a toddler: first sounds, then common words, then phrases, then sentences.

### ⚠️ Overfitting: memorising instead of learning (book 5.2)

Look at the chart after step 140: the **practice loss keeps falling, but the unseen-text loss starts rising**.

```mermaid
flowchart LR
    A["steps 0-140<br/>both losses fall<br/>✅ learning real patterns"] --> B["after step 140<br/>practice ⬇ but unseen ⬆<br/>⚠️ memorising the story"]
```

> 🧑‍🎓 A student who memorises last year's exam: perfect on those exact questions, lost on new ones.

The book shows exactly this. With only ~5,000 tokens of text, a model can memorise it. Real LLMs train on **billions** of tokens, far too many to memorise, so they have to learn general patterns.

### Our settings vs the book's

| | Book (GPT-2 small) | TinyGPT |
|---|---|---|
| model size | 163M numbers | 13.7M numbers |
| context length | 256 | 64 |
| learning rate | 0.0004 | 0.001 (small models can take bigger steps) |
| epochs | 10 | 25 |
| weight decay | 0.1 | 0.1 |
| final losses | train ≈ 0.4, validation ≈ 6.5 | train ≈ 0.8, validation ≈ 6.6 |

---

## 💾 Saving and loading (book 5.4)

Training takes time, so we save the learned numbers to a file and load them later.

```mermaid
flowchart LR
    M1["🧠 trained model"] -->|"model.save()<br/>write every parameter"| F[("💾 models/tinygpt.params<br/>55 MB")]
    F -->|"model.load()<br/>read them back in the same order"| M2["🧠 new model with the same config<br/>= exactly the same behaviour"]
    M2 --> G["./run.sh generate 'I had always'"]
```

The file only stores **numbers**, not the design. So the model you load into must have the **same `GPTConfig`**, or the numbers won't fit.

---

## 🔗 How training connects to everything

```mermaid
flowchart TD
    DATA["2 · GPTDataLoader"] -->|"inputs"| MODEL["5 · GPTModel"]
    DATA -->|"targets"| LOSS["🎯 Loss"]
    MODEL -->|"logits"| LOSS
    LOSS -->|"backward()"| GRAD["📐 a gradient for every number"]
    MODULE["Module.parameters()<br/>collects every number from<br/>Embedding, Linear, LayerNorm..."] --> ADAM["⛰️ AdamW"]
    GRAD --> ADAM
    ADAM -->|"updates"| MODEL
    TRAINER["🏋️ Trainer"] -.->|"runs the loop"| DATA
    TRAINER -.->|"prints samples using"| GEN["7 · TextGenerator"]
    MODEL -->|"save()"| FILE[("💾 file")]
```

- Training touches **every** part of the model: each `Embedding`, `Linear` and `LayerNorm` hands its numbers to the optimizer through `Module.parameters()`.
- **Dropout** is on during training (`setTraining(true)`) and off when measuring or writing (`setTraining(false)`).

---

## ▶️ Try it

```bash
./run.sh 10       # the loss of an untrained model
./run.sh train    # train, save, then write
```

**Experiments** (change the constants at the top of `TinyGPT.java`):
1. `EPOCHS = 10`: less memorising. Is the writing worse?
2. `LEARNING_RATE = 1e-2f` (10× bigger): what happens to the loss?
3. `LEARNING_RATE = 1e-4f` (10× smaller): how far does the loss get in 25 epochs?
4. An even smaller model: `new GPTConfig(50257, 64, 64, 2, 2, 0.1f, false)`. Does it still learn?

## 🧠 Remember

- **Loss** = how wrong (−ln of the probability given to the right answer, averaged). Random ≈ 10.8, perfect = 0.
- **Backward** gives every number a **gradient**: which way to move to lower the loss.
- **AdamW** moves every number a small step downhill, with momentum and smart step sizes.
- Loop: **batch → forward → loss → backward → step → wipe gradients**, many times.
- Watch the **validation loss**: if it rises while the training loss falls, the model is **memorising**.

## 📖 In the book

| Section | Topic | Our code |
|---|---|---|
| 5.1.1 | Generating text with the untrained model | `TinyGPT` "Before training" line |
| 5.1.2 | The 6-step loss, cross-entropy, perplexity | `Loss`, `./run.sh 10` |
| 5.1.3 | Train/validation split and losses | `TinyGPT.main`, `Trainer.averageLoss` |
| 5.2 | `train_model_simple`, AdamW, overfitting | `Trainer`, `AdamW`, `./run.sh train` |
| 5.4 | Saving and loading weights | `GPTModel.save()` / `load()` |
| Appendix A.4, A.7 | Autograd, the typical training loop | Tool 2, the training loop |
