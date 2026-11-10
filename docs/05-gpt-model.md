# 5. The GPT model: putting all the pieces together

**Book:** Chapter 4 · **Code:** [`model/`](../src/main/java/com/tinygpt/model), [`nn/`](../src/main/java/com/tinygpt/nn) · **Run:** `./run.sh 9` · **Class pictures:** [model](classes/5-model.md), [nn](classes/3-nn.md)

← [4. Attention](04-attention.md) · Next: [6. Training →](06-training.md)

```mermaid
flowchart LR
    A["0 · What is an LLM?"] --> B["1 · Tokens"] --> C["2 · Training data"] --> D["3 · Embeddings"] --> E["4 · Attention"] --> F["5 · GPT model"] --> G["6 · Training"] --> H["7 · Writing text"]
    classDef here fill:#ffd54f,stroke:#e65100,stroke-width:3px,color:#000
    class F here
```

---

## 🤔 The problem

We have the parts: tokens, embeddings and attention. But one attention layer on its own isn't very smart, and if we simply stack lots of layers, training **breaks down**: numbers grow wild and the learning signal fades away.

This chapter adds the "glue" that makes a **deep** stack work, then assembles the full GPT.

## 💡 The idea: a tall building of identical floors

> 🏢 GPT is a building. The ground floor (embeddings) turns words into numbers.
> Then come identical **floors** (transformer blocks), where words discuss and think.
> The roof (output head) announces a score for every possible next word.

```mermaid
flowchart TD
    R["🏠 roof: output head<br/>scores for 50,257 next tokens"]
    N["🎚️ final LayerNorm"]
    F4["🧱 floor 4: TransformerBlock"]
    F3["🧱 floor 3: TransformerBlock"]
    F2["🧱 floor 2: TransformerBlock"]
    F1["🧱 floor 1: TransformerBlock"]
    G["🚪 ground floor: token + position embeddings"]
    I["token IDs"]
    I --> G --> F1 --> F2 --> F3 --> F4 --> N --> R
```

The book starts with an empty skeleton of this building (*DummyGPTModel*, book 4.1), then builds each missing part. We'll do the same.

---

## 🧱 The four new pieces

### 1 · LayerNorm: keep the numbers at a steady size (book 4.2)

**Problem:** after many layers, some numbers become huge and others tiny. Training then becomes unstable, like a car that's either flooring the gas or slamming the brakes.

**Fix:** after each step, rescale every token's numbers so their **average is 0** and their **spread is 1**.

> 🎚️ A sound engineer keeps every singer at a comfortable volume. The *tune* stays the same; only the loudness is fixed.

```mermaid
flowchart LR
    A["[10, 20, 30]"] -->|"1 · average = 20<br/>subtract it"| B["[-10, 0, 10]"]
    B -->|"2 · spread = 8.16<br/>divide by it"| C["[-1.22, 0, 1.22]"]
    C -->|"3 · × scale + shift<br/>(learned)"| D["output"]
    A2["[1, 2, 3]"] -->|"same steps"| C
```

Both a "loud" `[10, 20, 30]` and a "quiet" `[1, 2, 3]` become `[-1.22, 0, 1.22]`.
The learnable `scale` and `shift` let the model choose a different volume if that helps.

### 2 · GELU: the smooth bend (book 4.3)

**Problem:** layers that only multiply and add can only learn straight-line patterns. Stacking 100 of them is still one straight line.

**Fix:** put a **bend** after the expanding layer. GELU lets positive numbers through and squashes negative ones, smoothly.
(The older ReLU makes a sharp corner at 0. GELU's smooth curve makes learning easier.)

```mermaid
xychart-beta
    title "GELU: input → output"
    x-axis "input" ["-3", "-2", "-1", "0", "1", "2", "3"]
    y-axis "output" -0.5 --> 3
    line [-0.004, -0.045, -0.159, 0, 0.841, 1.955, 2.996]
```

### 3 · FeedForward: each word's thinking time (book 4.3)

A small 2-layer network. Every token goes through it **on its own**: expand to 4× the size, bend, shrink back.

```mermaid
flowchart LR
    A["128 numbers"] -->|"Linear (expand)"| B["512 numbers<br/>room to think"]
    B -->|"GELU (bend)"| C["512 numbers"]
    C -->|"Linear (shrink)"| D["128 numbers"]
```

> 💭 Attention = a group discussion. FeedForward = quiet time to think over what you heard.

### 4 · Shortcut connections: keep the original (book 4.4)

**Problem: the vanishing signal.** During training, a "how should I change?" signal (the *gradient*) travels **backwards** from the top floor to the bottom.
Each floor it passes can shrink it, so the bottom floors hear almost nothing and barely learn.

> 📞 Like a game of telephone: by the time the message reaches the 20th person, it has faded to a whisper.

**Fix:** let each floor **add its result to its input**, instead of replacing it: `output = x + layer(x)`.
The signal then has a direct "express lane" around every floor.

```mermaid
flowchart LR
    X["x"] --> L["layer"] --> P(("➕"))
    X -->|"↪️ shortcut (express lane)"| P
    P --> Y["x + layer(x)"]
```

The book measured this on a 5-layer network: the strength of the learning signal reaching each layer.

```mermaid
xychart-beta
    title "WITHOUT shortcuts: the signal fades towards layer 1"
    x-axis "layer" ["1 (first)", "2", "3", "4", "5 (last)"]
    y-axis "signal strength" 0 --> 0.006
    bar [0.0002, 0.0001, 0.0007, 0.0014, 0.0050]
```

```mermaid
xychart-beta
    title "WITH shortcuts: every layer gets a strong signal"
    x-axis "layer" ["1 (first)", "2", "3", "4", "5 (last)"]
    y-axis "signal strength" 0 --> 1.4
    bar [0.22, 0.21, 0.33, 0.27, 1.33]
```

The first layer's signal is **about 1,000 times stronger** with shortcuts (0.22 vs 0.0002).

> 📝 Like editing an essay: keep the original text and write corrections in the margin, instead of rewriting it from scratch each time.

---

## 🧱 One floor: the TransformerBlock (book 4.5)

Now combine everything into one floor:

```mermaid
flowchart TD
    X["x: [8, 64, 128]"] --> N1["🎚️ LayerNorm"]
    N1 --> A["👀 MultiHeadAttention<br/>words discuss"]
    A --> D1["🎲 Dropout"]
    D1 --> P1(("➕"))
    X -->|"↪️ shortcut"| P1
    P1 --> N2["🎚️ LayerNorm"]
    N2 --> F["💭 FeedForward<br/>each word thinks"]
    F --> D2["🎲 Dropout"]
    D2 --> P2(("➕"))
    P1 -->|"↪️ shortcut"| P2
    P2 --> Y["output: [8, 64, 128]<br/>same shape as the input"]
```

```java
// TransformerBlock.forward(): the whole floor in two lines
x = x.add(dropShortcut.forward(attention.forward(norm1.forward(x))));     // discuss
x = x.add(dropShortcut.forward(feedForward.forward(norm2.forward(x))));   // think
```

**Same shape in and out.** That's what lets us stack as many floors as we like.

A tiny example of the ➕ shortcut for one token:

```
before the floor:        [1.0,  2.0]
the floor's suggestion:  [0.1, -0.2]
after the floor:         [1.1,  1.8]   ← a small correction, nothing thrown away
```

---

## 🏢 The whole building: GPTModel (book 4.6)

```mermaid
flowchart TD
    IN["token IDs [8, 64]"] --> TE["🗂️ token embedding"]
    IN --> PE["🗂️ position embedding"]
    TE --> ADD(("➕"))
    PE --> ADD
    ADD --> DO["🎲 Dropout"]
    DO --> B1["🧱 TransformerBlock 1"] --> B2["🧱 TransformerBlock 2"] --> B3["🧱 TransformerBlock 3"] --> B4["🧱 TransformerBlock 4"]
    B4 --> LN["🎚️ final LayerNorm"]
    LN --> OH["🧮 output head: Linear 128 → 50,257"]
    OH --> LG["logits [8, 64, 50,257]<br/>a score for every possible next token,<br/>at every position"]
```

### What are "logits"?

For each position, the model outputs **one score per token in the vocabulary** (50,257 scores). Higher = "more likely to come next".
They're raw scores, not percentages yet (softmax turns them into percentages later).

```mermaid
flowchart LR
    P["last position of<br/>'every effort moves you'"] --> S["' forward': 6.75 🥇<br/>' toward': 6.28 🥈<br/>' closer': 4.51<br/>...<br/>' pizza': -1.89<br/>(one score for each of 50,257 tokens)"]
```

*(These are the book's example scores from chapter 5.3. We use the same ones in [7. Writing text](07-generating-text.md).)*

### How big is it? (book 4.6)

```mermaid
pie title GPT-2 small: where the 163,009,536 numbers live (book's model)
    "Token embedding (50,257 × 768)" : 38597376
    "12 transformer blocks" : 85026816
    "Output head (768 × 50,257)" : 38597376
    "Position embedding (1,024 × 768)" : 786432
    "Final LayerNorm" : 1536
```

| | TinyGPT (`TINY`) | GPT-2 small (`GPT2_SMALL`) |
|---|---:|---:|
| floors (blocks) | 4 | 12 |
| heads per attention | 4 | 12 |
| numbers per token | 128 | 768 |
| longest input | 64 tokens | 1,024 tokens |
| learnable numbers | **13,665,792** | **163,009,536** |
| file size when saved | 55 MB | 652 MB |

OpenAI calls GPT-2 small "124M" because it **reuses** the token embedding table as the output head (weight tying): 163.0M − 38.6M ≈ 124.4M.

### Before training it writes nonsense (book 4.7)

All the numbers start random, so `./run.sh 9` gives:

```
Hello, I amás503 work hol riddledtaker
```

(The book got `Hello, I am Featureiman Byeswickattribute argue`. Different random numbers, same kind of nonsense.) Training comes next.

---

## 🏭 Watch the data move through `GPTModel.forward()`

One training batch through TinyGPT, with the shape at every stage:

```mermaid
flowchart TD
    A["NDArray tokenIds<br/>[8, 64]"] -->|"tokenEmbedding.forward"| B["[8, 64, 128]"]
    A2["arange(64)"] -->|"positionEmbedding.forward"| B2["[64, 128]"]
    B --> C(("add"))
    B2 --> C
    C -->|"dropout"| D["[8, 64, 128]"]
    D -->|"block 1 · block 2 · block 3 · block 4<br/>(each: [8,64,128] → [8,64,128])"| E["[8, 64, 128]"]
    E -->|"finalNorm"| F["[8, 64, 128]"]
    F -->|"outputHead (128 → 50,257)"| G["logits [8, 64, 50,257]<br/>= 25.7 million scores!"]
    G -->|"training"| L["6 · Loss: compare with targets"]
    G -->|"writing"| W["7 · TextGenerator: last position only"]
```

```mermaid
sequenceDiagram
    participant C as caller (Trainer or TextGenerator)
    participant G as GPTModel
    participant E as Embedding ×2
    participant T as TransformerBlock ×4
    participant N as LayerNorm
    participant H as Linear (output head)
    C->>G: forward(tokenIds [8, 64])
    G->>E: token rows + position rows
    E-->>G: [8, 64, 128]
    loop 4 floors
        G->>T: forward(x)
        T-->>G: x (same shape, smarter)
    end
    G->>N: forward(x)
    G->>H: forward(x)
    H-->>C: logits [8, 64, 50257]
```

---

## 🔗 Where every building block is used

How many times each brick appears inside TinyGPT:

```mermaid
flowchart LR
    subgraph GPT["GPTModel"]
        E["Embedding ×2"]
        TB["TransformerBlock ×4"]
        LNf["LayerNorm ×1 (final)"]
        H["Linear ×1 (output head)"]
        D0["Dropout ×1"]
    end
    subgraph BLK["inside each TransformerBlock"]
        LN2["LayerNorm ×2"]
        MHA["MultiHeadAttention ×1<br/>(4 Linear + 1 Dropout inside)"]
        FF["FeedForward ×1<br/>(2 Linear + GELU inside)"]
        D1["Dropout ×1 (used twice)"]
    end
    TB --> BLK
```

| Brick | Where | Total in TinyGPT |
|---|---|---:|
| `Embedding` | token table, position table | 2 |
| `Linear` | Q, K, V, attention output (×4 per block), FeedForward expand + shrink (×2 per block), output head | 4×6 + 1 = **25** |
| `LayerNorm` | 2 per block + 1 final | 2×4 + 1 = **9** |
| `GELU` | 1 per FeedForward | **4** |
| `Dropout` | after embeddings, attention weights, after each half of a block | 1 + 4×2 = **9** modules |
| `MultiHeadAttention` | 1 per block | **4** |
| `FeedForward` | 1 per block | **4** |

All of them are `Module`s, so `model.parameters()` can collect **every learnable number in the building** for the optimizer.

---

## ▶️ Try it

```bash
./run.sh 9     # builds the full GPT-2 small (needs about 1 GB of memory)
```

**Experiments**
1. In `Step09_GPTModel`, use `GPTConfig.TINY` instead. How many parameters now?
2. Make your own config with 8 layers: `new GPTConfig(50257, 64, 128, 4, 8, 0.1f, false)`. How much does the count grow? (Each block adds 197,888.)
3. In `TransformerBlock.forward`, remove the `x.add(...)` shortcuts and train. What happens to the loss?

## 🧠 Remember

- **LayerNorm** keeps numbers steady; **GELU** adds the bend that makes learning powerful; **FeedForward** is per-token thinking.
- **Shortcuts** (`x + layer(x)`) keep the learning signal strong, so deep models can train.
- A **TransformerBlock** = discuss (attention) + think (feed forward), with a norm and a shortcut around each. Same shape in and out.
- **GPTModel** = embeddings → N blocks → final norm → output head → **logits** (a score per possible next token).

## 📖 In the book

| Section | Topic | Our code |
|---|---|---|
| 4.1 | GPT config, DummyGPTModel skeleton | `GPTConfig` |
| 4.2 | Layer normalization | `LayerNorm` |
| 4.3 | GELU and feed forward | `GELU`, `FeedForward` |
| 4.4 | Shortcut connections, vanishing gradients | `TransformerBlock` (`x.add(...)`) |
| 4.5 | Transformer block | `TransformerBlock` |
| 4.6 | GPT model, 163,009,536 parameters, weight tying | `GPTModel`, `./run.sh 9` |
| 4.7 | Generating text (untrained) | `TextGenerator.generateGreedy`, `./run.sh 9` |
