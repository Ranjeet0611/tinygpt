# 8. How everything connects

**Book:** chapters 2–5 together · **Code:** everything · **Run:** `./run.sh train`, then `./run.sh generate "Every effort moves"`

← [7. Writing text](07-generating-text.md) · Next: [9. Chat with GPT-2 →](09-chat-with-gpt2.md)

The other docs zoom in on one part each. This one **zooms out**: we follow a single sentence through the whole machine, and see how the parts depend on each other.

---

## 🧭 The whole machine on one page

```mermaid
flowchart TD
    subgraph PREP["📖 Chapter 2 · getting text ready"]
        ST["SampleText<br/>the story"] --> TK["BpeTokenizer<br/>text → IDs"]
        TK --> DS["GPTDataset<br/>windows: input + target"]
        DS --> DL["GPTDataLoader<br/>batches of 8"]
    end
    subgraph MODEL["🧠 Chapters 3–4 · GPTModel"]
        EM["Embedding ×2<br/>what + where"] --> BL["TransformerBlock ×4<br/>LayerNorm · MultiHeadAttention · FeedForward"]
        BL --> FN["LayerNorm"] --> OH["Linear output head"]
    end
    subgraph LEARN["🏋️ Chapter 5 · learning"]
        LS["Loss"] --> AD["AdamW"]
        TR["Trainer"]
    end
    subgraph WRITE["✍️ Chapter 5 · writing"]
        TG["TextGenerator"]
    end
    DL -->|"inputs"| EM
    DL -->|"targets"| LS
    OH -->|"logits"| LS
    AD -->|"updates all numbers"| MODEL
    TR -.->|"runs"| DL
    OH -->|"logits"| TG
    TK -->|"encode / decode"| TG
```

---

## 🚂 Journey 1 · One sentence during **training**

Let's follow the window `"Every effort moves you"` through one training step (sizes from TinyGPT).

```mermaid
flowchart TD
    A["1 · text<br/>'Every effort moves you'"] -->|"BpeTokenizer.encode"| B["2 · IDs<br/>[6109, 3626, 6100, 345]"]
    B -->|"GPTDataset: input / target"| C["3 · input  [6109, 3626, 6100]<br/>target [3626, 6100, 345]"]
    C -->|"GPTDataLoader → NDArray"| D["4 · input grid [1, 3]"]
    D -->|"token Embedding + position Embedding"| E["5 · [1, 3, 128]<br/>each token: 128 numbers (what + where)"]
    E -->|"4 × TransformerBlock<br/>(attention: tokens look back; feed forward: think)"| F["6 · [1, 3, 128]<br/>each token now knows about the ones before it"]
    F -->|"final LayerNorm + output head"| G["7 · logits [1, 3, 50,257]<br/>a score for every possible next token, at each position"]
    G -->|"Loss: compare with target<br/>' effort', ' moves', ' you'"| H["8 · loss, e.g. 11.2 (untrained)"]
    H -->|"backward()"| I["9 · a gradient for each of the 13.7M numbers"]
    I -->|"AdamW.step()"| J["10 · every number nudged a tiny bit"]
    J -.->|"repeat 225 times with all the windows"| D
```

What each position is quizzed on in this one window (the causal mask keeps it fair):

| Position | Can see | Must guess |
|---:|---|---|
| 0 | "Every" | " effort" |
| 1 | "Every effort" | " moves" |
| 2 | "Every effort moves" | " you" |

---

## 🚂 Journey 2 · One sentence during **writing**

Now the trained model continues `"Every effort moves"`:

```mermaid
flowchart TD
    A["text 'Every effort moves'"] -->|"BpeTokenizer.encode"| B["[6109, 3626, 6100]"]
    B -->|"GPTModel.forward<br/>(dropout off)"| C["logits [1, 3, 50,257]"]
    C -->|"take the LAST position only"| D["50,257 scores"]
    D -->|"top-k · temperature · softmax · 🎲<br/>(or greedy: the biggest)"| E["' you' (345)"]
    E -->|"append"| F["[6109, 3626, 6100, 345]"]
    F -.->|"repeat for the next token"| C
    F -->|"BpeTokenizer.decode at the end"| G["'Every effort moves you know; and I...'"]
```

### Training vs writing, side by side

| | Training | Writing |
|---|---|---|
| Input | windows from the story | your prompt |
| Positions used | **all** (every position is a quiz) | **only the last** |
| Dropout | on | off |
| Loss, gradients, AdamW | ✅ used | ❌ not used |
| Numbers change? | yes, every step | no, read only |
| Output | a lower loss | new text |

---

## 🔗 The settings that tie everything together

Three numbers from `GPTConfig` and the tokenizer decide the size of many parts at once:

```mermaid
flowchart LR
    V["📖 vocabularySize<br/>50,257"] --> V1["token Embedding: 50,257 rows"]
    V --> V2["output head: 50,257 scores"]
    V --> V3["Loss: one-hot of 50,257"]
    V --> V4["TextGenerator: picks from 50,257"]

    C["📏 contextLength<br/>64"] --> C1["GPTDataset: window of 64"]
    C --> C2["position Embedding: 64 rows"]
    C --> C3["causal mask: up to 64 × 64"]
    C --> C4["TextGenerator: feed the last 64"]

    D["📐 embeddingDim<br/>128"] --> D1["both Embeddings: 128 per row"]
    D --> D2["attention: 128 in, 128 out"]
    D --> D3["LayerNorm: over 128"]
    D --> D4["FeedForward: 128 → 512 → 128"]
    D --> D5["output head: 128 in"]
```

Change one of these, and every part connected to it changes size automatically, because they all read it from the same `GPTConfig`.

---

## 🔁 Ideas that appear more than once

The same few ideas are reused all over the model. Once you understand them, you understand most of the code.

### Softmax: "scores → percentages"

```mermaid
flowchart LR
    SM(("softmax"))
    SM --> A["4 · Attention<br/>how much to look at each word"]
    SM --> B["6 · Loss<br/>probability of the right answer"]
    SM --> C["7 · Writing<br/>chances for the next token"]
```

### Linear: "multiply by learned weights and add up"

```mermaid
flowchart LR
    L(("Linear"))
    L --> Q["Query"]
    L --> K["Key"]
    L --> V["Value"]
    L --> O["attention output"]
    L --> F1["FeedForward expand"]
    L --> F2["FeedForward shrink"]
    L --> H["output head"]
```

Six per block × 4 blocks + the output head = **25 `Linear` layers** in TinyGPT.

### Embedding: "look up a row"

Used **twice**: for the token (*what*) and for the position (*where*). The output head does the opposite job (vector → a score per token).

### Add (➕): mixing information without losing any

| Where | What is added |
|---|---|
| Embeddings | token row + position row |
| Causal mask | scores + (0 or −∞) |
| Shortcuts | x + attention(x), x + feedForward(x) |
| Linear | the bias |

### Dropout: random switching-off during practice

Used **9 times**: once after the embeddings, and twice per block (on the attention weights, and on each half's output). Always switched off when writing.

### LayerNorm: steady volume

Used **9 times**: twice per block (before attention, before feed forward) and once at the very end.

---

## 🧮 Who owns which learnable numbers

```mermaid
flowchart TD
    G["GPTModel: 13,665,792 numbers"]
    G --> TE["token Embedding<br/>6,432,896"]
    G --> PE["position Embedding<br/>8,192"]
    G --> BS["4 × TransformerBlock<br/>4 × 197,888 = 791,552"]
    G --> FN["final LayerNorm<br/>256"]
    G --> OH["output head<br/>6,432,896"]
    BS --> B1["one block = 197,888"]
    B1 --> L1["2 × LayerNorm: 512"]
    B1 --> AT["MultiHeadAttention: 65,664<br/>(Q, K, V: 3 × 16,384 · output: 16,512)"]
    B1 --> FF["FeedForward: 131,712<br/>(expand 66,048 · shrink 65,664)"]
```

`Module.parameters()` walks this tree and hands **all** of them to `AdamW`, so training improves every single one.

---

## ⏱️ What happens when you run `./run.sh train`

```mermaid
sequenceDiagram
    participant Main as TinyGPT.main
    participant Tok as BpeTokenizer
    participant Data as GPTDataset/Loader
    participant Model as GPTModel
    participant Gen as TextGenerator
    participant Tr as Trainer
    participant Adam as AdamW
    Main->>Tok: encode the story (90% / 10%)
    Tok-->>Main: 4,612 + 534 token IDs
    Main->>Data: windows of 64, batches of 8
    Main->>Model: new GPTModel(TINY): random numbers
    Main->>Gen: write a sample (nonsense)
    Main->>Tr: train(25 epochs)
    loop 225 steps
        Tr->>Data: next batch
        Tr->>Model: forward(inputs)
        Model-->>Tr: logits
        Tr->>Tr: loss + backward
        Tr->>Adam: step(), zeroGradients()
        Adam->>Model: nudge every number
    end
    Main->>Model: save("models/tinygpt.params")
    Main->>Gen: write samples (real sentences!)
```

And `./run.sh generate "..."` only does: **new model → load the file → TextGenerator**. No training needed.

---

## 🧠 The whole project in 8 sentences

1. **Tokens**: text is cut into pieces, and each piece gets a number.
2. **Training data**: the story becomes quizzes, where the answer is always the next token.
3. **Embeddings**: each number becomes a list of numbers saying *what* the token is and *where* it sits.
4. **Attention**: each token looks back at earlier tokens and mixes in what's relevant, without peeking ahead.
5. **The GPT model**: blocks of attention + thinking are stacked, with norms and shortcuts, ending in a score for every possible next token.
6. **Training**: the loss says how wrong the guesses were, gradients say how to fix each number, and AdamW makes the fix, thousands of times.
7. **Writing**: pick a next token from the scores (greedy, temperature, top-k), add it, and repeat.
8. ChatGPT is the **same idea**, just with far more numbers, far more text, and extra fine-tuning. [9. Chat with GPT-2](09-chat-with-gpt2.md) does a small version of that fine-tuning (book chapter 7).

## 📖 Doc index

| # | Doc | Book |
|---:|---|---|
| 0 | [What is an LLM?](00-what-is-an-llm.md) | 1 |
| 1 | [Tokens](01-tokenization.md) | 2.2–2.5 |
| 2 | [Training data](02-training-data.md) | 2.6 |
| 3 | [Embeddings](03-embeddings.md) | 2.1, 2.7–2.8 |
| 4 | [Attention](04-attention.md) | 3 |
| 5 | [The GPT model](05-gpt-model.md) | 4 |
| 6 | [Training](06-training.md) | 5.1–5.2, 5.4 |
| 7 | [Writing text](07-generating-text.md) | 4.7, 5.3 |
| 8 | How everything connects (this page) | 2–5 |
| 9 | [Chat with GPT-2](09-chat-with-gpt2.md) | 5.5, 7 |
| — | [Every class, explained](classes/README.md) | — |
| — | [Glossary](glossary.md) | — |
