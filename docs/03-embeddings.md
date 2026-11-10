# 3. Embeddings: giving every token a meaning and a place

**Book:** Chapter 2.1, 2.7, 2.8 · **Code:** [`nn/Embedding.java`](../src/main/java/com/tinygpt/nn/Embedding.java), [`GPTModel.forward()`](../src/main/java/com/tinygpt/model/GPTModel.java) · **Run:** `./run.sh 4` · **Class pictures:** [Embedding](classes/3-nn.md#embedding)

← [2. Training data](02-training-data.md) · Next: [4. Attention →](04-attention.md)

```mermaid
flowchart LR
    A["0 · What is an LLM?"] --> B["1 · Tokens"] --> C["2 · Training data"] --> D["3 · Embeddings"] --> E["4 · Attention"] --> F["5 · GPT model"] --> G["6 · Training"] --> H["7 · Writing text"]
    classDef here fill:#ffd54f,stroke:#e65100,stroke-width:3px,color:#000
    class D here
```

---

## 🤔 The problem: a token ID is just a name tag

The ID for " cat" is `3797` and for " dog" is `3290`. Those numbers say **nothing** about what the words mean.
They're like phone numbers: two friends living next door can have completely different phone numbers.

The model needs numbers that **carry meaning**, so that similar words look similar.

## 💡 The idea: describe each token with a list of numbers

Give every token a **list of numbers** (a *vector*). Think of it as a character card in a game:

```mermaid
flowchart LR
    subgraph CAT["🐱 ' cat'"]
        c["alive: 9<br/>size: 2<br/>machine: 0<br/>cute: 8"]
    end
    subgraph DOG["🐶 ' dog'"]
        d["alive: 9<br/>size: 3<br/>machine: 0<br/>cute: 8"]
    end
    subgraph CAR["🚗 ' car'"]
        r["alive: 0<br/>size: 6<br/>machine: 9<br/>cute: 1"]
    end
```

Now **cat and dog have almost the same card**, and car is very different. The model can *see* that cats and dogs are alike.

If we use just 2 of the numbers, we can draw the words on a map (book figure 2.3):

```mermaid
quadrantChart
    title Words placed by 2 of their numbers
    x-axis "not alive" --> "alive"
    y-axis "small" --> "big"
    quadrant-1 "big living things"
    quadrant-2 "big objects"
    quadrant-3 "small objects"
    quadrant-4 "small living things"
    cat: [0.85, 0.25]
    dog: [0.88, 0.35]
    mouse: [0.8, 0.1]
    elephant: [0.9, 0.9]
    car: [0.1, 0.65]
    truck: [0.12, 0.85]
    pencil: [0.15, 0.1]
```

**Similar words end up close together.**

### ⚠️ Important: nobody writes these numbers by hand

The labels "alive", "size" and so on are just to help us imagine. In the real model:
- every token gets **128 numbers** in TinyGPT (768 in GPT-2), with no names;
- they **start out random**;
- **training moves them** a little at a time, until tokens used in similar ways end up with similar numbers.

The book explains that older methods (like Word2Vec) learned embeddings separately. **GPT learns its embeddings as part of its own training**, so they fit exactly the job they're used for.

---

## 🔍 Part 1 · The token embedding: a lookup table (book 2.7)

An embedding layer is simply a **table with one row per token ID**. To embed an ID, you **read that row**.

The book's tiny example uses 6 token IDs and 3 numbers each. Here are the real (random) starting numbers from `./run.sh 4`:

```mermaid
flowchart LR
    IN["IDs<br/>[2, 3, 5, 1]"] --> T
    subgraph T["📇 Embedding table: 6 rows × 3 numbers"]
        direction TB
        r0["row 0: [ 0.34, -0.18, -0.17]"]
        r1["row 1: [ 0.92,  1.58,  1.30]"]
        r2["row 2: [ 1.28, -0.20, -0.16]"]
        r3["row 3: [-0.40,  0.97, -1.15]"]
        r4["row 4: [-1.16,  0.33, -0.63]"]
        r5["row 5: [-2.84, -0.78, -1.41]"]
    end
    T --> OUT["4 rows out:<br/>row 2 → [ 1.28, -0.20, -0.16]<br/>row 3 → [-0.40,  0.97, -1.15]<br/>row 5 → [-2.84, -0.78, -1.41]<br/>row 1 → [ 0.92,  1.58,  1.30]"]
```

4 IDs in → 4 rows out. **Shape `[4]` became `[4, 3]`**: each ID was swapped for its 3 numbers.

### A lookup is the same as a special multiplication

The book points out that "pick row 2" gives exactly the same result as multiplying a **one-hot** list by the table:

```
one-hot for ID 2 (6 IDs):  [0, 0, 1, 0, 0, 0]
× table                 →  0·row0 + 0·row1 + 1·row2 + 0·row3 + 0·row4 + 0·row5  =  row 2
```

So an embedding is really a `Linear` layer with a shortcut: instead of multiplying by lots of zeros, it just reads the row.
That's why it can **learn the same way** every other layer learns.

---

## 🔍 Part 2 · The position embedding (book 2.8)

### The problem: attention can't see order

In the next chapter, attention compares every token with every other token. But it treats the tokens like **a bag of words**, so the order is lost:

```mermaid
flowchart LR
    A["dog bites man 🐕➡️👨"] --> BAG["🛍️ {dog, bites, man}"]
    B["man bites dog 👨➡️🐕"] --> BAG
    BAG --> Q["😕 same bag,<br/>but very different news!"]
```

### The fix: a second table, for positions

Make a second table with **one row per position**: 0, 1, 2, ... up to the longest text the model can read (`contextLength`).
Then **add** the position row to the token row.

```mermaid
flowchart LR
    subgraph W["what is it?"]
        T1["' dog' token row<br/>[0.5, 1.2, -0.3]"]
    end
    subgraph P["where is it?"]
        P1["position 0 row<br/>[0.1, -0.2, 0.4]"]
    end
    T1 --> ADD(("➕"))
    P1 --> ADD
    ADD --> R["'dog, at the start'<br/>[0.6, 1.0, 0.1]"]
```

Now `dog` at position 0 and `dog` at position 2 get **different** final numbers, so the model can tell `dog bites man` from `man bites dog`.

| | Token embedding | Position embedding |
|---|---|---|
| Answers | **what** is this token? | **where** is it? |
| Rows | one per token: 50,257 | one per position: 64 (TinyGPT), 1,024 (GPT-2) |
| Looked up by | the token ID | the position number 0, 1, 2, ... |
| Learned? | yes | yes (GPT uses *absolute*, learned positions) |

---

## 🏭 Watch the data move through the code

Inside [`GPTModel.forward()`](../src/main/java/com/tinygpt/model/GPTModel.java), for one training batch:

```mermaid
flowchart TD
    IDS["token IDs<br/>NDArray [8, 64]<br/>8 windows × 64 IDs"] -->|"tokenEmbedding.forward(ids)<br/>= weight.get(ids)"| TE["[8, 64, 128]<br/>each ID → its 128-number row"]
    POS["positions = arange(64)<br/>[0, 1, 2, ..., 63]"] -->|"positionEmbedding.forward(positions)"| PE["[64, 128]<br/>one row per position"]
    TE --> ADD(("➕ add"))
    PE -->|"same 64 position rows are<br/>added to each of the 8 windows"| ADD
    ADD --> X["[8, 64, 128]<br/>'input embeddings'"]
    X -->|"dropout.forward(x)"| OUT["→ into the first TransformerBlock"]
```

**How can `[8, 64, 128]` + `[64, 128]` work when the shapes are different?**
The smaller one is **reused for every window in the batch** (this is called *broadcasting*). Position 5 is position 5 in every window.

Real shapes printed by `./run.sh 4` (the book's example sizes: batch 8, 4 tokens, 256 numbers):

```
A batch of token IDs:              (8, 4)
Token embeddings  (what):          (8, 4, 256)
Position embeddings (where):       (4, 256)
Token + position = model input:    (8, 4, 256)
```

```java
// GPTModel.forward(): the embedding part
NDArray positions = tokenIds.getManager().arange(tokens);             // 0, 1, 2, ...
NDArray x = tokenEmbedding.forward(tokenIds)                            // what
              .add(positionEmbedding.forward(positions));               // + where
x = dropout.forward(x);
```

---

## 🔗 Why embeddings matter, and where they're used

### Where the `Embedding` class is used

```mermaid
flowchart TD
    EC["🗂️ Embedding class"]
    EC -->|"used as"| TE["tokenEmbedding<br/>50,257 × 128<br/>in GPTModel"]
    EC -->|"used as"| PE["positionEmbedding<br/>64 × 128<br/>in GPTModel"]
    EC -->|"used in demos"| S4["Step04_Embeddings"]
```

It's used **twice** in the model, both at the very start. Every other part works on the vectors these two tables produce.

### The embedding size (128) travels through the whole model

```mermaid
flowchart LR
    E["Embedding<br/>→ 128"] --> A["Attention<br/>128 → 128"]
    A --> N["LayerNorm<br/>over 128"]
    N --> F["FeedForward<br/>128 → 512 → 128"]
    F --> MORE["... 4 blocks ..."]
    MORE --> H["Output head<br/>128 → 50,257"]
```

`embeddingDim` (128) is the "width" of the whole model: **every layer reads and writes 128 numbers per token**.
That's why it's in `GPTConfig`. Change it once, and every layer changes size with it.

### The output head is the embedding in reverse

```mermaid
flowchart LR
    ID["token ID"] -->|"token Embedding<br/>ID → 128 numbers"| V["128 numbers"]
    V2["128 numbers<br/>(after all the blocks)"] -->|"output head (Linear)<br/>128 numbers → score per ID"| S["50,257 scores"]
```

At the start, an ID becomes a vector. At the end, a vector becomes scores for every ID.
OpenAI's GPT-2 even **shares one table** for both jobs (called *weight tying*). The book (and TinyGPT) keep them separate, which is why the book's model has 163M numbers instead of 124M.

### How embeddings learn

During training, only the rows for **tokens that appeared in the batch** get a gradient (a "which way to move" signal).
A token that never appears in the training text never learns anything useful. Its row only slowly shrinks towards 0 (AdamW's *weight decay*; see [6. Training](06-training.md)).
Our story uses only 1,416 of the 50,257 tokens, which is another reason real models need huge, varied training text.

### Embeddings are the biggest part of TinyGPT

```mermaid
pie title Where TinyGPT's 13.7 million numbers live
    "Token embedding (50,257 × 128)" : 6432896
    "Output head (128 × 50,257)" : 6432896
    "4 transformer blocks" : 791552
    "Position embedding (64 × 128)" : 8192
    "Final LayerNorm" : 256
```

Because our vocabulary is huge (50,257) and our model is narrow (128), **94% of the numbers are in the two vocabulary tables**.

---

## ▶️ Try it

```bash
./run.sh 4
```

**Experiments** (in `Step04_Embeddings`)
1. Look up the same ID twice: `new int[]{2, 2}`. Are the two rows identical? (Yes: same ID, same row.)
2. Change `outputDim` from 256 to 8. Which shapes change?
3. Run it twice. Do the random numbers change? (No: the random seed 123 makes them repeatable.)

## 🧠 Remember

- A token ID is just a name tag; an **embedding** is a list of numbers that carries meaning.
- An embedding layer is a **lookup table**: ID → row. The rows **start random and are learned**.
- **Token embedding** = *what*; **position embedding** = *where*. They're **added** together.
- `embeddingDim` is the width of the entire model; the output head turns vectors back into token scores.

## 📖 In the book

| Section | Topic | Our code |
|---|---|---|
| 2.1 | What embeddings are, Word2Vec, 2-D picture | 💡 The idea |
| 2.7 | `torch.nn.Embedding(6, 3)`, lookup of `[2, 3, 5, 1]` | `Embedding`, `Step04_Embeddings` part 1 |
| 2.8 | Absolute position embeddings, shapes `(8, 4, 256)` | `GPTModel.forward()`, `Step04_Embeddings` part 2 |
| 4.6 | Weight tying in GPT-2 | 🔗 The output head is the embedding in reverse |
