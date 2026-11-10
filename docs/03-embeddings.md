# 3. Embeddings: giving every token a meaning and a place

**Step 4** · Book chapter 2.7–2.8 · Code: [`nn/Embedding.java`](../src/main/java/com/tinygpt/nn/Embedding.java)

← [Training data](02-training-data.md) · Next: [Attention →](04-attention.md)

---

## Why aren't token IDs enough?

The ID `3290` ("cat") isn't "close" to `3290 + 1` in any meaningful way. IDs are just labels, like phone numbers.
The model needs something richer: a **list of numbers (a vector)** for each token, where similar words get similar lists.

```mermaid
quadrantChart
    title Imagine 2 of the numbers as a map
    x-axis "not an animal" --> "animal"
    y-axis "small" --> "big"
    quadrant-1 "big animals"
    quadrant-2 "big things"
    quadrant-3 "small things"
    quadrant-4 "small animals"
    cat: [0.8, 0.3]
    dog: [0.85, 0.4]
    elephant: [0.9, 0.9]
    mouse: [0.75, 0.1]
    truck: [0.1, 0.85]
    pencil: [0.15, 0.1]
```

Real models use hundreds of numbers per token (TinyGPT: 128, GPT-2: 768), so the "map" has hundreds of directions.
**Nobody decides these numbers by hand.** They start random, and training slowly moves similar words closer together.

---

## Token embedding: a lookup table

An embedding layer is just a table with **one row per token ID**. Looking up an ID returns its row.

```mermaid
flowchart LR
    IDS["IDs: [2, 3, 5, 1]"] --> TABLE
    subgraph TABLE["Embedding table (6 IDs × 3 numbers)"]
        direction TB
        r0["ID 0 → [ 0.34, -0.18, -0.17]"]
        r1["ID 1 → [ 0.92,  1.58,  1.30]"]
        r2["ID 2 → [ 1.28, -0.20, -0.16]"]
        r3["ID 3 → [-0.40,  0.97, -1.15]"]
        r4["ID 4 → [-1.16,  0.33, -0.63]"]
        r5["ID 5 → [-2.84, -0.78, -1.41]"]
    end
    TABLE --> OUT["4 rows out:<br/>row 2, row 3, row 5, row 1"]
```

## Position embedding: where is the word?

"dog bites man" and "man bites dog" contain the same tokens. Only the **order** changes the meaning.
Attention (next chapter) doesn't know about order by itself, so we add a second table with **one row per position**.

```mermaid
flowchart LR
    subgraph tok["Token embedding (what)"]
        A["'dog' → [0.2, 0.9, ...]"]
    end
    subgraph pos["Position embedding (where)"]
        B["position 0 → [0.1, -0.3, ...]"]
    end
    A --> PLUS(("+"))
    B --> PLUS
    PLUS --> C["model input<br/>[0.3, 0.6, ...]<br/>'dog, at the start'"]
```

## The shapes

```mermaid
flowchart LR
    A["token IDs<br/>[8, 4]<br/>8 examples × 4 tokens"] -->|"token embedding<br/>50,257 × 256 table"| B["[8, 4, 256]"]
    P["positions 0,1,2,3<br/>[4]"] -->|"position embedding<br/>4 × 256 table"| Q["[4, 256]"]
    B --> S(("+"))
    Q -->|"same for every example"| S
    S --> R["input embeddings<br/>[8, 4, 256]"]
```

---

## In the code

```java
// GPTModel.forward()
NDArray positions = tokenIds.getManager().arange(tokens);           // 0, 1, 2, ...
NDArray x = tokenEmbedding.forward(tokenIds)                          // what
              .add(positionEmbedding.forward(positions));             // + where
```

`Embedding.forward` is a single line: `weight.get(ids)`, i.e. pick those rows.

## Try it

```bash
./run.sh 4
```
