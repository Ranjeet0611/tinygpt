# 7. Writing text: choosing the next word

**Book:** Chapter 4.7, 5.3 · **Code:** [`generation/TextGenerator.java`](../src/main/java/com/tinygpt/generation/TextGenerator.java) · **Run:** `./run.sh 11`, `./run.sh generate "..."` · **Class pictures:** [TextGenerator](classes/7-generation.md)

← [6. Training](06-training.md) · Next: [8. How everything connects →](08-how-everything-connects.md)

```mermaid
flowchart LR
    A["0 · What is an LLM?"] --> B["1 · Tokens"] --> C["2 · Training data"] --> D["3 · Embeddings"] --> E["4 · Attention"] --> F["5 · GPT model"] --> G["6 · Training"] --> H["7 · Writing text"]
    classDef here fill:#ffd54f,stroke:#e65100,stroke-width:3px,color:#000
    class H here
```

---

## 🤔 The problem

The model only ever gives **scores for the next token**. It never writes a whole sentence by itself.
So two questions:
1. How do we get **many** words out of it?
2. Given 50,257 scores, **which** token do we pick?

## 💡 Part 1 · One token at a time (book 4.7)

Ask for the next token, **add it to the text**, and ask again. Repeat as many times as you like.

```mermaid
sequenceDiagram
    participant You
    participant TG as TextGenerator
    participant M as GPTModel
    You->>TG: generate("Every effort moves", 3 tokens)
    TG->>M: [6109, 3626, 6100]
    M-->>TG: scores → pick " you" (345)
    TG->>M: [6109, 3626, 6100, 345]
    M-->>TG: scores → pick " know" 
    TG->>M: [6109, 3626, 6100, 345, 760]
    M-->>TG: scores → pick ","
    TG-->>You: "Every effort moves you know,"
```

### Only the last position matters here

The model returns scores for **every** position (useful in training, where every position is a quiz). When writing, we only need the scores after the **last** token:

```mermaid
flowchart LR
    I["'Every effort moves'<br/>3 tokens"] --> M["model"] --> O["scores [1, 3, 50,257]"]
    O -->|"position 0: after 'Every'"| X1["ignored"]
    O -->|"position 1: after 'Every effort'"| X2["ignored"]
    O -->|"position 2: after 'Every effort moves'"| K["✅ use these 50,257 scores"]
```

### The model can only read so much

TinyGPT has 64 position rows, so it can read at most **64 tokens**. If the text gets longer, we feed in only the **most recent 64**:

```mermaid
flowchart LR
    T["text so far: 80 tokens"] -->|"keep the last 64"| W["tokens 17-80"] --> M["model"]
```

> 📜 Like reading through a window that only shows the last few lines of the page.

---

## 💡 Part 2 · Choosing the next token (book 5.3)

The book's example: after *"every effort moves you"*, a tiny 9-word vocabulary gets these scores:

| closer | every | effort | **forward** | inches | moves | pizza | toward | you |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.51 | 0.89 | −1.90 | **6.75** | 1.63 | −1.62 | −1.89 | 6.28 | 1.79 |

### Strategy 1 · Greedy: always the top score

Always **"forward"**.

| 👍 | 👎 |
|---|---|
| simple, predictable | boring; often gets stuck repeating the same words |

Our training samples use greedy, which is why early ones look like `, the, the, the`.

### Strategy 2 · Sampling with temperature

**Roll a weighted die**: turn the scores into percentages (softmax) and pick randomly, so likelier tokens are picked more often.
**Temperature** controls how adventurous the die is. It divides the scores before softmax:

```mermaid
flowchart LR
    S["scores"] -->|"÷ temperature"| D["adjusted scores"] -->|"softmax"| P["percentages"] -->|"🎲 roll"| W["next token"]
```

A tiny example with two tokens, scores `2` and `1`:

| Temperature | Scores become | Gap | Chances | Feels like |
|---:|---|---:|---|---|
| 0.5 | `4`, `2` | 2 (bigger) | 88% / 12% | careful |
| 1 | `2`, `1` | 1 | 73% / 27% | normal |
| 2 | `1`, `0.5` | 0.5 (smaller) | 62% / 38% | adventurous |

**Small temperature → bigger gaps → the favourite almost always wins.**
**Big temperature → smaller gaps → underdogs win more often.**

Picking 1,000 times with the book's scores (real output of `./run.sh 11`):

```mermaid
xychart-beta
    title "Temperature 0.1 (careful): 'forward' almost every time"
    x-axis [closer, every, effort, forward, inches, moves, pizza, toward, you]
    y-axis "times picked out of 1,000" 0 --> 1000
    bar [0, 0, 0, 992, 0, 0, 0, 8, 0]
```

```mermaid
xychart-beta
    title "Temperature 1 (normal)"
    x-axis [closer, every, effort, forward, inches, moves, pizza, toward, you]
    y-axis "times picked out of 1,000" 0 --> 1000
    bar [69, 3, 0, 563, 5, 0, 0, 357, 3]
```

```mermaid
xychart-beta
    title "Temperature 5 (wild): even 'pizza' gets picked 55 times"
    x-axis [closer, every, effort, forward, inches, moves, pizza, toward, you]
    y-axis "times picked out of 1,000" 0 --> 1000
    bar [176, 81, 42, 223, 79, 44, 55, 222, 78]
```

The book's run at temperature 1 got forward 582, toward 343, closer 73: almost the same as ours (forward 563, toward 357, closer 69). Different random dice, same pattern.

> ☕ Temperature is like how many cups of coffee the writer has had. Low: careful and predictable. High: creative, but might write "every effort moves you pizza".

### Strategy 3 · Top-k: only the best k may be picked

Before rolling the die, **throw away everything except the k highest scores** (set the rest to −∞, which becomes 0%).

```mermaid
flowchart LR
    A["all 9 scores"] -->|"keep the top 3"| B["forward 6.75<br/>toward 6.28<br/>closer 4.51<br/>(everything else → −∞)"]
    B -->|"softmax"| C["forward 58%<br/>toward 36%<br/>closer 6%<br/>pizza 0%"]
    C -->|"🎲 roll"| D["next token"]
```

With top-k = 3 (from `./run.sh 11`): forward 570, toward 359, closer 71, and **never** "pizza".

### Combining them

```mermaid
flowchart LR
    S["50,257 scores"] -->|"1 · top-k: keep the 25 best"| K["25 scores"]
    K -->|"2 · ÷ temperature"| T["adjusted"]
    T -->|"3 · softmax"| P["percentages"]
    P -->|"4 · 🎲 roll"| N["next token"]
```

**Top-k removes silly choices; temperature controls variety among the sensible ones.** That's what `./run.sh train` uses for its second sample:

```
Greedy:                    Every effort moves you know; and I had the craft was vindicated by little Claude Nutley...
Temperature 1, top-k 25:   Every effort moves you knowagged him down." I was taken with the factrec me in a self-conf...
```

---

## 🏭 Watch the data move through `TextGenerator.generate()`

```mermaid
flowchart TD
    P["String prompt<br/>'Every effort moves'"] -->|"tokenizer.encode"| I["int[] [6109, 3626, 6100]"]
    I -->|"keep the last ≤ 64"| W["int[] window"]
    W -->|"manager.create(new int[][]{window})"| N["NDArray [1, 3]"]
    N -->|"model.forward<br/>(training mode OFF: no dropout)"| L["logits [1, 3, 50,257]"]
    L -->|"logits.get(0, last).toFloatArray()"| F["float[50,257]<br/>plain Java from here on"]
    F --> Q{"temperature 0?"}
    Q -->|"yes"| G["argMax(scores)"]
    Q -->|"no"| K["keepTopK → softmax(temperature) → sample"]
    G --> NX["next token ID, e.g. 345"]
    K --> NX
    NX -->|"append to int[]"| I
    I -->|"after maxNewTokens:<br/>tokenizer.decode"| OUT["String 'Every effort moves you know, ...'"]
```

Choosing the token is written in **plain Java loops** (no tensors), so it's the easiest class to read:

| Method | What it does |
|---|---|
| `generate(prompt, maxNewTokens, temperature, topK)` | the loop above |
| `nextTokenScores(ids)` | run the model, return the last position's 50,257 scores |
| `argMax(scores)` | index of the biggest score (greedy) |
| `keepTopK(scores, k)` | everything except the top k → −∞ |
| `softmax(scores, temperature)` | scores → percentages that add up to 1 |
| `sample(...)` | roll the weighted die |

---

## 🔗 How writing connects to everything else

```mermaid
flowchart TD
    TOK["1 · Tokenizer"] -->|"encode the prompt / decode the result"| GEN["✍️ TextGenerator"]
    CFG["GPTConfig.contextLength = 64"] -->|"how many tokens to feed in"| GEN
    MODEL["5 · GPTModel<br/>(trained in 6, or loaded from file)"] -->|"scores"| GEN
    GEN -->|"used by"| TR["6 · Trainer<br/>prints a sample every epoch"]
    GEN -->|"used by"| MAIN["TinyGPT<br/>train and generate modes"]
    GEN -->|"used by"| S9["Step09, Step11"]
    SM["softmax"] -.->|"the same idea as in"| ATT["4 · Attention"]
    SM -.->|"and in"| LOSS["6 · Loss"]
    SM --> GEN
```

- **Writing uses no training tools at all**: no loss, no gradients, no optimizer, and dropout is switched off.
- **The same trained numbers** can be used for writing again and again; that's why we save them to a file.

---

## ▶️ Try it

```bash
./run.sh 11                              # the book's 9-word example
./run.sh train                           # train once (saves the model)
./run.sh generate "I had always"         # write with the saved model
./run.sh generate "She said"
```

**Experiments** (in `TinyGPT.showSamples`):
1. Temperature `0.5` vs `2.0`, with top-k 25. Which reads better?
2. Top-k `1`. What is that the same as? (Greedy!)
3. Top-k `0` (off) with temperature `1.5`. How strange does it get?

## 🧠 Remember

- The model writes **one token at a time**, feeding each new token back in.
- Only the **last position's** scores are used, and only the last `contextLength` tokens are fed in.
- **Greedy** = always the top token. **Temperature** = how adventurous. **Top-k** = only the k best are allowed.
- Writing needs only the **trained numbers**: no loss, no gradients, no dropout.

## 📖 In the book

| Section | Topic | Our code |
|---|---|---|
| 4.7 | `generate_text_simple`: greedy, cropping the context | `generateGreedy`, `nextTokenScores` |
| 5.1.1 | Text ↔ token ID helpers | `Tokenizer.encode` / `decode` |
| 5.3.1 | Temperature scaling, 1,000 samples | `softmax(scores, temperature)`, `./run.sh 11` |
| 5.3.2 | Top-k sampling | `keepTopK` |
| 5.3.3 | The combined `generate` function | `generate(prompt, n, temperature, topK)` |
