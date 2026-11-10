# 4. Attention: words looking at other words

**Steps 5–8** · Book chapter 3 · Code: [`attention/`](../src/main/java/com/tinygpt/attention)

← [Embeddings](03-embeddings.md) · Next: [The GPT model →](05-gpt-model.md)

---

## Why?

Take *"The **bank** of the river"* and *"I put money in the **bank**"*. Same word, different meaning.
To understand a word, you have to look at the words around it.
**Attention** lets every word look at the other words and collect what it needs from them.

We build it in four small steps, each adding one idea:

```mermaid
flowchart LR
    A["Step 5<br/><b>Simple</b><br/>compare words"] --> B["Step 6<br/><b>Self-attention</b><br/>+ learnable Q, K, V"]
    B --> C["Step 7<br/><b>Causal</b><br/>+ no peeking ahead"]
    C --> D["Step 8<br/><b>Multi-head</b><br/>+ many at once"]
    D --> G(("used in<br/>GPT"))
```

---

## Step 5: simple attention (no learning)

The example sentence: **"Your journey starts with one step"**, each word with 3 numbers.

```mermaid
flowchart TD
    subgraph s1["1 · Scores: how similar is 'journey' to each word? (dot product)"]
        direction LR
        y1["Your<br/>0.95"] ~~~ y2["journey<br/>1.50"] ~~~ y3["starts<br/>1.48"] ~~~ y4["with<br/>0.84"] ~~~ y5["one<br/>0.71"] ~~~ y6["step<br/>1.09"]
    end
    subgraph s2["2 · Softmax: turn scores into percentages (sum = 100%)"]
        direction LR
        w1["Your<br/>14%"] ~~~ w2["journey<br/>24%"] ~~~ w3["starts<br/>23%"] ~~~ w4["with<br/>12%"] ~~~ w5["one<br/>11%"] ~~~ w6["step<br/>16%"]
    end
    subgraph s3["3 · Mix: new 'journey' = 14%·Your + 24%·journey + 23%·starts + ..."]
        c["context vector for 'journey'<br/>[0.44, 0.65, 0.57]"]
    end
    s1 --> s2 --> s3
```

**Dot product**: multiply matching numbers and add them up. Similar lists give a big result.
**Softmax**: makes every number positive and makes them add up to 1, so they become percentages.

> Like reading a sentence with a highlighter: for each word, you highlight the other words that help explain it.

Doing it for all 6 words at once is just matrix multiplication: `weights = softmax(X × Xᵀ)`, `context = weights × X`.

---

## Step 6: self-attention with Query, Key, Value

In step 5, "similar" was fixed. Now we let the model **learn** what to look for, using three learnable weight matrices.

> **The library analogy**
> - **Query (Q)**: the question you bring: *"I'm looking for books about space."*
> - **Key (K)**: the label on each book's spine: *"Astronomy", "Cooking", ...*
> - **Value (V)**: what's inside the book, the thing you actually take home.
>
> Compare your question with every label (Q · K) → decide how much to read from each book → take that much of each content (V).

```mermaid
flowchart LR
    X["input<br/>6 words × 3"] -->|"× Wq"| Q["Queries<br/>6 × 2"]
    X -->|"× Wk"| K["Keys<br/>6 × 2"]
    X -->|"× Wv"| V["Values<br/>6 × 2"]
    Q --> S["Scores = Q × Kᵀ<br/>6 × 6"]
    K --> S
    S -->|"÷ √2, softmax"| W["Weights<br/>6 × 6"]
    W --> C["Context = Weights × V<br/>6 × 2"]
    V --> C
```

**Why divide by √(dimension)?** With many numbers, dot products get big, and softmax then puts nearly 100% on one word.
Dividing keeps the scores calm, so the model can still spread its attention while learning.

Step 6 uses the book's exact weights, so you get the book's output: `[[0.2996, 0.8053], [0.3061, 0.8210], ...]`.

---

## Step 7: causal attention (no peeking!)

When the model learns to guess the next word, it must not see that word, or it would just copy the answer.
So each word may only look at itself and the words **before** it.

```mermaid
block-beta
    columns 7
    h[" "] c1["Your"] c2["journey"] c3["starts"] c4["with"] c5["one"] c6["step"]
    r1["Your"] a11["✅"] a12["❌"] a13["❌"] a14["❌"] a15["❌"] a16["❌"]
    r2["journey"] a21["✅"] a22["✅"] a23["❌"] a24["❌"] a25["❌"] a26["❌"]
    r3["starts"] a31["✅"] a32["✅"] a33["✅"] a34["❌"] a35["❌"] a36["❌"]
    r4["with"] a41["✅"] a42["✅"] a43["✅"] a44["✅"] a45["❌"] a46["❌"]
    r5["one"] a51["✅"] a52["✅"] a53["✅"] a54["✅"] a55["✅"] a56["❌"]
    r6["step"] a61["✅"] a62["✅"] a63["✅"] a64["✅"] a65["✅"] a66["✅"]
```

**The trick:** before softmax, set every ❌ score to **−∞** (minus infinity). Softmax turns `e^(−∞)` into exactly **0%**,
and the remaining percentages still add up to 100%.

```mermaid
flowchart LR
    A["scores<br/>[1.2, 0.8, 1.5]"] -->|"+ mask<br/>[0, −∞, −∞]"| B["[1.2, −∞, −∞]"]
    B -->|"softmax"| C["[100%, 0%, 0%]"]
```

### Dropout

During training we also randomly hide some attention weights (**dropout**), so the model can't depend too much on any one connection.

> Like a football team practising with random players sitting out: everyone has to learn to play well.

```
Dropout(0.5) on all ones:     [0, 2, 0, 2, 0, 2]   ← half set to 0, the rest doubled to keep the total similar
```

Dropout is switched **off** when we use the model to write text.

---

## Step 8: multi-head attention

One attention can only look for one kind of thing at a time. **Multi-head attention** runs several in parallel,
and each "head" can learn to notice something different.

```mermaid
flowchart TD
    X["input [batch, tokens, 768]"] --> QKV["Q, K, V projections"]
    QKV --> SPLIT{{"split into 12 heads<br/>of 64 numbers each"}}
    SPLIT --> H1["head 1<br/><i>who is the subject?</i>"]
    SPLIT --> H2["head 2<br/><i>last adjective?</i>"]
    SPLIT --> H3["head 3<br/><i>matching quote mark?</i>"]
    SPLIT --> HN["... head 12"]
    H1 & H2 & H3 & HN --> CAT["glue heads back together<br/>[batch, tokens, 768]"]
    CAT --> OUT["output Linear<br/>(mixes what the heads found)"]
```

We don't really build 12 separate modules. We do one big Q/K/V, **reshape** it into heads, run causal attention in all heads at the same time,
then reshape back:

```
[batch, tokens, 768] → reshape → [batch, tokens, 12, 64] → transpose → [batch, 12, tokens, 64]
                                  … attention in every head …
[batch, 12, tokens, 64] → transpose → [batch, tokens, 12, 64] → reshape → [batch, tokens, 768]
```

(The heads' jobs above are just examples. In practice we don't choose them; they come out of training.)

---

## In the code

| Step | Class | Main idea |
|---|---|---|
| 5 | [`SimpleAttention`](../src/main/java/com/tinygpt/attention/SimpleAttention.java) | `softmax(X·Xᵀ)·X` |
| 6 | [`SelfAttention`](../src/main/java/com/tinygpt/attention/SelfAttention.java) | Q, K, V and `÷√d` |
| 7 | [`CausalAttention`](../src/main/java/com/tinygpt/attention/CausalAttention.java) | `causalMask()` + dropout |
| 8 | [`MultiHeadAttention`](../src/main/java/com/tinygpt/attention/MultiHeadAttention.java) | `splitHeads()` + output projection |

## Try it

```bash
./run.sh 5
./run.sh 6
./run.sh 7
./run.sh 8
```

**Experiment:** in `Step05_SimpleAttention`, change the numbers for "journey" to match "step" exactly. Which word does "journey" pay most attention to now?
