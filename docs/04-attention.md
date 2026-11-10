# 4. Attention: words looking at other words

**Book:** Chapter 3 · **Code:** [`attention/`](../src/main/java/com/tinygpt/attention) · **Run:** `./run.sh 5` to `./run.sh 8` · **Class pictures:** [attention classes](classes/4-attention.md)

← [3. Embeddings](03-embeddings.md) · Next: [5. The GPT model →](05-gpt-model.md)

```mermaid
flowchart LR
    A["0 · What is an LLM?"] --> B["1 · Tokens"] --> C["2 · Training data"] --> D["3 · Embeddings"] --> E["4 · Attention"] --> F["5 · GPT model"] --> G["6 · Training"] --> H["7 · Writing text"]
    classDef here fill:#ffd54f,stroke:#e65100,stroke-width:3px,color:#000
    class E here
```

---

## 🤔 The problem: a word alone isn't enough

After embedding, every token has its own vector, but each one knows **only about itself**.
Yet the meaning of a word depends on its neighbours:

- *"I sat on the river **bank**"* 🏞️ vs *"I put money in the **bank**"* 🏦
- *"The animal didn't cross the street because **it** was tired"*: what is "it"?

The book's example is translation (book figure 3.3). Translating German word by word gives nonsense, because words need information from **other** words, sometimes far away:

```mermaid
flowchart TB
    G["🇩🇪 Kannst du mir helfen diesen Satz zu übersetzen"]
    G -->|"word by word ❌"| W["Can you me help this sentence to translate"]
    G -->|"with context ✅"| R["Can you help me translate this sentence"]
```

### How older models tried (and struggled)

Before transformers, *recurrent networks* (RNNs) read the sentence one word at a time and squeezed **everything** into a single memory.
For long sentences, the start gets forgotten: like trying to remember a whole page and then answering questions without looking back.

```mermaid
flowchart LR
    w1["Kannst"] --> m1(("memory"))
    m1 --> w2["du"] --> m2(("memory"))
    m2 --> w3["mir"] --> m3(("memory"))
    m3 --> dots["..."] --> mN(("one memory for<br/>the whole sentence 😰"))
```

## 💡 The idea: let every word look back at every word

**Attention** lets each word look directly at all the (earlier) words and decide **how much each one matters right now**.
Nothing has to be squeezed into one memory; the model can always "look back at the page".

> 🖍️ It's like reading with a highlighter: for each word, you highlight the other words that help explain it.

We build attention in **four small steps**, exactly like the book:

```mermaid
flowchart LR
    A["Step 5 · Simple<br/>compare & mix<br/>(no learning)"] -->|"+ learnable<br/>Q, K, V"| B["Step 6 · Self-attention"]
    B -->|"+ hide the<br/>future"| C["Step 7 · Causal"]
    C -->|"+ several<br/>at once"| D["Step 8 · Multi-head<br/>⭐ used in GPT"]
```

---

## 🔍 Step 5 · Simple attention, no learning (book 3.3)

The book's sentence: **"Your journey starts with one step"**, each word with 3 numbers:

| Word | Numbers |
|---|---|
| Your | `[0.43, 0.15, 0.89]` |
| **journey** | `[0.55, 0.87, 0.66]` |
| starts | `[0.57, 0.85, 0.64]` |
| with | `[0.22, 0.58, 0.33]` |
| one | `[0.77, 0.25, 0.10]` |
| step | `[0.05, 0.80, 0.55]` |

Let's work out the new vector for **"journey"** in three moves.

### Move 1: how much does "journey" match each word? (dot product)

A **dot product** = multiply matching numbers, then add them up. Lists that point the same way give a big number.

```
journey · starts = 0.55×0.57 + 0.87×0.85 + 0.66×0.64
                 = 0.31     + 0.74     + 0.42      = 1.48
```

### Move 2: turn the scores into percentages (softmax)

**Softmax** makes every score positive and makes them add up to 100%. Bigger scores get a bigger slice of the pie.

### Move 3: mix

The new "journey" = each word's vector × its percentage, all added up.

```mermaid
flowchart TD
    subgraph M1["Move 1 · match scores for 'journey'"]
        direction LR
        a1["Your<br/>0.95"] ~~~ a2["journey<br/>1.50"] ~~~ a3["starts<br/>1.48"] ~~~ a4["with<br/>0.84"] ~~~ a5["one<br/>0.71"] ~~~ a6["step<br/>1.09"]
    end
    subgraph M2["Move 2 · softmax → percentages"]
        direction LR
        b1["Your<br/>14%"] ~~~ b2["journey<br/>24%"] ~~~ b3["starts<br/>23%"] ~~~ b4["with<br/>12%"] ~~~ b5["one<br/>11%"] ~~~ b6["step<br/>16%"]
    end
    subgraph M3["Move 3 · mix"]
        c["new 'journey' = 14%·Your + 24%·journey + 23%·starts + 12%·with + 11%·one + 16%·step<br/>= [0.44, 0.65, 0.57]"]
    end
    M1 --> M2 --> M3
```

```mermaid
pie title How much of each word goes into the new 'journey'
    "Your" : 13.9
    "journey" : 23.8
    "starts" : 23.3
    "with" : 12.4
    "one" : 10.8
    "step" : 15.8
```

The result is called a **context vector**: "journey" mixed with a bit of its context.

### All six words at once

Doing this word by word needs loops. Grids of numbers (matrices) do it all at once (book 3.3.2):

```mermaid
flowchart LR
    X["X<br/>6 words × 3"] -->|"X × Xᵀ"| S["scores<br/>6 × 6<br/>(every word vs every word)"]
    S -->|"softmax each row"| W["weights<br/>6 × 6<br/>(each row = 100%)"]
    W -->|"× X"| Z["context vectors<br/>6 words × 3"]
```

**What's missing?** The model can't learn anything here: "similar" just means "the numbers look alike". Step 6 fixes that.

---

## 🔍 Step 6 · Self-attention with Query, Key and Value (book 3.4)

### The library idea

Each word makes **three versions** of itself, each by multiplying with its own learnable weight grid:

> 📚 **Query (Q)** 🔍: the question you bring to the library: *"I need books about animals."*
> **Key (K)** 🏷️: the label on each book's spine: *"Animals", "Cooking", "Space"...*
> **Value (V)** 📖: what's actually inside each book.
>
> Compare your **question** with every **label** → decide how much to read from each book → take that much of each book's **content**.

**Why three different versions?** What a word *is looking for* is different from what it *offers*, and from what it *hands over*.
For "it" in "the animal ... it was tired": its Query asks "who am I?", and "animal"'s Key says "I'm a noun, a thing that can be tired".

```mermaid
flowchart LR
    X["each word<br/>(3 numbers)"] -->|"× Wq"| Q["🔍 query<br/>(2 numbers)"]
    X -->|"× Wk"| K["🏷️ key<br/>(2 numbers)"]
    X -->|"× Wv"| V["📖 value<br/>(2 numbers)"]
    Q --> SC["scores = Q × Kᵀ<br/>(compare every question<br/>with every label)"]
    K --> SC
    SC -->|"÷ √2 then softmax"| WT["weights<br/>(how much to read)"]
    WT --> CTX["context = weights × V"]
    V --> CTX
```

### Why divide by √(size)? (the "scaled" part)

With many numbers per word, the scores get big. Then softmax gives almost 100% to one word and ~0% to the rest, and learning gets stuck.
Dividing by the square root of the size keeps the scores calm.

```
scores [1, 0] → softmax → 73% / 27%     (calm: both words still get attention)
scores [8, 0] → softmax → 99.97% / 0.03% (too extreme: like one word shouting)
```

### Real result (book's exact weights)

`./run.sh 6` uses the same `Wq`, `Wk`, `Wv` numbers as the book, so you get the book's output:

```
Context vectors (book: [[0.2996, 0.8053], [0.3061, 0.8210], ...]):
[[0.2996, 0.8053],
 [0.3061, 0.821 ],
 [0.3058, 0.8203], ...
```

**What's new:** `Wq`, `Wk`, `Wv` are **learned**, so training decides what counts as a good match. Each is just a `Linear` layer.

---

## 🔍 Step 7 · Causal attention: no peeking! (book 3.5)

### The problem: cheating

GPT learns by guessing the next word. If each word could look at the words **after** it, it would just copy the answer:

```mermaid
flowchart LR
    Q["guess the word after 'Your'"] -->|"without a mask"| C["👀 'journey' is right there!<br/>copy it ❌ (learns nothing)"]
    Q -->|"with a mask"| G["🤔 must really guess<br/>from 'Your' alone ✅"]
```

### The fix: a mask that hides the future

Each word may only look at **itself and the words before it**:

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

**The trick (book 3.5.1):** before softmax, set each ❌ score to **minus infinity** (−∞).
Softmax turns −∞ into exactly **0%**, and the ✅ percentages still add up to 100%.

```mermaid
flowchart LR
    S["scores for 'journey'<br/>[0.95, 1.50, 1.48, 0.84, 0.71, 1.09]"] -->|"+ mask<br/>[0, 0, −∞, −∞, −∞, −∞]"| M["[0.95, 1.50, −∞, −∞, −∞, −∞]"]
    M -->|"softmax"| W["[37%, 63%, 0%, 0%, 0%, 0%]"]
```

Real weights from `./run.sh 7`: everything above the diagonal is 0, and every row adds up to 1:

```
[[1.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
 [0.368 , 0.632 , 0.    , 0.    , 0.    , 0.    ],
 [0.2284, 0.3893, 0.3822, 0.    , 0.    , 0.    ],
 ...
```

### ⭐ The mask is what makes training efficient

Remember from [2. Training data](02-training-data.md) that one window of 64 tokens holds **64 quizzes**. The mask makes this possible:
position 10 can only see positions 0–10, so it can be asked to guess token 11 **at the same time** as every other position guesses its own next token.
No cheating, and **64 quizzes in one pass**.

### Plus dropout (book 3.5.2)

While training, we also randomly switch off some attention weights, so the model doesn't rely too much on any single connection:

```
Dropout(0.5) on a grid of ones:   [0, 2, 0, 2, 0, 2]   ← about half → 0, the rest × 2
```

> ⚽ Like football practice where random players sit out. Everyone has to learn to play well.

Dropout is switched **off** when the model writes text.

---

## 🔍 Step 8 · Multi-head attention (book 3.6)

### The idea: several detectives at once

One attention can focus on one kind of relationship at a time. **Multi-head attention** runs several side by side, so each **head** can learn to look for something different.

```mermaid
flowchart TD
    X["the words"] --> H1["🕵️ head 1<br/>e.g. who is the subject?"]
    X --> H2["🕵️ head 2<br/>e.g. the last adjective?"]
    X --> H3["🕵️ head 3<br/>e.g. a matching quote mark?"]
    X --> H4["🕵️ head 4<br/>..."]
    H1 & H2 & H3 & H4 --> J["🔗 glue the findings together"]
    J --> O["Linear: mix the findings"]
```

*(The jobs above are just examples. Nobody assigns them; each head finds its own job during training.)*

### Two ways to build it (book 3.6.1 vs 3.6.2)

```mermaid
flowchart LR
    subgraph A["Simple way: several separate modules"]
        direction TB
        a1["CausalAttention #1"]
        a2["CausalAttention #2"]
        a3["... stack the outputs"]
    end
    subgraph B["Fast way ⭐ (ours): one big module, then split"]
        direction TB
        b1["one big Q, K, V"] --> b2["✂️ cut into heads"] --> b3["attention in all heads at once"] --> b4["🔗 glue back"]
    end
    A -->|"same result, but faster"| B
```

Splitting a word's numbers between heads:

```mermaid
flowchart LR
    W["one word: 128 numbers"] --> S{{"✂️ split into 4 heads"}}
    S --> h1["head 1: numbers 1-32"]
    S --> h2["head 2: numbers 33-64"]
    S --> h3["head 3: numbers 65-96"]
    S --> h4["head 4: numbers 97-128"]
```

| | TinyGPT | GPT-2 small |
|---|---:|---:|
| numbers per word | 128 | 768 |
| heads | 4 | 12 |
| numbers per head | 32 | 64 |
| learnable numbers per attention | 65,664 | 2,360,064 |

---

## 🏭 Watch the data move through `MultiHeadAttention.forward()`

With TinyGPT sizes, during training (8 windows of 64 tokens):

```mermaid
flowchart TD
    X["x<br/>[8, 64, 128]<br/>8 windows · 64 tokens · 128 numbers"] -->|"query/key/value.forward(x)<br/>(3 Linear layers)"| QKV["Q, K, V<br/>each [8, 64, 128]"]
    QKV -->|"splitHeads():<br/>reshape + transpose"| SH["Q, K, V<br/>each [8, 4, 64, 32]<br/>8 windows · 4 heads · 64 tokens · 32 numbers"]
    SH -->|"queries.matMul(keysᵀ)"| SC["scores<br/>[8, 4, 64, 64]<br/>every token vs every token, per head"]
    SC -->|"+ causalMask(64)"| MK["future scores = −∞"]
    MK -->|"÷ √32, softmax(-1)"| WT["weights<br/>[8, 4, 64, 64]<br/>each row = 100%"]
    WT -->|"dropout"| WD["some weights → 0"]
    WD -->|"weights.matMul(values)"| CX["context<br/>[8, 4, 64, 32]"]
    CX -->|"transpose + reshape<br/>(glue the heads)"| MG["[8, 64, 128]"]
    MG -->|"outputProjection (Linear)"| OUT["output<br/>[8, 64, 128]<br/>same shape as the input!"]
```

**Same shape in and out**: attention doesn't change *how many* numbers each token has, only *what they mean*: now every token carries information from the tokens before it.

```mermaid
sequenceDiagram
    participant TB as TransformerBlock
    participant MHA as MultiHeadAttention
    participant LIN as Linear (×4)
    participant CA as CausalAttention.causalMask
    participant DR as Dropout
    TB->>MHA: forward(x)
    MHA->>LIN: query, key, value
    LIN-->>MHA: Q, K, V
    MHA->>MHA: splitHeads, Q × Kᵀ
    MHA->>CA: causalMask(64)
    CA-->>MHA: 0 / −∞ grid
    MHA->>MHA: ÷ √32, softmax
    MHA->>DR: forward(weights)
    DR-->>MHA: weights (some zeroed)
    MHA->>MHA: × V, glue heads
    MHA->>LIN: outputProjection
    LIN-->>TB: [8, 64, 128]
```

---

## 🔗 How attention connects to everything else

```mermaid
flowchart TD
    EMB["3 · Embeddings<br/>token + position vectors"] -->|"input"| MHA["👀 MultiHeadAttention"]
    LIN["Linear<br/>(Q, K, V, output)"] -->|"built from"| MHA
    DR["Dropout"] -->|"built from"| MHA
    MASK["causal mask"] -->|"makes the 64 quizzes<br/>of a window fair"| DATA["2 · Training data<br/>target = input shifted by 1"]
    MASK --> MHA
    MHA -->|"lives inside"| TB["5 · TransformerBlock<br/>(4 of them in TinyGPT)"]
    SM["softmax"] -->|"also used in"| MHA
    SM -->|"also used in"| LOSS["6 · Loss"]
    SM -->|"also used in"| GEN["7 · TextGenerator"]
```

- **Softmax** ("scores → percentages") shows up three times in the project: here, in the loss, and when choosing the next word.
- The **causal mask** is the partner of the **sliding window**: together they let one window teach 64 quizzes at once.
- In TinyGPT there are **4 multi-head attentions** (one per block), each with its own learned weights.

---

## ▶️ Try it

```bash
./run.sh 5    # simple attention: match, percentages, mix
./run.sh 6    # Query, Key, Value with the book's weights
./run.sh 7    # the mask and dropout
./run.sh 8    # multi-head
```

**Experiments**
1. In `Step05_SimpleAttention`, make "journey" exactly the same as "step". Which word does it attend to most now?
2. In `Step07_CausalAttention`, change `Dropout(0.5f)` to `Dropout(0.1f)`. How many zeros now?
3. In `Step08_MultiHeadAttention`, try 768 numbers with 10 heads. Why does it fail? (768 ÷ 10 isn't a whole number.)

## 🧠 Remember

- Attention lets each word **mix in information from other words**, weighted by how relevant they are.
- **match (dot product) → percentages (softmax) → mix (weighted sum)**.
- **Q/K/V** = question / label / content, all learned. Divide by √size to keep scores calm.
- **Causal mask** = no peeking at future words (set to −∞ → 0%).
- **Multi-head** = several attentions side by side, glued together.

## 📖 In the book

| Section | Topic | Our code |
|---|---|---|
| 3.1–3.2 | Why attention, RNN memory problem | 🤔 The problem |
| 3.3.1–3.3.2 | Simple attention, "journey" example | `SimpleAttention`, `./run.sh 5` |
| 3.4.1–3.4.2 | Trainable Wq, Wk, Wv, scaled dot-product | `SelfAttention`, `./run.sh 6` |
| 3.5.1 | Causal mask with −∞ | `CausalAttention.causalMask()`, `./run.sh 7` |
| 3.5.2 | Dropout on attention weights | `Dropout` |
| 3.5.3 | Compact causal attention class, batches | `CausalAttention` |
| 3.6.1–3.6.2 | Multi-head by stacking, then by weight splits | `MultiHeadAttention`, `./run.sh 8` |
