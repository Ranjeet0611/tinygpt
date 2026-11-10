# 0. What is a large language model?

**Book:** Chapter 1 · **Code:** the whole project · **Run:** `./run.sh train` to see one being made

[README](../README.md) · Next: [1. Tokens →](01-tokenization.md)

```mermaid
flowchart LR
    A["0 · What is an LLM?"] --> B["1 · Tokens"] --> C["2 · Training data"] --> D["3 · Embeddings"] --> E["4 · Attention"] --> F["5 · GPT model"] --> G["6 · Training"] --> H["7 · Writing text"]
    classDef here fill:#ffd54f,stroke:#e65100,stroke-width:3px,color:#000
    class A here
```

---

## 🤔 The big question

How can a computer write a sentence that sounds human?

Old programs used **rules written by people** ("if the word is *please*, the tone is polite"). But language has far too many rules and exceptions for anyone to write them all down.

A **large language model (LLM)** does something different. It **learns from examples**: it reads a huge amount of text and slowly discovers the patterns by itself.

---

## 💡 The idea: a game of "guess the next word"

An LLM is trained on one game only:

> **Here are some words. What word comes next?**

```mermaid
flowchart LR
    A["The cat sat on the"] --> M(("🧠 LLM"))
    M --> P["mat 41%<br/>floor 22%<br/>sofa 15%<br/>moon 0.1%<br/>..."]
```

You've played this game too. If someone says *"Once upon a..."*, you already know the next word is *"time"*.
You know it because you've heard lots of stories, and that's exactly how an LLM learns.

**Why is this game so powerful?** To guess the next word *well*, the model has to pick up grammar ("a cats" sounds wrong), facts ("Paris is the capital of..."), and even a bit of reasoning.
The book calls skills that appear without being taught directly **emergent behaviour**: the model is only taught to guess the next word, but it ends up able to translate, summarise and answer questions.

---

## 🔤 What does each word in "Large Language Model" mean?

| Word | Meaning | In TinyGPT |
|---|---|---|
| **Large** | lots of learnable numbers ("parameters") and lots of training text | 13.7 million numbers, 1 short story |
| **Language** | it works with human text | English |
| **Model** | a program whose behaviour comes from numbers learned from data, not hand-written rules | the `GPTModel` class |

```mermaid
flowchart TB
    AI["🤖 Artificial Intelligence<br/>machines doing 'clever' things"] --> ML["📈 Machine Learning<br/>learning patterns from data"]
    ML --> DL["🕸️ Deep Learning<br/>learning with many-layered networks"]
    DL --> LLM["💬 Large Language Models<br/>deep learning on text"]
    LLM --> GPT["GPT (and TinyGPT!)"]
```

---

## 🏗️ How an LLM is built: three stages (book figure 1.9)

```mermaid
flowchart LR
    subgraph S1["Stage 1 · Build it"]
        A1["prepare text<br/>(chapter 2)"] --> A2["attention<br/>(chapter 3)"] --> A3["GPT architecture<br/>(chapter 4)"]
    end
    subgraph S2["Stage 2 · Pretrain it"]
        B1["train on lots of text<br/>(chapter 5)"] --> B2["📦 foundation model<br/>can continue any text"]
    end
    subgraph S3["Stage 3 · Fine-tune it"]
        C1["spam classifier<br/>(chapter 6)"]
        C2["chat assistant<br/>(chapter 7)"]
    end
    S1 --> S2 --> S3
    classDef done fill:#c8e6c9,stroke:#2e7d32,color:#000
    class A1,A2,A3,B1,B2 done
```

**This project covers stages 1 and 2** (the green boxes): chapters 2 to 5.

| Stage | What happens | Real-life picture |
|---|---|---|
| **Pretraining** | Read huge amounts of ordinary text and play "guess the next word". No one has to label anything: the text itself contains the answers. | A child reading thousands of books |
| **Fine-tuning** | Train a bit more on a smaller, special set of examples (like questions with good answers) | The same child then training for a particular job |

**Pretraining is expensive.** GPT-3 cost millions of dollars of computer time. That's why TinyGPT trains on a single short story instead: it shows *how* it works in 20 seconds on a laptop.

---

## 🏛️ The transformer: the design behind GPT

In 2017, researchers invented the **transformer**, designed for translating between languages. It had two halves:

```mermaid
flowchart LR
    IN["'This is an example'"] --> ENC["Encoder<br/>reads and understands<br/>the whole input"]
    ENC --> DEC["Decoder<br/>writes the output<br/>one word at a time"]
    DEC --> OUT["'Das ist ein Beispiel'"]
```

| Model family | Uses | Good at |
|---|---|---|
| BERT | the **encoder** only | understanding text (e.g. spotting toxic posts) |
| **GPT** | the **decoder** only | **writing** text, one word at a time |

GPT keeps only the writing half, which makes it simpler. TinyGPT is a GPT, so it is decoder-only too.

---

## 🔁 How a GPT writes: one word at a time

```mermaid
sequenceDiagram
    participant You
    participant GPT
    You->>GPT: "This is"
    GPT-->>You: next word: "an"
    You->>GPT: "This is an"
    GPT-->>You: next word: "example"
    You->>GPT: "This is an example"
    GPT-->>You: next word: "."
```

Each new word is added to the input, and the model is asked again.
Models that feed their own output back in like this are called **autoregressive**.

---

## 🗺️ The whole journey in this project

```mermaid
flowchart TD
    T["📄 text<br/>'Every effort moves'"] -->|"1 · Tokenizer"| I["🔢 token IDs<br/>[6109, 3626, 6100]"]
    I -->|"3 · Embedding"| V["📍 one list of 128 numbers per token"]
    V -->|"4 · Attention + 5 · Transformer blocks"| C["🧠 lists that now 'understand' the context"]
    C -->|"5 · Output head"| S["📊 a score for each of 50,257 tokens"]
    S -->|"7 · Choose"| N["' you'"]
    S -.->|"6 · Training: compare with the right answer,<br/>fix the numbers, repeat"| V
    D["2 · Training data<br/>(right answers from the story)"] -.-> S
```

Each numbered box has its own doc. Read them in order. Each one builds on the last.

---

## 🧠 Remember

- An LLM learns by **guessing the next word**, over and over.
- It **learns patterns from examples** instead of following hand-written rules.
- Building one = **build → pretrain → (fine-tune)**. TinyGPT does build + pretrain.
- GPT is the **decoder** half of a transformer: it **writes one token at a time**.

## 📖 In the book

| Section | Topic | Here |
|---|---|---|
| 1.1–1.2 | What LLMs are and what they're used for | 🤔 The big question |
| 1.3 | Pretraining and fine-tuning | 🏗️ Three stages |
| 1.4, 1.6 | Transformer, encoder vs decoder, GPT | 🏛️ The transformer |
| 1.5 | Large datasets | 🏗️ Three stages |
| 1.7 | The plan for the book | 🗺️ The whole journey |
