# TinyGPT: build a small ChatGPT-style model in Java, one step at a time

TinyGPT is a **small GPT language model written in Java**. It is built step by step while following the book
[*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka.
The book uses Python. This project does the same things in Java.

Every step is a small program you can run. Every idea has a picture.
If you know a little Java, you can follow along. No AI background needed.

```
Before training:  Every effort moves you Hero farmland tenth harness ATL subsidized Contracts fixes
After training:   Every effort moves you know; and I had the craft was vindicated by little Claude Nutley, who...
```

*(Trained for 20 seconds on a laptop, on one short story.)*

---

## What is a language model? (the 30-second version)

A language model plays one game, over and over: **guess the next word**.

```mermaid
flowchart LR
    A["Every effort moves"] --> M(("🧠 GPT"))
    M --> B["you  (62%)<br/>forward (20%)<br/>pizza (0.01%)<br/>..."]
    B --> C["Every effort moves <b>you</b>"]
    C -.->|"repeat: feed it back in"| M
```

1. It reads some text.
2. It gives every possible next word a score.
3. It picks a word, adds it to the text, and goes again.

ChatGPT does the same thing, just much bigger: it trained on far more text, with far more numbers inside.

---

## The whole journey on one page

```mermaid
flowchart TD
    subgraph CH2["📖 Chapter 2: Text into numbers"]
        T["Raw text<br/><i>'I HAD always thought'</i>"] --> TOK["✂️ Tokenizer<br/>split into pieces"]
        TOK --> IDS["🔢 Token IDs<br/>[40, 367, 2885, 1464]"]
        IDS --> EMB["📍 Embeddings<br/>each ID becomes a list of numbers<br/>+ where it sits in the sentence"]
    end
    subgraph CH3["👀 Chapter 3: Attention"]
        EMB --> ATT["Each word looks at the earlier words<br/>and decides which ones matter"]
    end
    subgraph CH4["🏗️ Chapter 4: The GPT model"]
        ATT --> BLK["Transformer block × N<br/>attention + thinking + shortcuts"]
        BLK --> OUT["Scores for all 50,257<br/>possible next tokens"]
    end
    subgraph CH5["🏋️ Chapter 5: Training and writing"]
        OUT --> LOSS["Loss: how wrong was the guess?"]
        LOSS --> LEARN["Nudge every number<br/>to be a little less wrong"]
        LEARN -.->|"repeat thousands of times"| EMB
        OUT --> GEN["✍️ Pick the next word, write text"]
    end
```

---

## Quick start

You need **Java 25** (or newer). Maven comes with the project (`./mvnw`).

```bash
git clone <this repo>
cd tinygpt

./run.sh 1                         # run step 1 (try 1 to 11)
./run.sh train                     # train the tiny GPT (about 20 seconds) and save it
./run.sh generate "I had always"   # let your trained model write
./mvnw test                        # run the tests
```

The first run downloads the libraries (mostly PyTorch), so it takes a while. After that it works offline.

On Windows, run the steps with Maven directly:
`mvnw compile exec:java -Dexec.mainClass=com.tinygpt.examples.Step01_SimpleTokenizer`

---

## The steps

Each step is one small program in [`src/main/java/com/tinygpt/examples`](src/main/java/com/tinygpt/examples).
Each guide in [`docs/`](docs) explains that step with pictures.

| Step | What you learn | Run | Guide | Book |
|---:|---|---|---|---|
| 1 | Split text into words and give each a number | `./run.sh 1` | [Tokenization](docs/01-tokenization.md) | 2.2–2.4 |
| 2 | GPT-2's tokenizer: break rare words into known pieces | `./run.sh 2` | [Tokenization](docs/01-tokenization.md) | 2.5 |
| 3 | Turn a story into "guess the next word" exercises | `./run.sh 3` | [Training data](docs/02-training-data.md) | 2.6 |
| 4 | Turn each token into a list of numbers, plus its position | `./run.sh 4` | [Embeddings](docs/03-embeddings.md) | 2.7–2.8 |
| 5 | Attention: words look at each other | `./run.sh 5` | [Attention](docs/04-attention.md) | 3.3 |
| 6 | Attention with learnable Query, Key, Value | `./run.sh 6` | [Attention](docs/04-attention.md) | 3.4 |
| 7 | Causal attention: no peeking at the future | `./run.sh 7` | [Attention](docs/04-attention.md) | 3.5 |
| 8 | Multi-head attention: many lookers at once | `./run.sh 8` | [Attention](docs/04-attention.md) | 3.6 |
| 9 | Build the full GPT-2 model (163 million numbers) | `./run.sh 9` | [The GPT model](docs/05-gpt-model.md) | 4 |
| 10 | Measure how wrong the model is (loss) | `./run.sh 10` | [Training](docs/06-training.md) | 5.1 |
| 11 | Choose the next word: greedy, temperature, top-k | `./run.sh 11` | [Writing text](docs/07-generating-text.md) | 5.3 |
| 12 | **Train TinyGPT and make it write** | `./run.sh train` | [Training](docs/06-training.md) | 5.2, 5.4 |

New words? See the [glossary](docs/glossary.md).

### 🖼️ Every class, explained with pictures

Not great at maths? Start here. **[docs/classes](docs/classes/README.md)** explains every single class in the project. Each one gets:
a one-sentence summary, a real-life comparison, a diagram, and a tiny worked example with small numbers you can check by hand.
Every class's Javadoc also links straight to its picture.

---

## How the code is organised

The packages follow the order of the book. Each package has one job. (Pictures for every class: [docs/classes](docs/classes/README.md).)

```mermaid
flowchart LR
    tokenizer["📦 tokenizer<br/>text ⇄ numbers"] --> data["📦 data<br/>windows + batches"]
    data --> model
    nn["📦 nn<br/>Lego bricks:<br/>Linear, Embedding,<br/>LayerNorm, GELU,<br/>Dropout, FeedForward"] --> attention["📦 attention<br/>Simple → Self →<br/>Causal → MultiHead"]
    nn --> model["📦 model<br/>TransformerBlock,<br/>GPTModel"]
    attention --> model
    model --> training["📦 training<br/>Loss, AdamW,<br/>Trainer"]
    model --> generation["📦 generation<br/>TextGenerator"]
```

```
src/main/java/com/tinygpt/
├── TinyGPT.java                  ← step 12: train, save and generate (the main program)
├── tokenizer/                    Chapter 2: text ⇄ numbers
│   ├── Tokenizer.java              the interface: encode() and decode()
│   ├── SimpleTokenizer.java        one ID per word, made by us
│   └── BpeTokenizer.java           GPT-2's real tokenizer
├── data/                         Chapter 2: training examples
│   ├── SampleText.java             loads "The Verdict" (the story we train on)
│   ├── GPTDataset.java             sliding window: input + target
│   └── GPTDataLoader.java          groups examples into batches
├── nn/                           Building blocks (like PyTorch's torch.nn)
│   ├── Module.java                 parent class: keeps the learnable numbers
│   ├── Linear.java                 multiply by a weight matrix
│   ├── Embedding.java              look up a row in a table
│   ├── LayerNorm.java              keep numbers at a steady size
│   ├── GELU.java                   the smooth "bend"
│   ├── Dropout.java                randomly switch numbers off while training
│   └── FeedForward.java            Linear → GELU → Linear
├── attention/                    Chapter 3
│   ├── SimpleAttention.java        no learnable weights
│   ├── SelfAttention.java          Query, Key, Value
│   ├── CausalAttention.java        + mask so nobody peeks ahead
│   └── MultiHeadAttention.java     several heads at once (used by GPT)
├── model/                        Chapter 4
│   ├── GPTConfig.java              all size settings (TINY and GPT2_SMALL)
│   ├── TransformerBlock.java       attention + feed forward + shortcuts
│   └── GPTModel.java               the whole model, plus save/load
├── training/                     Chapter 5
│   ├── Loss.java                   cross-entropy: "how wrong?"
│   ├── AdamW.java                  the optimizer: "which way to nudge?"
│   └── Trainer.java                the training loop
├── generation/
│   └── TextGenerator.java          greedy, temperature and top-k sampling
└── examples/                     Steps 1–11, one small program each
```

---

## Our tiny model compared with GPT-2

| | TinyGPT (`GPTConfig.TINY`) | GPT-2 small (`GPTConfig.GPT2_SMALL`) |
|---|---:|---:|
| Transformer blocks | 4 | 12 |
| Attention heads | 4 | 12 |
| Numbers per token | 128 | 768 |
| Longest text it can read | 64 tokens | 1,024 tokens |
| Learnable numbers | 13.7 million | 163 million |
| Training text | 1 short story (5,145 tokens) | 8 million web pages |
| Training time | ~20 seconds on a laptop | weeks on many GPUs |

Same design, just smaller. Change one line in `TinyGPT.java` to try other sizes.

---

## Why Java, and what is DJL?

The book uses PyTorch, a Python library that does two jobs for us:

1. **Tensors**: fast maths on big grids of numbers.
2. **Autograd**: works out automatically how to nudge each number during training.

In Java we use [DJL](https://djl.ai/) (Deep Java Library), which runs PyTorch underneath.
`NDArray` in this project is the same idea as `torch.Tensor` in the book.
Everything else (the layers, attention, the model, the optimizer, the training loop and text generation) is written out by hand, so you can read every step.

---

## Known issue: an occasional skipped step

On some computers, PyTorch's native code sometimes returns a NaN ("not a number") gradient during training. It happens at random,
and the same run can be fine the next time. We tested this a lot: the model's maths is correct, and it happens even on one thread.
If it isn't handled, a single NaN spreads until every weight is NaN.

So the optimizer checks the gradients first, and **skips any step that contains a NaN**. PyTorch's own mixed-precision trainer does the same.
If it happens you will see a line like `(Skipped 1 of 225 steps ...)`. Training is otherwise unaffected.

---

## Credits

- Book: *Build a Large Language Model (From Scratch)* by Sebastian Raschka (Manning, 2024). Python code: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).
- Training text: *The Verdict* by Edith Wharton (1908, public domain).
- Libraries: [DJL](https://djl.ai/) with PyTorch, and [JTokkit](https://github.com/knuddelsgmbh/jtokkit) for GPT-2's tokenizer.
