# Package `attention`: words looking at other words

← [nn](3-nn.md) · [All classes](README.md) · Next: [model →](5-model.md)

To understand a word, you look at the words around it. "Bank" means something different in
*"the river bank"* and *"money in the bank"*. Attention lets every word look at the others.

The four classes build up the idea one piece at a time:

```mermaid
flowchart LR
    A["SimpleAttention<br/>look and mix"] -->|"+ learn what<br/>to look for"| B["SelfAttention"]
    B -->|"+ no peeking<br/>at the future"| C["CausalAttention"]
    C -->|"+ many lookers<br/>at once"| D["MultiHeadAttention<br/>⭐ used in GPT"]
```

---

## SimpleAttention

🗣️ **In one sentence:** each word checks **how similar** it is to every word, turns that into percentages, and becomes a **mix** of all the words.

🌍 **Real-life picture:** making a smoothie. Each fruit you add is a word. The more a word matters, the more of it goes into the cup.

🖼️ **Diagram** (focusing on the word "mat")

```mermaid
flowchart TD
    subgraph S1["1 · How alike? (match score)"]
        direction LR
        c1["cat<br/>1"] ~~~ s1["sat<br/>1"] ~~~ m1["mat<br/>2"]
    end
    subgraph S2["2 · Turn into percentages (adds up to 100%)"]
        direction LR
        c2["cat<br/>21%"] ~~~ s2["sat<br/>21%"] ~~~ m2["mat<br/>58%"]
    end
    subgraph S3["3 · Mix 🥤"]
        r["new 'mat' = 21% cat + 21% sat + 58% mat"]
    end
    S1 --> S2 --> S3
```

🔢 **Tiny example:** three words, two numbers each:

| Word | Numbers |
|---|---|
| cat | `[1, 0]` |
| sat | `[0, 1]` |
| mat | `[1, 1]` |

**Step 1 – match score** (multiply the matching numbers, then add):
- mat with cat: `1×1 + 1×0 = 1`
- mat with sat: `1×0 + 1×1 = 1`
- mat with mat: `1×1 + 1×1 = 2`

**Step 2 – percentages** (*softmax*): bigger scores get a bigger slice of the pizza, and the slices always add up to 100%.
Scores `1, 1, 2` → `21%, 21%, 58%`.

**Step 3 – mix:** `0.21 × [1,0] + 0.21 × [0,1] + 0.58 × [1,1] = [0.79, 0.79]`

So "mat" now carries a little bit of "cat" and "sat" inside it.

🔗 **In the code:** [`SimpleAttention.java`](../../src/main/java/com/tinygpt/attention/SimpleAttention.java) · `scores()`, `weights()`, `contextVectors()` · Run it: `./run.sh 5`

---

## SelfAttention

🗣️ **In one sentence:** like SimpleAttention, but each word first makes three **learned** versions of itself: a **Query** (what am I looking for?), a **Key** (what do I have?) and a **Value** (what do I give?).

🌍 **Real-life picture:** a library.
- **Query** 🔍 = the question you bring: *"books about cats"*
- **Key** 🏷️ = the label on each book's spine
- **Value** 📖 = what's inside the book

You compare your question with every label, then read mostly from the books whose labels match.

🖼️ **Diagram**

```mermaid
flowchart LR
    W["each word"] -->|"Linear"| Q["🔍 Query"]
    W -->|"Linear"| K["🏷️ Key"]
    W -->|"Linear"| V["📖 Value"]
    Q --> M{"compare<br/>🔍 with every 🏷️"}
    K --> M
    M -->|"÷ a calming number,<br/>then percentages"| P["how much to read<br/>from each word"]
    P --> MIX(("mix"))
    V --> MIX
    MIX --> OUT["new word"]
```

🔢 **Tiny example:** the word "it" is looking at "cat" and "mat":

| | Query of "it" | Key | Value |
|---|---|---|---|
| | `[1, 0]` | | |
| cat | | `[1, 0]` | `[5, 0]` |
| mat | | `[0, 1]` | `[0, 5]` |

1. **Compare:** with cat `1×1 + 0×0 = 1`; with mat `1×0 + 0×1 = 0`
2. **Calm down** (divide by √2 ≈ 1.41, because each list has 2 numbers): `0.71` and `0`
3. **Percentages:** cat `67%`, mat `33%`
4. **Mix the Values:** `0.67 × [5,0] + 0.33 × [0,5] = [3.35, 1.65]`

"it" took mostly from "cat". The Query/Key/Value weights are **learned**, so the model decides for itself what should match.

🔗 **In the code:** [`SelfAttention.java`](../../src/main/java/com/tinygpt/attention/SelfAttention.java) · `forward()`, `attend()` · Run it: `./run.sh 6`

---

## CausalAttention

🗣️ **In one sentence:** self-attention where each word may only look at **itself and the words before it**, never the words after it.

🌍 **Real-life picture:** a test where you must cover the answers below with a sheet of paper. If you could peek, you'd learn nothing.

🖼️ **Diagram** (✅ = may look, ❌ = hidden)

```mermaid
block-beta
    columns 4
    h[" "] c1["I"] c2["love"] c3["pizza"]
    r1["I"] a11["✅"] a12["❌"] a13["❌"]
    r2["love"] a21["✅"] a22["✅"] a23["❌"]
    r3["pizza"] a31["✅"] a32["✅"] a33["✅"]
```

🔢 **Tiny example:** the word "love" has match scores `I: 2, love: 1, pizza: 3`.

- Without a mask: "pizza" would get the most attention, but "pizza" is the answer we want to guess. That's cheating!
- With the mask: the "pizza" score becomes **minus infinity** (smaller than any number), which turns into **0%**.
- Scores `2, 1, −∞` → percentages `73%, 27%, 0%` ✅

It also uses `Dropout` on the percentages while practising.

🔗 **In the code:** [`CausalAttention.java`](../../src/main/java/com/tinygpt/attention/CausalAttention.java) · `causalMask()` builds the ✅/❌ grid · Run it: `./run.sh 7`

---

## MultiHeadAttention

🗣️ **In one sentence:** runs **several causal attentions side by side** ("heads"), each free to look for something different, then joins their findings.

🌍 **Real-life picture:** a team of detectives searching the same room. One looks for fingerprints, one for footprints, one for notes. Together they understand much more.

🖼️ **Diagram**

```mermaid
flowchart TD
    X["word: [1, 2, 3, 4]"] --> SPLIT{{"✂️ split in half"}}
    SPLIT --> H1["🕵️ head 1 gets [1, 2]<br/>looks for one thing"]
    SPLIT --> H2["🕵️ head 2 gets [3, 4]<br/>looks for another"]
    H1 --> JOIN{{"🔗 glue back together"}}
    H2 --> JOIN
    JOIN --> L["Linear<br/>(heads share notes)"]
    L --> OUT["new word: 4 numbers"]
```

🔢 **Tiny example:** a word has 4 numbers and there are 2 heads, so each head gets 2 numbers.
Each head does its own causal attention. The results are glued back into 4 numbers, and a final `Linear` mixes the heads' findings.

| | TinyGPT | GPT-2 |
|---|---:|---:|
| Numbers per word | 128 | 768 |
| Heads | 4 | 12 |
| Numbers per head | 32 | 64 |

We don't choose what each head looks for. That is learned during training.

🔗 **In the code:** [`MultiHeadAttention.java`](../../src/main/java/com/tinygpt/attention/MultiHeadAttention.java) · `splitHeads()` does the ✂️ · used in every `TransformerBlock` · Run it: `./run.sh 8`
