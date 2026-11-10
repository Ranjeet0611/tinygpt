# Packages `pretrained`, `instruct` and `dialog`: borrowing GPT-2's brain and teaching it to chat

← [main and examples](8-main-and-examples.md) · [All classes](README.md) · Concept guides: [9. Chat with GPT-2](../09-chat-with-gpt2.md), [10. Talk](../10-talk-with-dailydialog.md)

```mermaid
flowchart LR
    Z["📏 GPT2Size<br/>small or medium"] --> D
    D["⬇️ Downloads"] --> S["📦 SafeTensors<br/>reads the weight file"]
    S --> W["🧠 GPT2Weights<br/>fills our GPTModel"]
    W --> F["🎓 FineTune<br/>practises instructions"]
    I["📝 Instruction"] --> L["📦 InstructionDataLoader"]
    L --> F
    F -->|"saves"| C["💬 Chat"]
    W --> FT["🎓 FineTuneTalk<br/>practises conversations"]
    DL["🎭 Dialogue"] --> L
    L --> FT
    FT -->|"saves"| T["🗣️ Talk"]
```

---

## Downloads

🔬 **Line by line with real numbers:** [walkthrough](../walkthroughs/step12-load-gpt2.md)

🗣️ **In one sentence:** downloads a file the first time it's needed, and reuses the saved copy after that.

🌍 **Real-life picture:** buying a book once and keeping it on your shelf, instead of going back to the shop every time you want to read it.

🖼️ **Diagram**

```mermaid
flowchart LR
    Q{"file already<br/>on disk?"} -->|"yes"| U["use it"]
    Q -->|"no"| D["download to<br/>'file.part'<br/>(prints 10%, 20%...)"]
    D --> R["rename to 'file'"]
    R --> U
```

🔢 **Tiny example:** GPT-2's weights are **548 MB** (`models/gpt2/model.safetensors`), and the book's instructions are **200 KB** (`data/instruction-data.json`).
Downloading to `.part` first means a download that was interrupted halfway is never mistaken for a finished one.

🔗 **In the code:** [`Downloads.java`](../../src/main/java/com/tinygpt/pretrained/Downloads.java) · `ifMissing(url, file)` · used by `FineTune` and `Step12_LoadGPT2`.

---

## SafeTensors

🔬 **Line by line with real numbers:** [walkthrough](../walkthroughs/step12-load-gpt2.md)

🗣️ **In one sentence:** reads a `.safetensors` file, the format Hugging Face uses to share a model's learned numbers.

🌍 **Real-life picture:** a book with a **table of contents** at the front: "Chapter *wte.weight* is on pages 287,209,472 to 441,598,976". You look up the name, then jump straight to those pages.

🖼️ **Diagram**

```mermaid
block-beta
    columns 3
    a["8 bytes:<br/>how long the<br/>table of contents is"] b["table of contents (JSON)<br/>wte.weight → F32, [50257, 768], bytes 287,209,472–441,598,976<br/>h.0.ln_1.weight → F32, [768], ...<br/>... 160 entries"] c["all the numbers,<br/>one after another<br/>(548 MB)"]
```

🔢 **Tiny example** (from the tests): a file whose table of contents says
`"a": shape [2, 2], bytes 0–16` and `"b": shape [2], bytes 16–24`, followed by the numbers `1 2 3 4 5 6`.
`get("a")` → `[1, 2, 3, 4]` (4 numbers × 4 bytes = 16 bytes) and `get("b")` → `[5, 6]`.

🔗 **In the code:** [`SafeTensors.java`](../../src/main/java/com/tinygpt/pretrained/SafeTensors.java) · `open(file)`, `names()`, `get(name)`.

---

## GPT2Weights

🔬 **Line by line with real numbers:** [walkthrough](../walkthroughs/step12-load-gpt2.md)

🗣️ **In one sentence:** copies OpenAI's trained GPT-2 numbers into **our own** `GPTModel`, matching each of OpenAI's part names to ours.

🌍 **Real-life picture:** moving into a house that's already furnished. The rooms (our model's parts) are the same shape as in the old house, so every piece of furniture fits; you just need to know which room each box goes in.

🖼️ **Diagram**

```mermaid
flowchart LR
    subgraph OPENAI["OpenAI's file"]
        o1["wte.weight"]
        o2["wpe.weight"]
        o3["h.0.attn.c_attn.weight<br/>[768, 2304]"]
        o4["h.0.mlp.c_fc.weight"]
        o5["ln_f.weight"]
    end
    subgraph OURS["our GPTModel"]
        u1["tokenEmbedding.weight"]
        u2["positionEmbedding.weight"]
        u3["blocks.0.attention.query.weight<br/>blocks.0.attention.key.weight<br/>blocks.0.attention.value.weight<br/>(each [768, 768])"]
        u4["blocks.0.feedForward.expand.weight"]
        u5["finalNorm.scale"]
        u6["outputHead.weight"]
    end
    o1 --> u1
    o1 -->|"turned sideways"| u6
    o2 --> u2
    o3 -->|"✂️ cut into 3"| u3
    o4 --> u4
    o5 --> u5
```

🔢 **Tiny example: cutting Query/Key/Value apart.** OpenAI stores all three side by side in one wide grid. With 2 rows and 6 columns, cut into 3:

```
[ 1  2 | 3  4 | 5  6 ]        query = [1 2]   key = [3  4]   value = [5  6]
[ 7  8 | 9 10 |11 12 ]                [7 8]         [9 10]         [11 12]
```

GPT-2 small has 12 blocks, so this happens 12 times. Then a check makes sure **every** part of our model got numbers. If anything is missing, it stops with an error.

The settings must match GPT-2 exactly, so they come from [`GPT2Size`](#gpt2size) (Q/K/V **with** bias, as OpenAI used, and dropout 0).

✅ **Proof it works:** with these numbers, our Java model continues *"I enjoy walking with my cute dog"* exactly like the real GPT-2:
*", but I'm not sure if I'll ever be able to walk with my dog."*

🔗 **In the code:** [`GPT2Weights.java`](../../src/main/java/com/tinygpt/pretrained/GPT2Weights.java) · `load(model, file)` · relies on `Module.namedParameters()` · Run it: `./run.sh 12`

---

## GPT2Size

🔬 **Line by line with real numbers:** [walkthrough](../walkthroughs/step12-load-gpt2.md)

🗣️ **In one sentence:** the two GPT-2 sizes you can download (small and medium), each with its settings card, download address and file names.

🌍 **Real-life picture:** a menu with a small and a large size. Same recipe, different amounts.

🖼️ **Diagram**

```mermaid
flowchart TD
    S["GPT2Size.SMALL<br/>124M numbers · 12 layers · 768 wide<br/>548 MB"] --> F1["models/gpt2/model.safetensors"]
    S --> C1["models/gpt2-chat.params"]
    M["GPT2Size.MEDIUM<br/>355M numbers · 24 layers · 1024 wide<br/>1.5 GB (the book's choice)"] --> F2["models/gpt2-medium/model.safetensors"]
    M --> C2["models/gpt2-medium-chat.params"]
```

🔢 **Tiny example:** `./run.sh finetune medium` → `GPT2Size.fromArgs(["medium"])` → `MEDIUM`. Anything else gives `SMALL`.

🔗 **In the code:** [`GPT2Size.java`](../../src/main/java/com/tinygpt/pretrained/GPT2Size.java) · `config()`, `url()`, `weightsFile()`, `chatModelFile()`.

---

## Instruction

🔬 **Line by line with real numbers:** [walkthrough](../walkthroughs/finetune-and-chat.md)

🗣️ **In one sentence:** one practice example (*a request, optional extra input, and a good answer*), and how to write it out so the model can tell the request and the answer apart.

🌍 **Real-life picture:** a worksheet with labelled boxes: **Question**, **Extra info**, **Your answer**. The labels tell you where to write.

🖼️ **Diagram**

```mermaid
flowchart TD
    J["JSON entry<br/>instruction · input · output"] --> P["prompt()"]
    P --> T["Below is an instruction that describes a task. ...<br/><br/>### Instruction:<br/>Rewrite the sentence using a simile.<br/><br/>### Input:<br/>The car is very fast.<br/><br/>### Response:"]
    T --> F["fullText() = prompt + the answer<br/>The car is as fast as lightning."]
```

🔢 **Tiny example:** if `input` is empty (e.g. *"Name a color."*), the `### Input:` section is left out completely.
When chatting, we send only `prompt()`, which ends at `### Response:`, and the model writes what comes next: the answer.

🔗 **In the code:** [`Instruction.java`](../../src/main/java/com/tinygpt/instruct/Instruction.java) · a Java `record` · `prompt()`, `fullText()`, `load(json)`.

---

## InstructionDataLoader

🔬 **Line by line with real numbers:** [walkthrough](../walkthroughs/finetune-and-chat.md)

🗣️ **In one sentence:** packs instruction examples (which all have **different lengths**) into batches, padding the short ones and marking the padding as "don't grade".

🌍 **Real-life picture:** packing different-length pencils into a box. Short ones get a cap on the end so they all line up, and the teacher knows the caps aren't part of the pencils.

🖼️ **Diagram** (the book's own example)

```mermaid
flowchart TD
    A["three examples<br/>[0 1 2 3 4]   [5 6]   [7 8 9]"] -->|"add one E (50256) each,<br/>pad with E to the longest"| B["[0 1 2 3 4 E]<br/>[5 6 E E E E]<br/>[7 8 9 E E E]"]
    B -->|"input = all but the last"| C["[0 1 2 3 4]<br/>[5 6 E E E]<br/>[7 8 9 E E]"]
    B -->|"target = all but the first"| D["[1 2 3 4 E]<br/>[6 E E E E]<br/>[8 9 E E E]"]
    D -->|"keep the first E,<br/>extra E's → -100"| E["[1 2 3 4 E]<br/>[6 E ✗ ✗ ✗]<br/>[8 9 E ✗ ✗]<br/>✗ = -100"]
```

🔢 **Why keep the first E?** It teaches the model to **say "I'm finished"** after an answer, which is how the chatbot knows when to stop.
**Why -100 for the rest?** They're just filler. `Loss` skips every position whose target is -100, so the model isn't graded on filler.

🔗 **In the code:** [`InstructionDataLoader.java`](../../src/main/java/com/tinygpt/instruct/InstructionDataLoader.java) · `collate()` · gives the same `Batch` as `GPTDataLoader`, so the same `Trainer` works for both.

---

## FineTune

🔬 **Line by line with real numbers:** [walkthrough](../walkthroughs/finetune-and-chat.md)

🗣️ **In one sentence:** the program that loads GPT-2, practises on 935 instruction examples for 2 rounds, and saves the result for chatting.

🌍 **Real-life picture:** a well-read person (GPT-2: read millions of web pages) who gets a short training course in **how to answer questions politely and to the point**.

🖼️ **Diagram**

```mermaid
flowchart TD
    A["⬇️ download GPT-2 + instructions"] --> B["✂️ split 1,100 examples<br/>935 practise · 55 check · 110 test"]
    B --> C["🧠 GPTModel(GPT2_SMALL) +<br/>GPT2Weights.load"]
    C --> D["❓ ask one question before training<br/>(it rambles)"]
    D --> E["🏋️ Trainer: 2 epochs<br/>learning rate 0.00005"]
    E --> F["💾 save models/gpt2-chat.params"]
    F --> G["📝 answer 3 test questions<br/>it has never seen"]
```

🔢 **Tiny example: counting the work.** 935 examples ÷ 8 per batch = 116 steps per round × 2 rounds = **232 steps**. On an M4 Pro, GPT-2 small took **6.1 minutes**.
The learning rate is **20 times smaller** than TinyGPT's (0.00005 vs 0.001): GPT-2 already knows a lot, and big steps would make it forget.

🔗 **In the code:** [`FineTune.java`](../../src/main/java/com/tinygpt/FineTune.java) · `answer()` asks a question and stops at `<|endoftext|>` · Run it: `./run.sh finetune`

---

## Chat

🔬 **Line by line with real numbers:** [walkthrough](../walkthroughs/finetune-and-chat.md)

🗣️ **In one sentence:** a terminal chat loop: it wraps what you type in the instruction format, lets the fine-tuned model answer, and repeats.

🌍 **Real-life picture:** a walkie-talkie. You say something, it answers, you say something else. But it **forgets** each message right after answering it.

🖼️ **Diagram**

```mermaid
sequenceDiagram
    participant You
    participant Chat
    participant TG as TextGenerator
    participant M as GPTModel (fine-tuned)
    You->>Chat: What is the opposite of hot?
    Chat->>Chat: wrap in "### Instruction: ... ### Response:"
    loop until it writes <|endoftext|>
        Chat->>TG: continuation(prompt, stop = 50256)
        TG->>M: next-token scores
        M-->>TG: pick the top one
    end
    TG-->>Chat: "The opposite of hot is cold."
    Chat-->>You: Bot: The opposite of hot is cold.
```

🔢 **Tiny example** (real answers from GPT-2 small after fine-tuning):

```
You: What is the capital of France?
Bot: The capital of France is Paris.
You: Give me a synonym for happy.
Bot: A synonym for happy is "joyful."
```

Type `Translate to French || Good morning`, and everything after `||` becomes the `### Input:` section.
It picks GPT-2 medium automatically if you've fine-tuned it; `./run.sh chat small` forces small.

🔗 **In the code:** [`Chat.java`](../../src/main/java/com/tinygpt/Chat.java) · Run it: `./run.sh chat` (after `./run.sh finetune`)

---

## Dialogue

🔬 **Line by line with real numbers:** [walkthrough](../walkthroughs/finetune-talk-and-talk.md)

🗣️ **In one sentence:** one everyday conversation from DailyDialog, written out as a **play script** (`A: ...` / `B: ...`, one line per turn).

🌍 **Real-life picture:** the script of a short school play: who says what, in order.

🖼️ **Diagram**

```mermaid
flowchart LR
    Z["📦 data/dailydialog.zip<br/>(data/dialogues.json)"] -->|"load(zip, 'train')"| D["Dialogue<br/>turns: ['Hi!', 'Hello. How are you?', 'Fine, thanks.']"]
    D -->|"text()"| T["A: Hi!<br/>B: Hello. How are you?<br/>A: Fine, thanks.<br/>"]
    D -->|"firstTokens(tokenizer, 256)"| C["only the whole turns<br/>that fit in 256 tokens"]
    H["history so far<br/>['Hi!']"] -->|"prompt(history)"| P["A: Hi!<br/>B:"]
```

🔢 **Tiny example:**
- `new Dialogue(List.of("Hi!", "Hello. How are you?")).text()` → `"A: Hi!\nB: Hello. How are you?\n"`
- `Dialogue.prompt(List.of("Hi!"))` → `"A: Hi!\nB:"`. The model finishes B's line.
- `Dialogue.NEW_LINE_ID` = **198**, GPT-2's token for `\n`. When the model writes it, its turn is over.

🔗 **In the code:** [`Dialogue.java`](../../src/main/java/com/tinygpt/dialog/Dialogue.java) · tested in `DialogueTest`

---

## FineTuneTalk

🔬 **Line by line with real numbers:** [walkthrough](../walkthroughs/finetune-talk-and-talk.md)

🗣️ **In one sentence:** the program that loads GPT-2, practises on 11,118 DailyDialog conversations for 1 round, and saves the result for talking.

🌍 **Real-life picture:** the same well-read person as in [FineTune](#finetune), but this time they watch thousands of short plays about everyday life to learn **how people chat**.

🖼️ **Diagram**

```mermaid
flowchart TD
    A["⬇️ download GPT-2 + DailyDialog (3.7 MB)"] --> B["🎭 Dialogue.load<br/>11,118 practise · 1,000 check"]
    B --> C["✂️ firstTokens(256).text()<br/>play scripts"]
    C --> L["📦 InstructionDataLoader<br/>padding + -100, as in FineTune"]
    G["🧠 GPTModel + GPT2Weights.load"] --> E
    L --> E["🏋️ Trainer: 1 epoch<br/>learning rate 0.00005"]
    E --> F["💾 save models/gpt2-talk.params"]
```

🔢 **Tiny example: counting the work.** 11,118 conversations ÷ 8 per batch = **1,389 steps** (one round). On an M4 Pro, GPT-2 small took **71.6 minutes** (validation loss 2.39 → 2.22).

`reply(generator, history)` asks for the next line: `continuation(Dialogue.prompt(history), 60, 0.7, 40, stop = 198)`.

🔗 **In the code:** [`FineTuneTalk.java`](../../src/main/java/com/tinygpt/FineTuneTalk.java) · Run it: `./run.sh finetune-talk` (or `./run.sh finetune-talk medium`)

---

## Talk

🔬 **Line by line with real numbers:** [walkthrough](../walkthroughs/finetune-talk-and-talk.md)

🗣️ **In one sentence:** a terminal conversation that **remembers**: it sends the whole conversation so far each time, and the model writes B's next line.

🌍 **Real-life picture:** unlike [Chat](#chat)'s walkie-talkie, this is a phone call: the other person remembers what you said a minute ago.

🖼️ **Diagram**

```mermaid
flowchart TD
    Y["You type a line"] --> H["history.add(your line)<br/>(keep the last 10 turns)"]
    H --> P["Dialogue.prompt(history)<br/>'A: ...\nB: ...\nA: your line\nB:'"]
    P --> R["FineTuneTalk.reply<br/>writes until \n"]
    R --> H2["history.add(reply)"]
    H2 --> O["Bot: reply"]
    O --> Y
```

🔢 **Tiny example:** after two turns the history is `["Hi! How are you?", "I'm fine, thanks. And you?"]`, so the next prompt starts with `A: Hi! How are you?\nB: I'm fine, thanks. And you?\nA: ...`.
Type `new` to clear the history, `exit` to quit.

🔗 **In the code:** [`Talk.java`](../../src/main/java/com/tinygpt/Talk.java) · Run it: `./run.sh talk` (after `./run.sh finetune-talk`)
