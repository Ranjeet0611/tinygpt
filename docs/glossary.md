# Glossary: new words, in plain language

← [Back to README](../README.md)

| Word | Simple meaning | Where |
|---|---|---|
| **LLM** (Large Language Model) | A program that has read a huge amount of text and learned to guess the next word really well. | [what is an LLM](00-what-is-an-llm.md) |
| **GPT** | "Generative Pre-trained Transformer". A type of LLM: it *generates* text, is *pre-trained* on lots of text, and is built from *transformer* blocks. | [what is an LLM](00-what-is-an-llm.md) |
| **Pretraining / fine-tuning** | First learning from lots of ordinary text; then extra training for a special job. | [what is an LLM](00-what-is-an-llm.md) |
| **Encoder / decoder** | The "understanding" half and the "writing" half of the original transformer. GPT keeps only the decoder. | [what is an LLM](00-what-is-an-llm.md) |
| **Broadcasting** | Reusing a smaller grid for every row of a bigger one, e.g. the same position rows for every window in a batch. | [embeddings](03-embeddings.md) |
| **Context vector** | A token's vector after attention: itself, mixed with the relevant words before it. | [attention](04-attention.md) |
| **Vanishing gradient** | The learning signal fading away as it travels back through many layers. Shortcuts fix it. | [model](05-gpt-model.md) |
| **Token** | A piece of text: a word, part of a word, or a punctuation mark. | [tokenization](01-tokenization.md) |
| **Token ID** | The number that stands for a token. | [tokenization](01-tokenization.md) |
| **Vocabulary** | The list of all tokens the model knows. GPT-2 knows 50,257. | [tokenization](01-tokenization.md) |
| **BPE** (Byte Pair Encoding) | A way to split text into common word-pieces, so no word is ever "unknown". | [tokenization](01-tokenization.md) |
| **Context length** | The most tokens the model can read at once. Like how much of the page fits through a window. | [training data](02-training-data.md) |
| **Batch** | A group of examples processed together. | [training data](02-training-data.md) |
| **Vector** | A list of numbers, e.g. `[0.2, -1.3, 0.8]`. | [embeddings](03-embeddings.md) |
| **Tensor / NDArray** | A grid of numbers with any number of dimensions. A vector is 1D, a table is 2D, a batch of tables is 3D... | everywhere |
| **Shape** | The size of a tensor in each direction, e.g. `[8, 4, 256]` = 8 × 4 × 256 numbers. | everywhere |
| **Embedding** | The vector that represents a token (or a position). Learned during training. | [embeddings](03-embeddings.md) |
| **Dot product** | Multiply two lists number by number and add it all up. Big result = similar. | [attention](04-attention.md) |
| **Softmax** | Turns any list of scores into percentages that add up to 100%. | [attention](04-attention.md) |
| **Attention** | Lets each word look at other words and take information from the important ones. | [attention](04-attention.md) |
| **Query / Key / Value** | The question a word asks / the label each word shows / the information each word hands over. | [attention](04-attention.md) |
| **Causal mask** | Hides future words so the model can't cheat. | [attention](04-attention.md) |
| **Head** | One attention unit. Multi-head = several looking for different things. | [attention](04-attention.md) |
| **Dropout** | Randomly switching off some numbers during training, so the model doesn't rely on any one of them. | [attention](04-attention.md) |
| **Linear layer** | Multiply by a weight matrix (and add a bias). The most basic learnable layer. | [model](05-gpt-model.md) |
| **Weights / parameters** | All the numbers inside the model that training changes. TinyGPT has 13.7 million. | [model](05-gpt-model.md) |
| **LayerNorm** | Rescales numbers to a steady size (average 0, spread 1). | [model](05-gpt-model.md) |
| **GELU** | A smooth bend that lets the model learn more than straight lines. | [model](05-gpt-model.md) |
| **Feed forward** | A small 2-layer network each token goes through on its own. | [model](05-gpt-model.md) |
| **Shortcut / residual** | Adding a layer's input back to its output, so information isn't lost. | [model](05-gpt-model.md) |
| **Transformer block** | Attention + feed forward + norms + shortcuts. GPT stacks many of these. | [model](05-gpt-model.md) |
| **Logits** | The raw scores the model gives each possible next token (before softmax). | [model](05-gpt-model.md) |
| **Loss** | One number that says how wrong the model was. Training makes it smaller. | [training](06-training.md) |
| **Cross-entropy** | The kind of loss we use: −ln(probability of the right answer). | [training](06-training.md) |
| **Perplexity** | e^loss. Roughly "how many tokens the model is still unsure between". | [training](06-training.md) |
| **Gradient** | For each weight: which way to move it, and how much, to lower the loss. | [training](06-training.md) |
| **Backpropagation / autograd** | Computing all gradients automatically, working backwards from the loss. | [training](06-training.md) |
| **Optimizer (AdamW)** | The rule for turning gradients into weight updates. | [training](06-training.md) |
| **Learning rate** | How big each update step is. | [training](06-training.md) |
| **Epoch** | One full pass over all the training data. | [training](06-training.md) |
| **Validation set** | Text kept aside and never trained on, used to check the model on new text. | [training](06-training.md) |
| **Overfitting** | Memorising the training text instead of learning general patterns. | [training](06-training.md) |
| **NaN** | "Not a Number": what you get from things like 0 ÷ 0. One NaN in training spreads everywhere. | [README](../README.md#known-issue-an-occasional-skipped-step) |
| **Pretrained weights** | Numbers someone else already trained (e.g. OpenAI's GPT-2) that you load into the same design instead of training from zero. | [chat](09-chat-with-gpt2.md) |
| **safetensors** | A file format for sharing model numbers: a table of contents, then all the numbers. | [chat](09-chat-with-gpt2.md) |
| **Instruction fine-tuning** | Extra practice on "request → good answer" examples, so the model answers instead of just continuing text. | [chat](09-chat-with-gpt2.md) |
| **Prompt format (Alpaca)** | The fixed layout (`### Instruction:` … `### Response:`) that shows the model where the question ends and its answer begins. | [chat](09-chat-with-gpt2.md) |
| **Padding** | Filler tokens added so different-length examples fit in one batch. | [chat](09-chat-with-gpt2.md) |
| **Ignore index (-100)** | A target value meaning "don't grade this position", used for padding. | [chat](09-chat-with-gpt2.md) |
| **Weight tying** | Using the same table for the token embedding and the output head, as GPT-2 does. | [embeddings](03-embeddings.md) |
| **Hallucination** | A model stating something false with confidence. | [chat](09-chat-with-gpt2.md) |
| **Greedy decoding** | Always picking the highest-scoring next token. | [writing text](07-generating-text.md) |
| **Temperature** | How random the choice is. Low = careful, high = creative. | [writing text](07-generating-text.md) |
| **Top-k** | Only choosing among the k best tokens. | [writing text](07-generating-text.md) |
