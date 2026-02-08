# Storyteller LLM

A GPT-2-class transformer trained on TinyStories for conversational story generation, built from scratch.

## Foundations

### Chapter 1: Bigram Language Model

A character-level bigram model for name generation, implemented two ways: a counting-based statistical approach and an equivalent single-layer neural network trained with gradient descent. Establishes the core training loop (forward → loss → backward → update) and key concepts (NLL loss, softmax, one-hot encoding) that everything else builds on.

**Notebook**: [`foundations/bigram.ipynb`](foundations/bigram.ipynb)

### Chapter 2: Micrograd — Backpropagation

A minimal autograd engine built from scratch: a `Value` class that tracks a computational graph and computes gradients via reverse-mode automatic differentiation. On top of it, a small neural network library (`Neuron`, `Layer`, `MLP`) trained on a binary classification task (make_moons), with gradients verified against PyTorch.

**Code**: [`foundations/micrograd/`](foundations/micrograd/) · **Demo**: [`foundations/micrograd/demo.ipynb`](foundations/micrograd/demo.ipynb)

### Chapter 3: MLP Language Model

A character-level MLP language model following Bengio et al. (2003), extending the bigram model from 1 character of context to n characters using learned embeddings, a hidden layer with tanh activation, and softmax output. Includes a learning rate finder, train/dev/test evaluation, a GELU vs tanh activation comparison, context length experiments (block_size 3 vs 8), and a 2D embedding visualization showing how similar characters cluster together. Achieves ~2.17 NLL on dev set vs the bigram's ~2.45.

**Notebook**: [`foundations/mlp.ipynb`](foundations/mlp.ipynb)
