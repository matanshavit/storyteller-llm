# Storyteller LLM

A GPT-2-class transformer trained on TinyStories for conversational story generation, built from scratch.

## Foundations

### Chapter 1: Bigram Language Model

A character-level bigram model for name generation, implemented two ways: a counting-based statistical approach and an equivalent single-layer neural network trained with gradient descent. Establishes the core training loop (forward → loss → backward → update) and key concepts (NLL loss, softmax, one-hot encoding) that everything else builds on.

**Notebook**: [`foundations/bigram.ipynb`](foundations/bigram.ipynb)

### Chapter 2: Micrograd — Backpropagation

A minimal autograd engine built from scratch: a `Value` class that tracks a computational graph and computes gradients via reverse-mode automatic differentiation. On top of it, a small neural network library (`Neuron`, `Layer`, `MLP`) trained on a binary classification task (make_moons), with gradients verified against PyTorch.

**Code**: [`foundations/micrograd/`](foundations/micrograd/) · **Demo**: [`foundations/micrograd/demo.ipynb`](foundations/micrograd/demo.ipynb)
