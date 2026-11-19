"""mlp_lm.py - a small neural language model: an embedding table plus a hidden layer.

What it does
    Predicts the next word from the few words before it, in four steps:
      1. look up each context word in an embedding table (every word owns a short list of numbers),
      2. glue those lists together,
      3. pass them through one hidden layer (multiply, add a bias, apply the tanh bend), and
      4. produce one score (a logit) for every word in the vocabulary.
    Training uses the same four-move loop as lesson 2 (forward, loss, backward, step), with cross-entropy as the loss.

What it needs
    PyTorch (see pyproject.toml). It runs on a laptop CPU in a couple of seconds.

What to expect
    With the settings in demos/d04_train_neural.py the loss falls from about ln(24) = 3.2 (a blind guess among
    24 words) to under 1, and perplexity on sentences the model has never seen is lower than the counting
    model's. The saved numbers are in data/expected_output/d04_train_neural.txt.
"""
import math

import torch
from torch import nn

from .corpus import make_examples, to_ids


class NeuralLM(nn.Module):
    """Embedding table -> hidden layer -> one score per word."""

    def __init__(self, vocab_size, context_size=2, embed_dim=2, hidden=16):
        super().__init__()
        self.context_size = context_size
        self.embed = nn.Embedding(vocab_size, embed_dim)            # one row of embed_dim numbers per word
        self.hidden = nn.Linear(context_size * embed_dim, hidden)   # multiply and add, like the units of lesson 2
        self.out = nn.Linear(hidden, vocab_size)                    # one score per word in the vocabulary

    def forward(self, context_ids):
        """context_ids has shape (batch, context_size); the result has shape (batch, vocab_size)."""
        e = self.embed(context_ids)          # (batch, context_size, embed_dim)
        e = e.flatten(start_dim=1)           # glue the context words' lists together: (batch, context_size * embed_dim)
        h = torch.tanh(self.hidden(e))       # the bend (activation function), here tanh instead of ReLU
        return self.out(h)                   # the scores (logits)


def count_parameters(model):
    """Total number of adjustable numbers (parameters) in the model."""
    return sum(p.numel() for p in model.parameters())


def make_tensors(sentences, index, context_size):
    """Turn sentences into two tensors: contexts (examples, context_size) and targets (examples,)."""
    contexts, targets = to_ids(make_examples(sentences, context_size), index)
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def train(model, contexts, targets, steps=300, lr=0.02, weight_decay=1.0, log_every=50):
    """Train with the four-move loop: forward, loss, backward, step. Returns a list of (step, loss) pairs.

    weight_decay=1.0 shrinks every parameter by 2 percent per step (lr * weight_decay), which keeps the numbers small
    and makes words that appear in similar contexts end up with similar embeddings.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for step in range(steps + 1):
        logits = model(contexts)                                   # 1. forward: scores for every example
        loss = nn.functional.cross_entropy(logits, targets)        # 2. loss: average -ln(probability of the right word)
        if step % log_every == 0:
            history.append((step, loss.item()))
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()                                            # 3. backward: the slope of the loss for every parameter
        optimizer.step()                                           # 4. step: nudge every parameter downhill
    return history


@torch.no_grad()
def evaluate(model, sentences, index):
    """Average loss and perplexity on sentences. Perplexity is e raised to the loss."""
    contexts, targets = make_tensors(sentences, index, model.context_size)
    loss = nn.functional.cross_entropy(model(contexts), targets).item()
    return loss, math.exp(loss)


@torch.no_grad()
def next_word_probs(model, context_words, words, index):
    """Return {word: probability} for the next word after the given context words, for every word in the vocabulary."""
    ids = torch.tensor([[index[w] for w in context_words]], dtype=torch.long)
    probs = torch.softmax(model(ids)[0], dim=-1)
    return {w: probs[i].item() for i, w in enumerate(words)}


@torch.no_grad()
def logits_for(model, context_words, index):
    """The raw scores (before softmax) for the next word, as a plain Python list."""
    ids = torch.tensor([[index[w] for w in context_words]], dtype=torch.long)
    return model(ids)[0].tolist()


@torch.no_grad()
def step_probs(model, sentence, words, index):
    """The probability the model gave to each word of one sentence (one number per word)."""
    from .corpus import padded
    p = padded(sentence, model.context_size)
    out = []
    for i in range(model.context_size, len(p)):
        probs = next_word_probs(model, p[i - model.context_size:i], words, index)
        out.append(probs[p[i]])
    return out
