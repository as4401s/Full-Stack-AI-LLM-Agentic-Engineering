"""skipgram.py - train an embedding table from scratch on the invented clinical sentences (PyTorch).

What it does
    Builds the smallest model that learns word vectors: an embedding table (one row of numbers per word) followed by one
    linear layer that scores every word of the vocabulary. The question it practises is "this word is in a sentence; which
    other word is in the same sentence?". Training nudges the rows so that words with the same company get similar rows.
    This is the skip-gram idea of word2vec (Mikolov et al., 2013) in a few lines, with the same loss, softmax and Adam
    steps used in lessons 2 and 3.
What it needs
    PyTorch and NumPy (see pyproject.toml). It runs on a laptop CPU in well under a minute.
What to expect
    train() returns the learned table as a NumPy array plus the loss after every epoch, so a demo can print the curve.
    Random guessing among V words costs ln(V) per pair; a model that has learned the company of each word scores lower.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SkipGram(nn.Module):
    """Embedding table -> linear layer. forward() returns one score (logit) per vocabulary word for every centre word."""

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.table = nn.Embedding(vocab_size, dim)      # the lookup table: row i holds the `dim` numbers of word i
        self.out = nn.Linear(dim, vocab_size)           # turns a row into one score for every possible neighbour word

    def forward(self, centre_ids):
        """centre_ids: tensor of word ids with shape (batch,). Returns scores with shape (batch, vocab_size)."""
        return self.out(self.table(centre_ids))


def average_loss(model, centre, neighbour, batch=8192):
    """Average cross-entropy loss over all pairs, without changing the model (used for the unseen test sentences)."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(centre), batch):
            logits = model(centre[start:start + batch])
            total += F.cross_entropy(logits, neighbour[start:start + batch], reduction="sum").item()
    return total / len(centre)


def train(centre, neighbour, vocab_size, dim, epochs=20, lr=0.02, batch=4096, seed=0, test_pairs=None):
    """Train a SkipGram model and return {'table', 'history', 'train_loss', 'test_loss'}.

    centre, neighbour: integer arrays from corpus.neighbour_pairs().   dim: numbers per word.
    epochs: passes over all pairs.   lr: Adam step size.   batch: pairs per step.   seed: makes the run repeatable.
    test_pairs: optional (centre, neighbour) arrays from sentences the model never trains on.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    centre_t, neighbour_t = torch.as_tensor(centre), torch.as_tensor(neighbour)
    model = SkipGram(vocab_size, dim)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(centre_t), generator=generator)   # a fresh shuffle of the pairs every epoch
        total = 0.0
        for start in range(0, len(order), batch):
            ids = order[start:start + batch]
            loss = F.cross_entropy(model(centre_t[ids]), neighbour_t[ids])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(ids)
        history.append(total / len(order))
    result = {"table": model.table.weight.detach().numpy().copy(), "history": history, "train_loss": average_loss(model, centre_t, neighbour_t)}
    if test_pairs is not None:
        result["test_loss"] = average_loss(model, torch.as_tensor(test_pairs[0]), torch.as_tensor(test_pairs[1]))
    return result
