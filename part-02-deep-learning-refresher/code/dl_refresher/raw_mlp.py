"""raw_mlp.py - a small neural network written with plain tensors.

What it does: builds the smallest useful network (2 numbers in, 16 hidden units, 2 class scores out) from
four loose tensors, and trains it by hand: forward pass, loss, backward pass, then a manual step that
nudges every parameter downhill. Everything is built from plain tensors, so every line of the learning loop
is visible.
What it needs: torch.
What to expect: the loss falls from about 0.7 to about 0.06 and the accuracy rises to about 98 percent.
"""
import math
import time

import torch


def init_params(hidden: int = 16, seed: int = 0):
    """Make the four parameter tensors: W1, b1 (first layer) and W2, b2 (second layer).

    requires_grad=True tells PyTorch to remember how each parameter was used, so it can work out
    how the loss changes when the parameter moves.
    """
    gen = torch.Generator().manual_seed(seed)              # a seeded generator keeps every run identical
    w1 = (torch.randn(2, hidden, generator=gen) * math.sqrt(2 / 2)).requires_grad_()       # 2 inputs -> hidden units
    b1 = torch.zeros(hidden, requires_grad=True)                                           # one offset per hidden unit
    w2 = (torch.randn(hidden, 2, generator=gen) * math.sqrt(2 / hidden)).requires_grad_()  # hidden units -> 2 classes
    b2 = torch.zeros(2, requires_grad=True)                                                # one offset per class
    return [w1, b1, w2, b2]


def forward(params, X):
    """The forward pass: points in, one score per class out (called logits)."""
    w1, b1, w2, b2 = params
    hidden = torch.relu(X @ w1 + b1)                       # (batch, 2) @ (2, hidden) -> (batch, hidden); ReLU keeps positives
    return hidden @ w2 + b2                                # (batch, hidden) @ (hidden, 2) -> (batch, 2)


def softmax_cross_entropy(logits, y):
    """Loss for classification, written out by hand.

    Step 1, softmax: turn each row of scores into probabilities that add up to 1.
    Step 2, cross-entropy: take minus the log of the probability given to the right class.
    """
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)    # log of softmax, computed in a stable way
    return -log_probs[torch.arange(len(y)), y].mean()                    # average over the batch


def accuracy(logits, y) -> float:
    """Share of points whose highest score is the right class."""
    return (logits.argmax(dim=1) == y).float().mean().item()


def evaluate(params, X, y):
    """Loss and accuracy without recording any gradients (this is 'sitting the exam')."""
    with torch.no_grad():
        logits = forward(params, X)
        return softmax_cross_entropy(logits, y).item(), accuracy(logits, y)


def train_raw(train, val, hidden: int = 16, lr: float = 1.0, steps: int = 1000, seed: int = 0, snapshot_steps=()):
    """Train with plain gradient descent on all points at once.

    Returns a dict with the history (loss and accuracy at every step), the trained params and
    any requested snapshots (a copy of the params at the chosen steps, used for the lesson visuals).
    """
    (Xtr, ytr), (Xva, yva) = train, val
    params = init_params(hidden, seed)
    history = {"step": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    snapshots = {}

    def record(step):
        tl, ta = evaluate(params, Xtr, ytr)
        vl, va = evaluate(params, Xva, yva)
        for key, value in zip(history, (step, tl, ta, vl, va)):
            history[key].append(value)
        if step in snapshot_steps:
            snapshots[step] = [p.detach().clone() for p in params]

    start = time.perf_counter()
    record(0)                                              # the starting point, before any learning
    for step in range(1, steps + 1):
        logits = forward(params, Xtr)                      # 1. forward: make predictions
        loss = softmax_cross_entropy(logits, ytr)          # 2. loss: how far from the targets?
        loss.backward()                                    # 3. backward: a gradient for every parameter
        with torch.no_grad():                              # 4. step: nudge every parameter downhill
            for p in params:
                p -= lr * p.grad                           # new parameter = old parameter - learning rate * gradient
                p.grad = None                              # clear the gradient before the next round
        record(step)
    seconds = time.perf_counter() - start
    return {"history": history, "params": params, "snapshots": snapshots, "seconds": seconds}
