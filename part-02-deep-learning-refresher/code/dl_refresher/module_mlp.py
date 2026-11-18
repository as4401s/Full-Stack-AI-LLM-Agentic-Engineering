"""module_mlp.py - the same kind of network, written with PyTorch's building blocks.

What it does: wraps the layers in an nn.Module (so PyTorch finds all parameters for us), then trains it with
the standard five-line loop: zero the gradients, forward, loss, backward, optimizer step. The optimizer
(plain SGD, SGD with momentum, or AdamW) and the learning-rate schedule are chosen with plain arguments.
What it needs: torch.
What to expect: with 2 inputs, 16 hidden units and 2 outputs the network has 82 parameters; AdamW at a
learning rate of 0.01 reaches about 98 percent accuracy on the spirals within a few hundred steps.
"""
import copy
import math
import time

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """A multilayer perceptron: Linear -> ReLU (-> Dropout) repeated, then a final Linear.

    hidden=(16,) builds 2 -> 16 -> 2. hidden=(128, 128) builds 2 -> 128 -> 128 -> 2.
    dropout is the share of hidden units switched off at random while training (0.0 means none).
    """

    def __init__(self, hidden=(16,), dropout: float = 0.0, in_features: int = 2, out_features: int = 2):
        super().__init__()
        layers = []
        width = in_features
        for size in hidden:
            layers.append(nn.Linear(width, size))      # weights (size, width) and a bias (size)
            layers.append(nn.ReLU())                   # keeps positive numbers, turns negatives into zero
            if dropout > 0:
                layers.append(nn.Dropout(dropout))     # active in train mode, switched off in eval mode
            width = size
        layers.append(nn.Linear(width, out_features))  # one score per class
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Points in, class scores (logits) out."""
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters (every number that learning may change)."""
    return sum(p.numel() for p in model.parameters())


def make_optimizer(name: str, params, lr: float, weight_decay: float = 0.0, momentum: float = 0.9):
    """Pick the rule that turns gradients into parameter changes.

    sgd      - new parameter = old parameter - lr * gradient
    momentum - the same, but remembers the direction of recent steps (a rolling ball)
    adamw    - scales each parameter's step by how large its recent gradients were, plus clean weight decay
    """
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def warmup_cosine(warmup: int, total: int):
    """Learning-rate multiplier: climb in a straight line for `warmup` steps, then glide down like half a cosine wave."""
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def evaluate(model: nn.Module, X, y):
    """Loss and accuracy in eval mode (dropout off, no gradients) - 'sitting the exam'."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    model.train()
    return loss, acc


def train_module(train, val, model: nn.Module, optimizer: str = "adamw", lr: float = 0.01,
                 weight_decay: float = 0.0, steps: int = 500, warmup: int = 0, cosine: bool = False,
                 snapshot_steps=(), device: str = "cpu"):
    """Train `model` on the training points and watch the validation points after every step.

    Returns a dict with the history (loss, accuracy and learning rate per step), the model, the
    requested snapshots (copies of the parameters at chosen steps) and the training time in seconds.
    """
    (Xtr, ytr), (Xva, yva) = train, val
    Xtr, ytr, Xva, yva = (t.to(device) for t in (Xtr, ytr, Xva, yva))
    model.to(device).train()
    opt = make_optimizer(optimizer, model.parameters(), lr, weight_decay)
    scheduler = None
    if cosine or warmup:
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(warmup, steps) if cosine else (lambda s: min(1.0, (s + 1) / warmup)))
    history = {"step": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    snapshots = {}

    def record(step):
        tl, ta = evaluate(model, Xtr, ytr)
        vl, va = evaluate(model, Xva, yva)
        for key, value in zip(history, (step, tl, ta, vl, va, opt.param_groups[0]["lr"])):
            history[key].append(value)
        if step in snapshot_steps:
            snapshots[step] = copy.deepcopy(model.state_dict())

    start = time.perf_counter()
    record(0)
    for step in range(1, steps + 1):
        opt.zero_grad()                                # 1. clear the gradients left from the last round
        logits = model(Xtr)                            # 2. forward: predictions
        loss = F.cross_entropy(logits, ytr)            # 3. loss: how far from the targets?
        loss.backward()                                # 4. backward: a gradient for every parameter
        opt.step()                                     # 5. step: the optimizer changes every parameter
        if scheduler is not None:
            scheduler.step()                           # move to the next learning rate
        record(step)
    seconds = time.perf_counter() - start
    return {"history": history, "model": model, "snapshots": snapshots, "seconds": seconds}
