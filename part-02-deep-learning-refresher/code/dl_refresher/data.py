"""data.py - the two-spirals data set.

What it does: draws two curved arms that twist around the centre, one for class 0 and one for class 1.
Every point is a pair of numbers (x, y) and a label (0 or 1). Only a curved boundary separates the arms,
so the network has to learn a curve, which makes it a nice small test for learning.
What it needs: numpy and torch.
What to expect: the same points every time you use the same seed.
"""
import numpy as np
import torch


def two_spirals(n_per_class: int, noise: float = 0.08, seed: int = 0, turns: float = 1.0):
    """Return (X, y): X has shape (2 * n_per_class, 2) and y has shape (2 * n_per_class,).

    The first arm starts near the centre and winds outwards; the second arm is the first one turned
    half a circle. `noise` blurs every point a little, so the groups touch each other a bit near the centre.
    """
    rng = np.random.default_rng(seed)                      # a seeded random generator: same seed, same points
    radius = rng.uniform(0.1, 1.0, n_per_class)            # distance of each point from the centre
    angle = turns * 2 * np.pi * radius                     # the further out, the more it has turned
    arm0 = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    arm1 = -arm0                                           # the other arm: the same curve turned by half a circle
    points = np.concatenate([arm0, arm1]) + rng.normal(0.0, noise, (2 * n_per_class, 2))
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    order = rng.permutation(len(labels))                   # mix the two classes together
    X = torch.tensor(points[order], dtype=torch.float32)   # float32 numbers: what the network works with
    y = torch.tensor(labels[order], dtype=torch.long)      # whole numbers (class ids): what the loss expects
    return X, y


def make_datasets(n_train: int = 200, n_val: int = 200, noise: float = 0.08, turns: float = 1.0):
    """Two independent draws from the same shape: one to study from (train) and one to sit the exam on (validation)."""
    train = two_spirals(n_train, noise=noise, seed=0, turns=turns)
    val = two_spirals(n_val, noise=noise, seed=1, turns=turns)
    return train, val
