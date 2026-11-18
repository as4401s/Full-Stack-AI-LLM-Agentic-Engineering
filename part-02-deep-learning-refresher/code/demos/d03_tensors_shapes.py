"""Demo 3 - tensors, shapes and the three errors beginners meet most often.

What it does: shows the shape of every tensor on the way through a layer, broadcasting of a bias, and
catches three real shape mistakes so we can read their messages calmly.
What it needs: torch.
What to expect: (4, 2) @ (2, 16) gives (4, 16); the three mistakes raise errors that name the shapes.
Run it: uv run python -m demos.d03_tensors_shapes
"""
import torch
import torch.nn.functional as F
from torch import nn

from dl_refresher.report import run_and_save


def first_line(error: Exception) -> str:
    """The first line of an error message, shortened so it stays readable."""
    return str(error).strip().splitlines()[0][:110]


def main():
    torch.manual_seed(0)
    points = torch.randn(4, 2)                           # 4 points, 2 numbers each: shape (batch, features)
    w1 = torch.randn(2, 16)                              # first layer weights: (features in, units out)
    b1 = torch.zeros(16)                                 # one offset per hidden unit
    hidden = points @ w1 + b1                            # (4, 2) @ (2, 16) -> (4, 16); b1 is added to every row
    print("points      ", tuple(points.shape))
    print("w1          ", tuple(w1.shape))
    print("points @ w1 ", tuple((points @ w1).shape))
    print("b1          ", tuple(b1.shape), "-> broadcast across 4 rows")
    print("hidden      ", tuple(hidden.shape))
    print()
    print("three mistakes, caught on purpose")
    try:
        points @ torch.randn(3, 5)                       # the inner sizes 2 and 3 differ
    except RuntimeError as error:
        print("1) wrong inner size   :", type(error).__name__, "-", first_line(error))
    try:
        nn.Linear(2, 16)(torch.randn(4, 3))              # the layer expects 2 features, got 3
    except RuntimeError as error:
        print("2) wrong feature count:", type(error).__name__, "-", first_line(error))
    try:
        F.cross_entropy(torch.randn(4, 2), torch.tensor([0, 1, 1]))   # 4 rows of scores but 3 labels
    except (RuntimeError, ValueError) as error:
        print("3) scores vs labels   :", type(error).__name__, "-", first_line(error))


if __name__ == "__main__":
    run_and_save("d03_tensors_shapes", main)
