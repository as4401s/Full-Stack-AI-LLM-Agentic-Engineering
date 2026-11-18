"""Tests for the two-spirals data: same seed gives the same points, shapes and balance are right."""
import torch

from dl_refresher.data import make_datasets, two_spirals


def test_same_seed_gives_same_points():
    a, _ = two_spirals(50, seed=3)
    b, _ = two_spirals(50, seed=3)
    assert torch.equal(a, b)


def test_shapes_and_balance():
    X, y = two_spirals(40)
    assert X.shape == (80, 2) and y.shape == (80,)
    assert y.sum().item() == 40                      # half of the labels are 1


def test_train_and_validation_differ():
    (Xtr, _), (Xva, _) = make_datasets(n_train=50, n_val=50)
    assert not torch.equal(Xtr, Xva)
