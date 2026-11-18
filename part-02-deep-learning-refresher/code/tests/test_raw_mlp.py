"""Tests for the raw-tensor network: shapes, hand-written loss and a short training run."""
import torch
import torch.nn.functional as F

from dl_refresher.data import make_datasets
from dl_refresher.raw_mlp import forward, init_params, softmax_cross_entropy, train_raw


def test_forward_shape():
    params = init_params(hidden=16)
    assert forward(params, torch.randn(5, 2)).shape == (5, 2)


def test_hand_written_loss_matches_pytorch():
    logits = torch.randn(7, 2)
    labels = torch.randint(0, 2, (7,))
    assert torch.allclose(softmax_cross_entropy(logits, labels), F.cross_entropy(logits, labels), atol=1e-6)


def test_training_lowers_loss():
    train, val = make_datasets()
    out = train_raw(train, val, steps=200, lr=1.0)
    history = out["history"]
    assert history["train_loss"][-1] < 0.8 * history["train_loss"][0]
    assert history["train_acc"][-1] >= history["train_acc"][0]
