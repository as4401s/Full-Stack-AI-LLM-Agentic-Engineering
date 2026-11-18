"""Tests for the lesson's small hand-checkable examples: gradients, gradient descent, shapes, softmax, precision."""
import math

import pytest
import torch
import torch.nn.functional as F


def test_backprop_numbers_match_hand_calculation():
    w1, b1, w2, b2 = (torch.tensor(v, requires_grad=True) for v in (1.0, 1.0, 1.0, 0.0))
    prediction = w2 * torch.relu(w1 * 2.0 + b1) + b2
    ((prediction - 6.0) ** 2).backward()
    assert (w1.grad.item(), w2.grad.item(), b1.grad.item(), b2.grad.item()) == (-12.0, -18.0, -6.0, -6.0)


def test_gradient_descent_sequence():
    w = torch.tensor(0.0, requires_grad=True)
    path = []
    for _ in range(3):
        ((w - 2.0) ** 2).backward()
        with torch.no_grad():
            w -= 0.1 * w.grad
            w.grad = None
        path.append(round(w.item(), 3))
    assert path == [0.4, 0.72, 0.976]


def test_large_learning_rate_overshoots():
    w = 0.0
    for _ in range(2):
        w -= 1.1 * 2 * (w - 2.0)
    assert w < 0                                       # jumped past the target and back below zero


def test_shape_mistakes_are_caught():
    with pytest.raises(RuntimeError):
        torch.randn(4, 2) @ torch.randn(3, 5)


def test_softmax_sums_to_one_and_matches_hand_numbers():
    probs = F.softmax(torch.tensor([2.0, 1.0, 0.1]), dim=0)
    assert abs(probs.sum().item() - 1.0) < 1e-6
    assert [round(v, 3) for v in probs.tolist()] == [0.659, 0.242, 0.099]
    assert abs(-math.log(probs[0].item()) - 0.417) < 1e-3


def test_bfloat16_rounds_pi():
    assert torch.tensor(math.pi, dtype=torch.bfloat16).item() == 3.140625
