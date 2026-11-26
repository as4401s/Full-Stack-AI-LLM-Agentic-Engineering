"""Tests for rope.py: rotation keeps length, and the score depends only on the distance between two positions."""
import numpy as np
import pytest

from embedding_lab import rope


def test_rotation_preserves_length():
    rng = np.random.default_rng(0)
    for dim in (2, 8, 64):
        x = rng.normal(size=dim)
        for position in (0, 1, 17, 1000, 40000):
            assert np.linalg.norm(rope.rotate(x, position)) == pytest.approx(np.linalg.norm(x), rel=1e-12)


def test_position_zero_changes_nothing():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(rope.rotate(x, 0), x)


def test_score_depends_only_on_the_distance():
    rng = np.random.default_rng(1)
    q, k = rng.normal(size=16), rng.normal(size=16)
    for m, n, shift in ((5, 2, 100), (30, 3, 777), (4, 40, 1000)):
        assert rope.score(q, k, m + shift, n + shift) == pytest.approx(rope.score(q, k, m, n), abs=1e-9)


def test_different_distances_give_different_scores():
    rng = np.random.default_rng(2)
    q, k = rng.normal(size=16), rng.normal(size=16)
    assert abs(rope.score(q, k, 10, 5) - rope.score(q, k, 10, 4)) > 1e-6


def test_one_pair_turns_by_the_expected_angle():
    # first pair turns one radian per position, so position 1 of [1, 0] is [cos 1, sin 1]
    out = rope.rotate(np.array([1.0, 0.0]), 1)
    assert out[0] == pytest.approx(np.cos(1.0)) and out[1] == pytest.approx(np.sin(1.0))


def test_position_interpolation_shrinks_positions():
    assert rope.scale_positions(6000, 4096, 8192) == pytest.approx(3000)


def test_ntk_base_grows_with_the_stretch():
    assert rope.ntk_scaled_base(10000.0, 64, 2.0) > 10000.0
    assert rope.ntk_scaled_base(10000.0, 64, 1.0) == pytest.approx(10000.0)


def test_half_pairing_keeps_length_and_relative_property():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=16), rng.normal(size=16)
    assert np.linalg.norm(rope.rotate(x, 123, pairing="half")) == pytest.approx(np.linalg.norm(x), rel=1e-12)
    a = float(np.dot(rope.rotate(x, 40, pairing="half"), rope.rotate(y, 37, pairing="half")))
    b = float(np.dot(rope.rotate(x, 540, pairing="half"), rope.rotate(y, 537, pairing="half")))
    assert a == pytest.approx(b, abs=1e-9)


def test_half_pairing_is_the_same_rotation_in_another_order():
    # list the numbers of the interleaved layout as (0, 2, 4, ...) then (1, 3, 5, ...) and the half layout gives the same rotated numbers
    x = np.random.default_rng(4).normal(size=8)
    order = np.r_[0:8:2, 1:8:2]
    assert np.allclose(rope.rotate(x[order], 17, pairing="half"), rope.rotate(x, 17)[order])
