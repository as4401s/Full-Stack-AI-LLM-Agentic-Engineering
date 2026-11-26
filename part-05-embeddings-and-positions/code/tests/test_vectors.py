"""Tests for vectors.py: the hand example from the video and the basic properties of cosine."""
import numpy as np
import pytest

from embedding_lab import vectors


def test_hand_example_gives_point_eight():
    assert vectors.dot([1, 2, 0], [2, 1, 0]) == 4
    assert vectors.cosine([1, 2, 0], [2, 1, 0]) == pytest.approx(0.8)


def test_cosine_of_a_vector_with_itself_is_one():
    assert vectors.cosine([3, -1, 2], [3, -1, 2]) == pytest.approx(1.0)


def test_cosine_ignores_length_but_distance_does_not():
    a = np.array([1.0, 2.0, 0.0])
    assert vectors.cosine(a, 2 * a) == pytest.approx(1.0)
    assert vectors.euclidean(a, 2 * a) > 2


def test_right_angle_and_opposite():
    assert vectors.cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert vectors.cosine([1, 2], [-1, -2]) == pytest.approx(-1.0)


def test_nearest_skips_excluded_words():
    words = ["a", "b", "c"]
    index = {w: i for i, w in enumerate(words)}
    matrix = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    assert vectors.nearest("a", words, index, matrix, k=1)[0][0] == "b"
    assert vectors.nearest("a", words, index, matrix, k=1, exclude=("b",))[0][0] == "c"
