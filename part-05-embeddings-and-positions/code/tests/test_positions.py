"""Tests for positions.py: the sinusoidal table and the bag-of-tokens demonstration."""
import numpy as np
import pytest

from embedding_lab import positions


def test_table_shape_and_first_row():
    table = positions.sinusoidal_table(10, 8)
    assert table.shape == (10, 8)
    assert np.allclose(table[0], [0, 1, 0, 1, 0, 1, 0, 1])


def test_every_position_has_its_own_row():
    table = positions.sinusoidal_table(200, 16)
    assert len({tuple(np.round(r, 9)) for r in table}) == 200


def test_dot_product_depends_only_on_the_distance():
    table = positions.sinusoidal_table(60, 8)
    first = float(table[0] @ table[3])
    for p in range(1, 50):
        assert float(table[p] @ table[p + 3]) == pytest.approx(first)


def test_attention_alone_sees_a_bag_of_tokens():
    result = positions.bag_demo()
    for word in ("dog", "bites", "man"):
        assert np.allclose(result["dog bites man"]["plain"][word], result["man bites dog"]["plain"][word])


def test_position_codes_make_the_sentences_different():
    result = positions.bag_demo()
    assert not np.allclose(result["dog bites man"]["with_positions"]["dog"], result["man bites dog"]["with_positions"]["dog"])


def test_attention_rows_add_up_to_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 6))
    _, weights = positions.self_attention(x, *(rng.normal(size=(6, 6)) for _ in range(3)))
    assert np.allclose(weights.sum(axis=1), 1.0)
