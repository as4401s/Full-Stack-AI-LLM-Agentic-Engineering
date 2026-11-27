"""Tests for numpy_attention.py and worked_examples.py (NumPy only)."""
import contextlib
import io

import numpy as np

import worked_examples
from attention_lab import numpy_attention as na

KEYS = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
VALUES = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])


def test_softmax_rows_add_up_to_one_and_survive_large_scores():
    w = na.softmax(np.array([[1000.0, 1001.0], [0.0, 0.0]]))
    assert np.allclose(w.sum(axis=1), 1) and np.all(np.isfinite(w))


def test_hand_example_weights_and_output():
    out, w = na.attention(np.array([[1.0, 0.0]]), KEYS, VALUES)
    assert np.allclose(w[0], [0.40111209, 0.19777581, 0.40111209]) and np.allclose(out[0], [1.2033, 0.7967], atol=1e-4)


def test_causal_table_is_lower_triangular_and_rows_add_up_to_one():
    _, w = na.attention(KEYS, KEYS, VALUES, causal=True)
    assert np.all(w[np.triu_indices(3, k=1)] == 0) and np.allclose(w.sum(axis=1), 1)
    assert np.allclose(w[1], [0.3302, 0.6698, 0.0], atol=1e-4)


def test_causal_output_ignores_the_future_and_open_output_does_not():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(6, 4)) for _ in range(3))
    k2, v2 = k.copy(), v.copy()
    k2[5], v2[5] = 9.0, 9.0
    assert np.allclose(na.attention(q, k, v, causal=True)[0][:5], na.attention(q, k2, v2, causal=True)[0][:5])
    assert not np.allclose(na.attention(q, k, v)[0][:5], na.attention(q, k2, v2)[0][:5])


def test_tiled_equals_plain_with_and_without_the_mask():
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(50, 16)) for _ in range(3))
    for causal in (False, True):
        plain, _ = na.attention(q, k, v, causal=causal)
        tiled, held = na.attention_tiled(q, k, v, block=7, causal=causal)
        assert np.allclose(plain, tiled, atol=1e-12) and held == 50 * 7


def test_multi_head_shapes_and_two_different_maps():
    x = np.array([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    out, maps = na.multi_head(x, np.eye(4), np.eye(4), np.eye(4), np.eye(4), n_heads=2)
    assert out.shape == (3, 4) and len(maps) == 2 and not np.allclose(maps[0], maps[1])


def test_worked_examples_print_the_video_numbers():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        worked_examples.main()
    out = buffer.getvalue()
    for expected in ("query = x times table W_q = [1*1 + 2*1, 1*0 + 2*1] = [3, 2]", "value = x times table W_v = [1*2 + 2*0, 1*0 + 2*2] = [2, 4]",
                     "768 x 768 per table = 589,824 numbers each", "scores q.k = [1*1 + 0*0, 1*0 + 0*1, 1*1 + 0*1] = [1, 0, 1]", "divide by sqrt(2) = 1.4142: [0.7071, 0.0, 0.7071]",
                     "weights = exp / sum = [0.4011, 0.1978, 0.4011]  -> [40, 20, 40] percent, total 100 percent",
                     "[1.2033, 0.7967]", "row 3: weights [0.2483, 0.2483, 0.5035]", "weights [0.3302, 0.6698]",
                     "n = 100,000: 100,000 x 100,000 = 10,000,000,000 scores", "= 4.1036 / 1.8711 = 2.1932"):
        assert expected in out, expected
