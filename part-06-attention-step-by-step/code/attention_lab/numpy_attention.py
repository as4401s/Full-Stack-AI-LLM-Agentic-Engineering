"""numpy_attention.py - attention written with NumPy only, so every number can be checked on a calculator.

What it does
    Holds the small building blocks of lesson 6: a stable softmax, scaled dot-product attention for one sequence, the causal mask,
    multi-head attention for one sequence, and a tiled version that never builds the full score table (the idea behind FlashAttention).

What it needs
    NumPy only (no PyTorch), so worked_examples.py runs anywhere.

What to expect
    Every function returns plain NumPy arrays. The shapes are written in the docstrings: n = number of tokens, d = numbers per token.
"""
import math

import numpy as np


def softmax(scores, axis=-1):
    """Turn scores into weights that are positive and add up to 1 along `axis`.

    We subtract the largest score first: the result is the same and large scores cannot overflow.
    """
    shifted = scores - np.max(scores, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def causal_mask(n):
    """An n x n table that is True above the diagonal: the positions a token must not look at (its future)."""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def attention(q, k, v, causal=False):
    """Scaled dot-product attention for one sequence.

    q: (n_queries, d)   one question per token
    k: (n_keys, d)      one label per token
    v: (n_keys, d_v)    one content per token
    Returns (output, weights): output is (n_queries, d_v); weights is (n_queries, n_keys) and every row adds up to 1.
    """
    d = q.shape[-1]
    scores = q @ k.T                                   # how well each question matches each label
    scaled = scores / math.sqrt(d)                     # divide by the square root of d so the scores stay a friendly size
    if causal:
        scaled = np.where(causal_mask(scaled.shape[0]), -np.inf, scaled)   # the future gets weight 0 after the softmax
    weights = softmax(scaled, axis=-1)                 # each row: how much to listen to each token
    return weights @ v, weights                        # the weighted sum of the contents


def multi_head(x, w_q, w_k, w_v, w_o, n_heads, causal=False):
    """Multi-head attention for one sequence.

    x: (n, d_model); w_q, w_k, w_v, w_o: (d_model, d_model); n_heads must divide d_model.
    Each head works on its own slice of d_model / n_heads numbers; the heads' outputs are glued together and mixed by w_o.
    Returns (output (n, d_model), list of weights, one (n, n) table per head).
    """
    n, d_model = x.shape
    d_head = d_model // n_heads
    q, k, v = x @ w_q, x @ w_k, x @ w_v                # one projection each, then split into heads by column slices
    outputs, maps = [], []
    for h in range(n_heads):
        cols = slice(h * d_head, (h + 1) * d_head)
        out_h, weights_h = attention(q[:, cols], k[:, cols], v[:, cols], causal=causal)
        outputs.append(out_h)
        maps.append(weights_h)
    return np.concatenate(outputs, axis=-1) @ w_o, maps


def attention_tiled(q, k, v, block=128, causal=False):
    """The same result as `attention`, computed block by block with a running (online) softmax.

    Only an (n_queries x block) piece of the score table exists at any moment. For every query row we keep three running values:
    m = the largest score so far, l = the sum of exp(score - m) so far, acc = the weighted sum of values so far (scaled by exp(-m)).
    When a new block brings a larger maximum, the old l and acc are scaled down to match.
    Returns (output, largest_piece) where largest_piece is the biggest number of scores held at once.
    """
    n_q, d = q.shape
    n_k = k.shape[0]
    m = np.full(n_q, -np.inf)
    l = np.zeros(n_q)
    acc = np.zeros((n_q, v.shape[1]))
    held = 0
    for start in range(0, n_k, block):
        stop = min(start + block, n_k)
        s = q @ k[start:stop].T / math.sqrt(d)         # scores of every query against this block of keys
        if causal:
            future = np.arange(start, stop)[None, :] > np.arange(n_q)[:, None]
            s = np.where(future, -np.inf, s)
        held = max(held, s.size)
        m_new = np.maximum(m, s.max(axis=1))
        scale_old = np.where(np.isfinite(m), np.exp(m - m_new), 0.0)   # shrink what we had if the maximum grew
        p = np.exp(s - m_new[:, None])
        l = l * scale_old + p.sum(axis=1)
        acc = acc * scale_old[:, None] + p @ v[start:stop]
        m = m_new
    return acc / l[:, None], held
