"""attention.py - scaled dot-product attention in PyTorch, the function shown in lesson 6.

What it does
    Takes queries, keys and values for a batch of sequences and several heads at once, and returns the attended values together with the weights.
    The three steps are the ones we did by hand: scores = q . k, divide by the square root of the head size, softmax, then a weighted sum of the values.

What it needs
    PyTorch (torch.nn.functional.scaled_dot_product_attention is used only in the checks, never inside this function).

What to expect
    attention(q, k, v) returns (output, weights). For input of shape (batch, heads, seq, head_dim) the output has the same shape and the
    weights have shape (batch, heads, seq, seq); every row of weights adds up to 1, and with causal=True everything above the diagonal is 0.
"""
import math

import torch


def attention(q, k, v, causal: bool = True):
    """Scaled dot-product attention.

    q, k, v: (batch, heads, seq, head_dim). Returns (output, weights).
    causal=True lets each token listen only to itself and earlier tokens (what text generation needs).
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))      # (batch, heads, seq_q, seq_k): question-label matches, scaled
    if causal:
        seq_q, seq_k = q.size(-2), k.size(-2)
        future = torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device), diagonal=1)   # True above the diagonal
        scores = scores.masked_fill(future, float("-inf"))        # listen only to the past: minus infinity becomes weight 0
    weights = torch.softmax(scores, dim=-1)                       # each row sums to 1
    return weights @ v, weights                                   # weighted sum of the values
