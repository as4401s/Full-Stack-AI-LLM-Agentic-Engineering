"""rope.py - rotary position embeddings (RoPE) with NumPy: turn each pair of numbers by an angle that grows with position.

What it does
    * rotate(): turns every pair of numbers (x0, x1), (x2, x3), ... of a query or key vector by position * theta_i, where theta_i
      is slower for later pairs. Pair i uses theta_i = base^(-2i/dim), the same frequencies as the sinusoidal table.
    * scale_positions() and ntk_scaled_base(): two simple ways to stretch a trained window (position interpolation, Chen et al. 2023,
      and an NTK-style change of base) so that longer texts fall inside angles the model has already seen.
What it needs
    NumPy.
What to expect
    * rotation keeps the length of a vector (the tests check it);
    * the dot product of a rotated query and a rotated key depends only on how far apart the two positions are.
    Note: by default this file pairs neighbouring numbers (0 with 1, 2 with 3). Several published models (Llama, Qwen) pair the first half
    with the second half instead; pairing="half" does that, and demo 13 checks it against Qwen's own code.
"""
import numpy as np


def rope_frequencies(dim, base=10000.0):
    """theta_i for i = 0 .. dim/2 - 1: how many radians pair i turns per position (fast first, slow last)."""
    return base ** (-np.arange(0, dim, 2) / dim)


def rotate(x, position, base=10000.0, pairing="interleaved"):
    """Rotate x (last axis = dim, even) by `position` steps. position can be a number, or an array matching x's leading axes.

    pairing="interleaved": pair (0, 1), (2, 3), ... as in the RoFormer paper.
    pairing="half": pair (0, dim/2), (1, dim/2 + 1), ... as the Hugging Face Llama and Qwen code does (the same rotation, other order).
    """
    x = np.asarray(x, dtype=float)
    angle = np.asarray(position, dtype=float)[..., None] * rope_frequencies(x.shape[-1], base)   # one angle per pair
    cos, sin = np.cos(angle), np.sin(angle)
    out = np.empty_like(x)
    if pairing == "interleaved":
        out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    else:
        half = x.shape[-1] // 2
        out[..., :half] = x[..., :half] * cos - x[..., half:] * sin
        out[..., half:] = x[..., :half] * sin + x[..., half:] * cos
    return out


def score(q, k, q_position, k_position, base=10000.0):
    """Attention score numerator: dot product of the rotated query and the rotated key."""
    return float(np.dot(rotate(q, q_position, base), rotate(k, k_position, base)))


def scale_positions(position, trained_length, new_length):
    """Position interpolation: shrink positions by trained_length / new_length so a longer text uses angles seen in training."""
    return np.asarray(position, dtype=float) * trained_length / new_length


def ntk_scaled_base(base, dim, scale):
    """NTK-style stretch: raise the base so the slowest pairs turn even more slowly while the fastest barely change."""
    return base * scale ** (dim / (dim - 2))
