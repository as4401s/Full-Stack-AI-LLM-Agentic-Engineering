"""sampler.py - choose the next word from a model's scores.

What it does
    A language model hands back one score (a logit) for every word. This file turns scores into
    probabilities with softmax and temperature, optionally keeps only the top-k words or the smallest
    group of words that covers probability p (top-p), and then picks one word: the best one (greedy) or a
    weighted random one.

What it needs
    NumPy. The same lines work on a PyTorch model's scores: call `logits.detach().numpy()` first.

What to expect
    For the scores [2.0, 1.0, 0.1]: temperature 0.5 gives 0.864, 0.117, 0.019; temperature 2 gives
    0.502, 0.304, 0.194; top-k with k=2 gives 0.731, 0.269.
"""
import numpy as np


def softmax(logits, temperature=1.0):
    """Turn scores into probabilities that add up to 1. A low temperature sharpens, a high one flattens."""
    scaled = np.asarray(logits, dtype=float) / temperature
    scaled = scaled - scaled.max()          # subtracting the biggest score keeps exp() from overflowing; the result is identical
    e = np.exp(scaled)
    return e / e.sum()


def top_k_filter(probs, k):
    """Keep the k most probable words, set the rest to zero, and rescale so the kept ones add up to 1."""
    probs = np.asarray(probs, dtype=float)
    keep = np.argsort(-probs, kind="stable")[:k]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def top_p_filter(probs, p):
    """Keep the smallest group of most probable words whose probabilities add up to at least p; rescale."""
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs, kind="stable")
    before = np.cumsum(probs[order]) - probs[order]   # probability already used up before each word
    keep = order[before < p]                           # a word stays while the words ahead of it have not reached p yet
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    """Pick the index of the next word.

    temperature 0 means greedy: always the highest score. Otherwise apply temperature, then top-k, then top-p,
    and draw one index at random in proportion to the remaining probabilities.
    """
    logits = np.asarray(logits, dtype=float)
    if temperature == 0:
        return int(np.argmax(logits))
    probs = softmax(logits, temperature)
    if top_k is not None:
        probs = top_k_filter(probs, top_k)
    if top_p is not None:
        probs = top_p_filter(probs, top_p)
    rng = rng if rng is not None else np.random.default_rng()
    return int(rng.choice(len(probs), p=probs))
