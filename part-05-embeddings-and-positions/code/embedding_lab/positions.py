"""positions.py - why order needs a code, and the sinusoidal position table (NumPy only).

What it does
    * sinusoidal_table(): the fixed position codes of the original Transformer paper (Vaswani et al., 2017).
    * self_attention(): one head of attention with plain matrices, to show that attention alone sees a bag of tokens.
    * bag_demo(): runs "dog bites man" and "man bites dog" through that attention, first without and then with position codes.
What it needs
    NumPy.
What to expect
    Without position codes the output row of every word is identical in both sentences; with position codes the rows differ.
"""
import numpy as np


def sinusoidal_table(n_positions, dim, base=10000.0):
    """Return an array (n_positions, dim). Pair i of every row holds sin(pos / base^(2i/dim)) and cos(pos / base^(2i/dim)).

    Low pairs turn fast (they tell neighbouring positions apart); high pairs turn slowly (they tell far positions apart).
    """
    pos = np.arange(n_positions)[:, None]
    pair = np.arange(dim // 2)[None, :]
    angle = pos / base ** (2 * pair / dim)
    table = np.zeros((n_positions, dim))
    table[:, 0::2] = np.sin(angle)
    table[:, 1::2] = np.cos(angle)
    return table


def softmax_rows(scores):
    """Softmax along every row, so each row adds up to 1."""
    shifted = np.exp(scores - scores.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def self_attention(x, wq, wk, wv):
    """One attention head: every row of x looks at every row of x and takes a weighted mix of their value vectors."""
    q, k, v = x @ wq, x @ wk, x @ wv
    weights = softmax_rows(q @ k.T / np.sqrt(k.shape[1]))      # row i: how much word i listens to each word
    return weights @ v, weights


def bag_demo(dim=8, seed=5):
    """Run the two sentences through attention without and with position codes; return the outputs for comparison.

    Returns a dict with the output rows (keyed by word) for both sentences and both settings.
    """
    rng = np.random.default_rng(seed)
    vocab = {w: rng.normal(size=dim) for w in ("dog", "bites", "man")}          # one made-up vector per word
    wq, wk, wv = (0.5 * rng.normal(size=(dim, dim)) for _ in range(3))        # moderate attention weights, fixed for both sentences, so attention stays spread out
    positions = sinusoidal_table(3, dim)
    results = {}
    for label, sentence in (("dog bites man", ["dog", "bites", "man"]), ("man bites dog", ["man", "bites", "dog"])):
        x = np.stack([vocab[w] for w in sentence])
        plain, _ = self_attention(x, wq, wk, wv)
        with_pos, _ = self_attention(x + positions, wq, wk, wv)
        results[label] = {"plain": dict(zip(sentence, plain)), "with_positions": dict(zip(sentence, with_pos))}
    return results
