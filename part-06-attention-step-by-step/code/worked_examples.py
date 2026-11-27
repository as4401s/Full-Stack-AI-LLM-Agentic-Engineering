"""worked_examples.py - reproduces every small number calculated by hand in lesson 6.

What it does
    Recomputes, with NumPy only, each calculation we do on a calculator in the video: one question against three labels (scores, divide by
    the square root of d, softmax, weighted sum), why the division is there, the full table where every token asks, the causal mask,
    two heads on a tiny sequence, how fast the score table grows with the length of the text, and the running softmax that lets a
    long row be handled in blocks.

What it needs
    NumPy.

What to expect
    Run:  python worked_examples.py
    Every line printed here is a number you can also get on a calculator. The saved output is in data/expected_output/worked_examples.txt.
"""
import math

import numpy as np

from attention_lab import numpy_attention as na

QUERY = np.array([[1.0, 0.0]])                                   # one token asking its question
KEYS = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])            # the labels on three earlier tokens
VALUES = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])          # the contents of those three tokens


def projections():
    """Where query, key and value come from: one token vector x is multiplied by three small learned tables."""
    print("== 1. Where query, key and value come from (x times three tables)")
    x = np.array([1.0, 2.0])
    w_q = np.array([[1.0, 0.0], [1.0, 1.0]])
    w_k = np.array([[0.0, 1.0], [1.0, 0.0]])
    w_v = np.array([[2.0, 0.0], [0.0, 2.0]])
    print(f"  token vector x = [1, 2]")
    print(f"  query = x times table W_q = [1*1 + 2*1, 1*0 + 2*1] = {(x @ w_q).astype(int).tolist()}")
    print(f"  key   = x times table W_k = [1*0 + 2*1, 1*1 + 2*0] = {(x @ w_k).astype(int).tolist()}")
    print(f"  value = x times table W_v = [1*2 + 2*0, 1*0 + 2*2] = {(x @ w_v).astype(int).tolist()}")
    print(f"  three tables of 2 x 2 = {3 * 4} learned numbers; GPT-2 small uses 768 x 768 per table = {768 * 768:,} numbers each")


def one_question():
    """Query [1, 0] against three keys: scores, scaling by the square root of d = 2, softmax, weighted sum of the values."""
    print("== 2. One question, three labels (d = 2)")
    scores = (QUERY @ KEYS.T)[0]
    print(f"  query q = [1, 0]; keys k1 = [1, 0], k2 = [0, 1], k3 = [1, 1]; values v1 = [2, 0], v2 = [0, 2], v3 = [1, 1]")
    print(f"  scores q.k = [1*1 + 0*0, 1*0 + 0*1, 1*1 + 0*1] = {scores.astype(int).tolist()}")
    scaled = scores / math.sqrt(2)
    print(f"  divide by sqrt(2) = {math.sqrt(2):.4f}: {[round(float(s), 4) for s in scaled]}")
    exps = np.exp(scaled)
    print(f"  exp of each: {[round(float(e), 4) for e in exps]}   sum = {exps.sum():.4f}")
    weights = exps / exps.sum()
    print(f"  weights = exp / sum = {[round(float(w), 4) for w in weights]}  -> {[round(float(w) * 100) for w in weights]} percent, total {weights.sum() * 100:.0f} percent")
    out, w2 = na.attention(QUERY, KEYS, VALUES)
    print(f"  weighted sum = 0.4011*[2, 0] + 0.1978*[0, 2] + 0.4011*[1, 1] = {[round(float(x), 4) for x in out[0]]}")
    rounded = 0.4 * VALUES[0] + 0.2 * VALUES[1] + 0.4 * VALUES[2]
    print(f"  with the rounded weights 0.40, 0.20, 0.40: {[round(float(x), 2) for x in rounded]}   (the same to two decimals: {[round(float(x), 2) for x in out[0]]})")
    assert np.allclose(w2[0], weights)


def why_divide():
    """Dot products of random vectors grow like the square root of d; dividing by it keeps the softmax from collapsing onto one token."""
    print("\n== 3. Why divide by the square root of d")
    rng = np.random.default_rng(0)
    print("  random vectors with ordinary-sized numbers (mean 0, spread 1), 20,000 pairs per row:")
    print("      d   spread of q.k   after dividing")
    for d in (4, 16, 64, 256):
        q, k = rng.normal(size=(20000, d)), rng.normal(size=(20000, d))
        raw = (q * k).sum(axis=1)
        print(f"  {d:5d}   {raw.std():13.2f}   {(raw / math.sqrt(d)).std():14.2f}")
    d, n_keys, rows = 64, 8, 10000
    biggest_raw, biggest_scaled = [], []
    for _ in range(rows):
        q, keys = rng.normal(size=d), rng.normal(size=(n_keys, d))
        raw = keys @ q
        biggest_raw.append(na.softmax(raw).max())
        biggest_scaled.append(na.softmax(raw / math.sqrt(d)).max())
    biggest_raw, biggest_scaled = np.array(biggest_raw), np.array(biggest_scaled)
    print(f"  d = 64, eight keys, {rows:,} random rows: the biggest weight in a row averages {biggest_raw.mean():.2f} without the division and {biggest_scaled.mean():.2f} with it")
    print(f"  rows whose biggest weight is above 0.95: {100 * (biggest_raw > 0.95).mean():.0f} percent without the division, {100 * (biggest_scaled > 0.95).mean():.0f} percent with it")
    one = np.random.default_rng(4)
    q, keys = one.normal(size=d), one.normal(size=(n_keys, d))
    raw = keys @ q
    print(f"  one example row, raw scores {[round(float(s), 1) for s in raw]}")
    print(f"    softmax of the raw scores    : biggest weight {na.softmax(raw).max():.3f}")
    print(f"    softmax after / sqrt(64) = 8 : biggest weight {na.softmax(raw / 8).max():.3f}   scaled scores {[round(float(s), 2) for s in raw / 8]}")


def full_table():
    """Every token asks and every token answers: a 3 x 3 table whose rows each add up to 1."""
    print("\n== 4. Every token asks: the full 3 x 3 table (queries = keys = [[1, 0], [0, 1], [1, 1]])")
    queries = KEYS.copy()
    scores = queries @ KEYS.T
    print(f"  scores q.k: {scores.astype(int).tolist()}")
    out, weights = na.attention(queries, KEYS, VALUES)
    for i, row in enumerate(weights):
        print(f"  row {i + 1}: weights {[round(float(w), 4) for w in row]}  = {[round(float(w) * 100) for w in row]} percent  (sum {row.sum():.2f})")
    print(f"  outputs: {[[round(float(x), 3) for x in r] for r in out]}")


def causal():
    """The same table when each token may only look at itself and earlier tokens."""
    print("\n== 5. The causal mask (the upper triangle is switched off, then every row is shared out again)")
    queries = KEYS.copy()
    scaled = queries @ KEYS.T / math.sqrt(2)
    mask = na.causal_mask(3)
    print(f"  switched off (True = future): {mask.astype(int).tolist()}")
    out, weights = na.attention(queries, KEYS, VALUES, causal=True)
    print(f"  row 1 sees token 1 only: weights {[round(float(w), 4) for w in weights[0]]}")
    e = np.exp(scaled[1, :2])
    print(f"  row 2 sees tokens 1 and 2: exp {[round(float(x), 4) for x in e]}, sum {e.sum():.4f}, weights {[round(float(x), 4) for x in e / e.sum()]}")
    print(f"  row 3 sees all three: weights {[round(float(w), 4) for w in weights[2]]}  (nothing to switch off)")
    print(f"  all rows add up to 1: {np.allclose(weights.sum(axis=1), 1)}")
    keys2, values2 = KEYS.copy(), VALUES.copy()
    keys2[2], values2[2] = [9, 9], [9, 9]
    queries2 = np.vstack([KEYS[:2], [[9, 9]]])
    out_open, _ = na.attention(queries, KEYS, VALUES, causal=False)
    out_open2, _ = na.attention(queries2, keys2, values2, causal=False)
    out2, _ = na.attention(queries2, keys2, values2, causal=True)
    print("  replace token 3 by [9, 9] and look at the outputs of tokens 1 and 2:")
    print(f"    with the mask    : tokens 1 and 2 keep their outputs: {np.allclose(out2[:2], out[:2])}")
    print(f"    without the mask : token 1 changes from {[round(float(x), 3) for x in out_open[0]]} to {[round(float(x), 3) for x in out_open2[0]]}  (it peeked at the future)")


def two_heads():
    """A sequence of three tokens with four numbers each, two heads of two numbers each."""
    print("\n== 6. Two heads (3 tokens, d_model = 4, each head works on 2 numbers)")
    x = np.array([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    eye = np.eye(4)
    out, maps = na.multi_head(x, eye, eye, eye, eye, n_heads=2)
    print(f"  x = {x.astype(int).tolist()}   (rows are tokens)")
    print(f"  head 1 reads columns 1-2, head 2 reads columns 3-4; the weight matrices here are the identity so the split is easy to see")
    for h, m in enumerate(maps, 1):
        print(f"  head {h} weights: {[[round(float(w), 2) for w in row] for row in m]}")
    print(f"  head 1 row 3 = {[round(float(w), 2) for w in maps[0][2]]}  versus head 2 row 3 = {[round(float(w), 2) for w in maps[1][2]]}  (different readers, different focus)")
    print(f"  glued together and mixed by w_o: output shape {out.shape}, first row {[round(float(v), 3) for v in out[0]]}")
    print(f"  parameters: four weight matrices of 4 x 4 = {4 * 16}; the same for one head or many heads (the split costs nothing extra)")


def growth():
    """The score table has n x n entries: ten times the text means a hundred times the entries."""
    print("\n== 7. The n x n score table")
    for n in (1_000, 100_000):
        entries = n * n
        size = f"{entries * 2 / 1e6:,.0f} MB" if entries < 10 ** 9 else f"{entries * 2 / 1e9:,.0f} GB"
        print(f"  n = {n:>7,}: {n:,} x {n:,} = {entries:,} scores per head per layer; at 2 bytes each that is {size}")
    print(f"  ratio: {100_000 ** 2 // 1_000 ** 2:,} times more entries for 100 times more text")
    print(f"  with 32 heads in one layer at n = 100,000 and 2 bytes: {32 * 100_000 ** 2 * 2 / 1e9:,.0f} GB")


def running_softmax():
    """One row of four scores handled in two blocks of two, keeping a running maximum m, a running sum l and a running weighted sum."""
    print("\n== 8. The running softmax (what the tiled version does for one row)")
    s = np.array([1.0, 0.0, 1.0, 2.0])
    v = np.array([2.0, 0.0, 1.0, 3.0])
    print(f"  scores {s.tolist()}, values {v.tolist()}")
    exact = na.softmax(s)
    print(f"  all at once: weights {[round(float(w), 4) for w in exact]}, output {float(exact @ v):.4f}")
    m, l, acc = -math.inf, 0.0, 0.0
    for b, (lo, hi) in enumerate(((0, 2), (2, 4)), 1):
        block = s[lo:hi]
        m_new = max(m, float(block.max()))
        shrink = 0.0 if m == -math.inf else math.exp(m - m_new)
        p = np.exp(block - m_new)
        l = l * shrink + float(p.sum())
        acc = acc * shrink + float(p @ v[lo:hi])
        note = "nothing to shrink yet" if b == 1 else f"shrink the old totals by exp({m:g} - {m_new:g}) = {shrink:.4f}"
        print(f"  block {b}: scores {block.tolist()}  new maximum m = {m_new:g}  {note}  running sum l = {l:.4f}  running weighted sum = {acc:.4f}")
        m = m_new
    print(f"  at the end: output = weighted sum / l = {acc:.4f} / {l:.4f} = {acc / l:.4f}  (the same as all at once)")


def main():
    projections()
    one_question()
    why_divide()
    full_table()
    causal()
    two_heads()
    growth()
    running_softmax()


if __name__ == "__main__":
    main()
