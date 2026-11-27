"""Demo 8 - the tiled (FlashAttention-style) way gives the same answer while holding a much smaller piece of the score table.

What it does: runs plain attention and attention_tiled() on random numbers (1,024 tokens, 64 numbers per head) with block sizes 64, 128 and 256,
with and without the causal mask, and prints the biggest difference and the largest piece of the score table that was ever held at once.
What it needs: NumPy only.
What to expect: differences around 1e-15; the largest piece is n x block instead of n x n.
Run it: uv run python -m demos.d08_tiled_check
"""
import numpy as np

from attention_lab import numpy_attention as na
from attention_lab.report import save_results


def main():
    rng = np.random.default_rng(0)
    n, d = 1024, 64
    q, k, v = rng.normal(size=(n, d)), rng.normal(size=(n, d)), rng.normal(size=(n, d))
    results = {}
    print(f"n = {n} tokens, {d} numbers per head; plain attention holds {n * n:,} scores at once")
    for causal in (False, True):
        plain, _ = na.attention(q, k, v, causal=causal)
        for block in (64, 128, 256):
            tiled, held = na.attention_tiled(q, k, v, block=block, causal=causal)
            diff = float(np.abs(plain - tiled).max())
            print(f"  causal={causal!s:5} block {block:3d}: biggest difference {diff:.1e}   largest piece held {held:,} scores ({held / (n * n) * 100:.1f} percent of the full table)")
            results[f"causal_{causal}_block_{block}"] = {"max_abs_diff": diff, "held": held}
    save_results("d08_tiled_check", results, digits=12)
