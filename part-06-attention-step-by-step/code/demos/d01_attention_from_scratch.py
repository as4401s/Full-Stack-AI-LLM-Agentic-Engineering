"""Demo 1 - attention from scratch in PyTorch, on the three-token example we calculate by hand.

What it does: runs attention() on query [1, 0] against the keys [1, 0], [0, 1], [1, 1] (values [2, 0], [0, 2], [1, 1]), then on the full 3 x 3 table,
then with the causal mask, and compares everything with the NumPy numbers of worked_examples.py.
What it needs: torch and NumPy.
What to expect: weights 0.4011, 0.1978, 0.4011 and output [1.2033, 0.7967]; the masked table has zeros above the diagonal and every row adds up to 1.
Run it: uv run python -m demos.d01_attention_from_scratch
"""
import numpy as np
import torch

from attention_lab import numpy_attention as na
from attention_lab.attention import attention
from attention_lab.report import save_results

KEYS = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
VALUES = [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]


def shaped(rows):
    """A list of rows -> a tensor of shape (batch 1, heads 1, seq, dim), the shape attention() expects."""
    return torch.tensor(rows).unsqueeze(0).unsqueeze(0)


def main():
    k, v = shaped(KEYS), shaped(VALUES)
    q = shaped([[1.0, 0.0]])
    out, w = attention(q, k, v, causal=False)
    print(f"one query [1, 0]: weights {[round(x, 4) for x in w[0, 0, 0].tolist()]}  output {[round(x, 4) for x in out[0, 0, 0].tolist()]}")
    ref_out, ref_w = na.attention(np.array([[1.0, 0.0]]), np.array(KEYS), np.array(VALUES))
    print(f"  same numbers as NumPy: {np.allclose(w[0, 0].numpy(), ref_w) and np.allclose(out[0, 0].numpy(), ref_out)}")

    q_all = shaped(KEYS)
    out_full, w_full = attention(q_all, k, v, causal=False)
    out_mask, w_mask = attention(q_all, k, v, causal=True)
    print("\nfull table (every token asks):")
    for i, row in enumerate(w_full[0, 0].tolist(), 1):
        print(f"  row {i}: {[round(x, 4) for x in row]}  sum {sum(row):.4f}")
    print("\ncausal table (the future is switched off):")
    for i, row in enumerate(w_mask[0, 0].tolist(), 1):
        print(f"  row {i}: {[round(x, 4) for x in row]}  sum {sum(row):.4f}")
    upper = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    print(f"\nlargest weight above the diagonal in the causal table: {w_mask[0, 0][upper].max().item():.1f}")
    print(f"largest distance from 1 of any row sum: {(w_mask.sum(-1) - 1).abs().max().item():.1e}")
    save_results("d01_attention_from_scratch", {"weights_full": w_full[0, 0], "weights_causal": w_mask[0, 0], "output_one_query": out[0, 0, 0]})
