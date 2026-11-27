"""Demo 2 - check our attention() against PyTorch's built-in kernel.

What it does: makes random queries, keys and values (batch 2, 4 heads, 16 tokens, 8 numbers per head), runs our attention() and
torch.nn.functional.scaled_dot_product_attention on them with and without the causal mask, and prints the biggest difference.
What it needs: torch.
What to expect: differences of about 1e-7 (the size of float32 rounding), rows that add up to 1, and zeros above the diagonal.
Run it: uv run python -m demos.d02_check_builtin
"""
import torch
import torch.nn.functional as F

from attention_lab.attention import attention
from attention_lab.report import save_results


def main():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
    results = {}
    for causal in (False, True):
        ours, weights = attention(q, k, v, causal=causal)
        built_in = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        diff = (ours - built_in).abs().max().item()
        row_sum_gap = (weights.sum(-1) - 1).abs().max().item()
        above = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
        above_max = weights[..., above].max().item()
        print(f"causal={causal!s:5}: biggest difference from the built-in kernel {diff:.1e}   rows add up to 1 within {row_sum_gap:.1e}   largest weight above the diagonal {above_max:.1f}")
        print(f"             same answer within 1e-5: {torch.allclose(ours, built_in, atol=1e-5)}")
        results[f"causal_{causal}"] = {"max_abs_diff": diff, "row_sum_gap": row_sum_gap}
    print(f"\nshapes: q {tuple(q.shape)} -> output {tuple(ours.shape)}, weights {tuple(weights.shape)}")
    save_results("d02_check_builtin", results, digits=10)
