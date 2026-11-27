"""Demo 3 - multi-head attention as a module, checked against torch.nn.MultiheadAttention.

What it does: builds our MultiHeadAttention(32 numbers, 4 heads), copies its weights into PyTorch's own nn.MultiheadAttention, feeds both the same
input with the causal mask, compares outputs and attention weights, and counts parameters for 1, 2, 4 and 8 heads.
What it needs: torch.
What to expect: differences of about 1e-7; shapes (2, 10, 32) for the output and (2, 4, 10, 10) for the weights; the same parameter count for every number of heads.
Run it: uv run python -m demos.d03_multihead_module
"""
import torch
from torch import nn

from attention_lab.multihead import MultiHeadAttention
from attention_lab.report import save_results


def main():
    torch.manual_seed(0)
    d_model, n_heads, batch, seq = 32, 4, 2, 10
    ours = MultiHeadAttention(d_model, n_heads).eval()
    x = torch.randn(batch, seq, d_model)
    out, weights = ours(x, causal=True)
    print(f"input {tuple(x.shape)} -> output {tuple(out.shape)}, weights {tuple(weights.shape)} (batch, heads, tokens, tokens)")

    theirs = nn.MultiheadAttention(d_model, n_heads, batch_first=True).eval()
    with torch.no_grad():                                        # copy our four matrices into PyTorch's layout
        theirs.in_proj_weight.copy_(torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight]))
        theirs.in_proj_bias.copy_(torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias]))
        theirs.out_proj.weight.copy_(ours.w_o.weight)
        theirs.out_proj.bias.copy_(ours.w_o.bias)
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    ref_out, ref_weights = theirs(x, x, x, attn_mask=future, need_weights=True, average_attn_weights=False)
    d_out = (out - ref_out).abs().max().item()
    d_w = (weights - ref_weights).abs().max().item()
    print(f"biggest difference from torch.nn.MultiheadAttention: output {d_out:.1e}, weights {d_w:.1e}")
    print(f"same answer within 1e-5: {torch.allclose(out, ref_out, atol=1e-5) and torch.allclose(weights, ref_weights, atol=1e-5)}")

    print("\nparameters for d_model = 32 (four matrices of 32 x 32 plus four bias lists of 32):")
    counts = {}
    for heads in (1, 2, 4, 8):
        counts[heads] = sum(p.numel() for p in MultiHeadAttention(d_model, heads).parameters())
        print(f"  {heads} head(s) of {d_model // heads} numbers each: {counts[heads]:,} parameters")
    print(f"  4 x (32 x 32 + 32) = {4 * (32 * 32 + 32):,}")
    print("\nthe heads look at different things even at the start (random weights): row 10 of each head, first 5 columns:")
    for h in range(n_heads):
        print(f"  head {h + 1}: {[round(a, 3) for a in weights[0, h, 9, :5].tolist()]}")
    save_results("d03_multihead_module", {"output_diff": d_out, "weights_diff": d_w, "parameters": counts}, digits=10)
