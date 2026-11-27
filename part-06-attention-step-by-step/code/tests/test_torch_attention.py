"""Tests for attention.py and multihead.py (they need PyTorch and are skipped without it)."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from attention_lab.attention import attention  # noqa: E402
from attention_lab.multihead import MultiHeadAttention  # noqa: E402


def test_matches_the_built_in_kernel_with_and_without_the_mask():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 9, 8) for _ in range(3))
    for causal in (False, True):
        out, w = attention(q, k, v, causal=causal)
        assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v, is_causal=causal), atol=1e-5)
        assert torch.allclose(w.sum(-1), torch.ones(2, 3, 9), atol=1e-5)


def test_causal_weights_are_zero_above_the_diagonal():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 6, 4)
    _, w = attention(q, q, q, causal=True)
    assert torch.all(w[..., torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)] == 0)


def test_multihead_matches_torch_module_and_parameter_count_is_head_independent():
    torch.manual_seed(0)
    ours = MultiHeadAttention(16, 4).eval()
    theirs = nn.MultiheadAttention(16, 4, batch_first=True).eval()
    with torch.no_grad():
        theirs.in_proj_weight.copy_(torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight]))
        theirs.in_proj_bias.copy_(torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias]))
        theirs.out_proj.weight.copy_(ours.w_o.weight)
        theirs.out_proj.bias.copy_(ours.w_o.bias)
    x = torch.randn(2, 7, 16)
    mask = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
    out, _ = ours(x, causal=True)
    ref, _ = theirs(x, x, x, attn_mask=mask, need_weights=False)
    assert torch.allclose(out, ref, atol=1e-5)
    assert sum(p.numel() for p in MultiHeadAttention(16, 1).parameters()) == sum(p.numel() for p in MultiHeadAttention(16, 8).parameters())
