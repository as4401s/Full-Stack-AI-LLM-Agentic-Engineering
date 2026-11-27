"""multihead.py - multi-head attention as a small PyTorch module.

What it does
    Projects the input x into queries, keys and values with four learned matrices, splits them into heads (each head gets its own slice of
    the numbers), runs attention() in every head at once, glues the heads' outputs together and mixes them with an output matrix.

What it needs
    PyTorch and attention.py from this folder.

What to expect
    MultiHeadAttention(d_model=32, n_heads=4)(x) with x of shape (batch, seq, 32) returns (output (batch, seq, 32), weights (batch, 4, seq, seq)).
    The number of parameters is the same for 1 head or for 4 heads: the split costs nothing extra.
"""
import torch
from torch import nn

from attention_lab.attention import attention


class MultiHeadAttention(nn.Module):
    """Several attention heads side by side, each reading its own slice of the numbers."""

    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide evenly into heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=bias)         # builds one question per token
        self.w_k = nn.Linear(d_model, d_model, bias=bias)         # builds one label per token
        self.w_v = nn.Linear(d_model, d_model, bias=bias)         # builds one content per token
        self.w_o = nn.Linear(d_model, d_model, bias=bias)         # mixes the heads' answers together

    def split_heads(self, x):
        """(batch, seq, d_model) -> (batch, heads, seq, head_dim): cut the numbers into one slice per head."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, causal: bool = True):
        """Run all heads; returns (output, weights)."""
        q, k, v = self.split_heads(self.w_q(x)), self.split_heads(self.w_k(x)), self.split_heads(self.w_v(x))
        out, weights = attention(q, k, v, causal=causal)          # every head at once
        batch, _, seq, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.n_heads * self.head_dim)   # glue the heads back together
        return self.w_o(out), weights
