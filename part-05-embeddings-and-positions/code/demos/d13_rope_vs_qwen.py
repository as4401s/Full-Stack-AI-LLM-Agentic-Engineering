"""Demo 13 - does our rope.py rotate like a real model? A check against Qwen2.5's own rotary code.

What it does: builds the rotary-embedding module that the transformers library uses for Qwen2.5-0.5B-Instruct (from its config.json only; the
model weights are never downloaded), rotates random query vectors at several positions with it, rotates the same vectors with our NumPy
rope.rotate (half-and-half pairing, the model's rope_theta), and compares the two answers number by number.
What it needs: torch, transformers and numpy; a few kilobytes of config.json.
What to expect: the turning speeds agree to about seven digits, and the rotated vectors agree to six or seven digits at small positions (the library works in
32-bit floats, we work in 64-bit floats); at position 20,000 the 32-bit rounding grows to about 3 in 10,000, still far below any difference that matters.
Run it: uv run python -m demos.d13_rope_vs_qwen
"""
import numpy as np
import torch

from embedding_lab import realmodels, rope
from embedding_lab.report import run_and_save, save_results

REPO = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    from transformers import AutoConfig
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    config = AutoConfig.from_pretrained(REPO)
    raw = realmodels.hub_config(REPO)
    theta = raw.get("rope_theta") or raw.get("rope_parameters", {}).get("rope_theta")
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"{REPO}: rope_theta {theta:,.0f}, {config.num_attention_heads} heads of {head_dim} numbers, window {config.max_position_embeddings:,} tokens")

    rotary = qwen.Qwen2RotaryEmbedding(config=config)
    library_frequencies = rotary.inv_freq.detach().numpy().astype(float)             # turning speed of each pair, as the library stores it
    our_frequencies = rope.rope_frequencies(head_dim, base=theta)
    print(f"turning speeds (radians per position) of the {head_dim // 2} pairs: library vs rope.rope_frequencies")
    print(f"  fastest pair: {library_frequencies[0]:.6f} vs {our_frequencies[0]:.6f}; slowest pair: {library_frequencies[-1]:.3e} vs {our_frequencies[-1]:.3e}")
    print(f"  largest relative difference over all pairs: {np.max(np.abs(library_frequencies / our_frequencies - 1)):.1e}\n")

    positions = np.array([0, 1, 2, 3, 10, 100, 1000, 20000])
    queries = np.random.default_rng(0).normal(size=(len(positions), head_dim))
    q = torch.tensor(queries, dtype=torch.float32)[None, None]                      # shape (batch 1, heads 1, positions, numbers)
    cos, sin = rotary(q, torch.tensor(positions)[None])                              # the library's cos and sin for these positions
    theirs, _ = qwen.apply_rotary_pos_emb(q, q, cos, sin)                            # (the key copy is not needed here)
    ours = rope.rotate(queries, positions, base=theta, pairing="half")
    difference = np.abs(theirs[0, 0].numpy().astype(float) - ours)
    print(f"\nrotated {len(positions)} random query vectors of {head_dim} numbers at positions {positions.tolist()}")
    for p, d in zip(positions, difference.max(axis=1)):
        print(f"  position {p:6d}: largest difference between the library and rope.rotate = {d:.1e}")
    print(f"\nlargest difference over all {difference.size:,} numbers: {difference.max():.1e}")
    same = bool(difference.max() < 1e-3)
    print(f"the two agree to better than 0.001 everywhere: {same}")
    interleaved = rope.rotate(queries, positions, base=theta, pairing="interleaved")
    other = np.abs(theirs[0, 0].numpy().astype(float) - interleaved).max()
    print(f"with the other pairing (neighbouring numbers) the largest difference would be {other:.2f}, because the numbers are paired in another order")
    save_results("d13_rope_vs_qwen", {"frequency_relative_difference": float(np.max(np.abs(library_frequencies / our_frequencies - 1))), "max_difference": float(difference.max()), "agree": same, "other_pairing_difference": float(other)}, digits=8)


if __name__ == "__main__":
    run_and_save("d13_rope_vs_qwen", main)
