"""Demo 11 - rotary position embeddings (RoPE): turn the arrows, and see why relative position falls out.

What it does: rotates one query and one key vector to many positions, checks that rotation keeps their length, shows that the attention score
depends only on the distance between the two positions, and tries two ways of stretching a context window. It also reads Qwen2.5's RoPE settings.
What it needs: NumPy; huggingface_hub (a few kilobytes of config) for the Qwen settings.
What to expect: lengths unchanged to 15 digits; identical scores for equal distances; shrunken positions inside the trained range.
Run it: uv run python -m demos.d11_rope
"""
import numpy as np

from embedding_lab import realmodels, rope
from embedding_lab.report import run_and_save, save_results


def main():
    rng = np.random.default_rng(7)
    q, k = rng.normal(size=8), rng.normal(size=8)
    print("one rotating pair: the first two numbers of the query, turned by position * 1 radian (57.3 degrees per position)")
    for p in range(6):
        r = rope.rotate(q, p)
        print(f"  position {p}: ({r[0]:6.3f}, {r[1]:6.3f})   angle {np.degrees(p * 1.0):6.1f} degrees")
    worst = max(abs(np.linalg.norm(rope.rotate(q, p)) - np.linalg.norm(q)) for p in range(0, 2000, 7))
    print(f"\nlength of the query: {np.linalg.norm(q):.6f}; largest change in length over 286 positions: {worst:.1e}")
    print("\nattention score numerator (dot of rotated query and rotated key) for several positions:")
    results = {}
    for m, n in ((5, 2), (105, 102), (1005, 1002), (7, 3), (107, 103)):
        s = rope.score(q, k, m, n)
        print(f"  query at {m:4d}, key at {n:4d} (distance {m - n}): {s:9.5f}")
        results[f"{m}_{n}"] = s
    gaps = max(abs(rope.score(q, k, m + d, n + d) - rope.score(q, k, m, n))
               for m, n, d in ((5, 2, 100), (30, 3, 500), (9, 9, 77), (4, 40, 1000)))
    print(f"largest change when both positions move together by the same amount: {gaps:.1e}")

    print("\nstretching a window trained for 4096 tokens to 8192 tokens:")
    new_pos = rope.scale_positions(np.array([100, 4096, 6000, 8191]), 4096, 8192)
    print(f"  position interpolation: positions 100, 4096, 6000, 8191 become {new_pos.tolist()}  (all inside 0..4096)")
    base = rope.ntk_scaled_base(10000.0, 64, 2.0)
    slow_old, slow_new = rope.rope_frequencies(64)[-1], rope.rope_frequencies(64, base)[-1]
    fast_old, fast_new = rope.rope_frequencies(64)[0], rope.rope_frequencies(64, base)[0]
    print(f"  NTK-style base: 10000 -> {base:,.0f}; slowest pair turns {slow_old:.2e} -> {slow_new:.2e} radians per position, fastest {fast_old:.2f} -> {fast_new:.2f}")
    results["interpolated"] = new_pos.tolist()
    results["ntk_base"] = base
    qwen = realmodels.try_load("qwen config", realmodels.hub_config, "Qwen/Qwen2.5-0.5B-Instruct")
    if qwen:
        head = qwen["hidden_size"] // qwen["num_attention_heads"]
        theta = qwen.get("rope_theta") or qwen.get("rope_parameters", {}).get("rope_theta")      # older and newer config layouts
        print(f"\nQwen2.5-0.5B-Instruct: rope_theta {theta:,.0f}, max_position_embeddings {qwen['max_position_embeddings']:,}, "
              f"{qwen['num_attention_heads']} heads of {head} numbers each, so {head // 2} turning pairs per head")
        results["qwen"] = {"rope_theta": theta, "max_positions": qwen["max_position_embeddings"], "head_dim": head}
    save_results("d11_rope", results)


if __name__ == "__main__":
    run_and_save("d11_rope", main)
