"""Demo 4 - draw the three-token attention maps (full and causal) as pictures.

What it does: computes the 3 x 3 weights of the hand example, with and without the causal mask, and saves two PNG pictures next to the results.
What it needs: torch and matplotlib (the numbers print even when matplotlib is missing).
What to expect: two files, data/results/map_toy_full.png and data/results/map_toy_causal.png; the causal one has a white upper triangle.
Run it: uv run python -m demos.d04_toy_maps
"""
import torch

from attention_lab.attention import attention
from attention_lab.maps import plot_map
from attention_lab.report import RESULTS_DIR


def main():
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).unsqueeze(0).unsqueeze(0)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tokens = ["token 1", "token 2", "token 3"]
    for causal, name in ((False, "map_toy_full"), (True, "map_toy_causal")):
        _, w = attention(keys, keys, keys, causal=causal)
        table = w[0, 0]
        print(f"{name}: rows listen, columns are listened to")
        for t, row in zip(tokens, table.tolist()):
            print(f"  {t}: {[round(x, 2) for x in row]}  sum {sum(row):.2f}")
        ok = plot_map(table, tokens, RESULTS_DIR / f"{name}.png", "causal: only the past" if causal else "full: every token listens to all")
        print(f"  picture saved: {ok}")
