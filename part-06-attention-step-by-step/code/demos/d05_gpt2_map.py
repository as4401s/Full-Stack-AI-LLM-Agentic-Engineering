"""Demo 5 - read a real attention map: how GPT-2 handles "it" in "the animal stayed home because it was tired".

What it does: runs the sentence through GPT-2 (12 layers x 12 heads = 144 heads), checks that the real weights show the causal shape (zeros above
the diagonal, rows adding up to 1), looks at the row of the word "it" in every head, and draws the map of the head that listens most to "animal".
What it needs: transformers, torch, matplotlib (optional). The first run downloads about 550 MB of model files.
What to expect: 8 tokens; a handful of heads put a large share of the "it" row on "animal"; many heads put most of it on the first token.
Run it: uv run python -m demos.d05_gpt2_map
"""
import torch

from attention_lab.gpt2_maps import SENTENCE, attention_for
from attention_lab.maps import plot_map
from attention_lab.report import RESULTS_DIR, save_results


def main():
    tokens, w = attention_for(SENTENCE)
    n_layers, n_heads, n, _ = w.shape
    it, animal = tokens.index("it"), tokens.index("animal")
    print(f"sentence: {SENTENCE!r}")
    print(f"tokens ({n}): {tokens}")
    print(f"weights tensor: {n_layers} layers x {n_heads} heads x {n} x {n} = {w.numel():,} numbers")
    above = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    print(f"largest weight above the diagonal in any head: {w[..., above].max().item():.1f}")
    print(f"largest distance from 1 of any row sum: {(w.sum(-1) - 1).abs().max().item():.1e}")

    to_animal = w[:, :, it, animal]                                # weight of the row "it" on the column "animal", every head
    flat = [(to_animal[l, h].item(), l + 1, h + 1) for l in range(n_layers) for h in range(n_heads)]
    flat.sort(reverse=True)
    print(f"\nthe row of 'it' (position {it + 1}) listening to 'animal' (position {animal + 1}), over all {n_layers * n_heads} heads:")
    print(f"  average weight {to_animal.mean().item():.3f}; heads with weight above 0.20: {sum(1 for a, _, _ in flat if a > 0.20)}; above 0.10: {sum(1 for a, _, _ in flat if a > 0.10)}")
    print("  the five heads that listen most to 'animal' (layer, head: weight):")
    for a, l, h in flat[:5]:
        print(f"    layer {l}, head {h}: {a:.3f}")
    first = w[:, :, it, 0]
    print(f"  for comparison, the row of 'it' on the first token '{tokens[0]}': average {first.mean().item():.3f}; heads with weight above 0.5: {int((first > 0.5).sum())} of {n_layers * n_heads}")

    best_w, best_l, best_h = flat[0]
    row = w[best_l - 1, best_h - 1, it]
    print(f"\nrow of 'it' in the best head (layer {best_l}, head {best_h}):")
    for t, a in zip(tokens[:it + 1], row[:it + 1].tolist()):
        print(f"  {t:>8s} {a:.3f}")
    others = [(row[j].item(), tokens[j]) for j in range(1, it)]
    print(f"  biggest weight on a word other than the first token: {max(others)[1]} ({max(others)[0]:.3f})")
    mean_map = w.mean(dim=(0, 1))
    print(f"\nrow of 'it' averaged over all {n_layers * n_heads} heads:")
    for t, a in zip(tokens[:it + 1], mean_map[it, :it + 1].tolist()):
        print(f"  {t:>8s} {a:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ok1 = plot_map(w[best_l - 1, best_h - 1], tokens, RESULTS_DIR / "map_gpt2_best_head.png", f"GPT-2 layer {best_l}, head {best_h}")
    ok2 = plot_map(mean_map, tokens, RESULTS_DIR / "map_gpt2_average.png", "GPT-2 average of all 144 heads")
    print(f"\npictures saved: {ok1 and ok2}")
    save_results("d05_gpt2_map", {"tokens": tokens, "best_head": [best_l, best_h], "best_head_map": w[best_l - 1, best_h - 1], "average_map": mean_map,
                                 "top5": [[l, h, a] for a, l, h in flat[:5]], "mean_it_to_animal": to_animal.mean().item()})
