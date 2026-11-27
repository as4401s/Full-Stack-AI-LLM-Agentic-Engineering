"""Demo 6 - what do the 144 heads of GPT-2 usually watch? A team of reviewers, each with a habit.

What it does: runs ten plain sentences through GPT-2 and, for every head, averages how much of its weight goes to (a) the token just before,
(b) the very first token and (c) the token itself. It lists the heads with the strongest habits and saves the map of the strongest head of each habit on our sentence.
What it needs: transformers and torch (about 550 MB of model files on the first run).
What to expect: some heads put most of their weight on the previous token, many put most on the first token, and a few stay on themselves.
Run it: uv run python -m demos.d06_head_roles
"""
import torch

from attention_lab.gpt2_maps import SENTENCE, STATS_SENTENCES, attention_for
from attention_lab.report import save_results


def main():
    prev_sum = first_sum = self_sum = None
    used = 0
    for text in STATS_SENTENCES:
        tokens, w = attention_for(text)
        n = w.shape[-1]
        idx = torch.arange(1, n)                                   # skip the first token: it can only look at itself
        prev = w[:, :, idx, idx - 1].mean(-1)                      # (layers, heads): weight on the token just before
        first = w[:, :, idx, 0].mean(-1)                           # weight on the first token
        itself = w[:, :, idx, idx].mean(-1)                        # weight on itself
        if prev_sum is None:
            prev_sum, first_sum, self_sum = prev, first, itself
        else:
            prev_sum, first_sum, self_sum = prev_sum + prev, first_sum + first, self_sum + itself
        used += 1
    prev, first, itself = prev_sum / used, first_sum / used, self_sum / used
    n_layers, n_heads = prev.shape
    print(f"{used} sentences, {n_layers} layers x {n_heads} heads = {n_layers * n_heads} heads; each number is an average over all tokens after the first")
    results = {}
    best = {}                                                      # the strongest head of each habit
    for name, table in (("the previous token", prev), ("the first token", first), ("their own token", itself)):
        ranked = sorted(((table[l, h].item(), l + 1, h + 1) for l in range(n_layers) for h in range(n_heads)), reverse=True)
        print(f"\nheads that listen most to {name}:")
        for a, l, h in ranked[:3]:
            print(f"  layer {l}, head {h}: {a:.2f}")
        print(f"  heads with average weight above 0.5 on {name}: {sum(1 for a, _, _ in ranked if a > 0.5)} of {n_layers * n_heads}")
        best[name.replace("the ", "").replace("their ", "")] = ranked[0]
        results[name.replace("the ", "").replace("their ", "")] = {"top3": [[l, h, a] for a, l, h in ranked[:3]], "above_half": sum(1 for a, _, _ in ranked if a > 0.5)}
    print(f"\nall heads together put on average {prev.mean().item():.2f} of their weight on the previous token, {first.mean().item():.2f} on the first token, {itself.mean().item():.2f} on themselves")
    # the strongest head of each habit, drawn on our own sentence (these 8 x 8 maps are used in the lesson)
    tokens, w = attention_for(SENTENCE)
    results["maps_on_sentence"] = {name: {"layer": l, "head": h, "map": [[round(float(v), 3) for v in row] for row in w[l - 1, h - 1].tolist()]}
                                   for name, (_, l, h) in best.items()}
    results["tokens"] = tokens
    print(f"\nthe strongest head of each habit, on '{SENTENCE}': " + ", ".join(f"{name} = layer {l}, head {h}" for name, (_, l, h) in best.items()))
    save_results("d06_head_roles", results)
