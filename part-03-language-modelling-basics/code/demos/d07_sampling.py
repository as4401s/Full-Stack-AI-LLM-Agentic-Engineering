"""Demo 7 - sampling from the same three scores with different settings.

What it does: takes the scores [2.0, 1.0, 0.1] for the words checks, greets and reads (the same scores as lesson 2),
prints the probabilities at temperatures 0.5, 1 and 2, with top-k and with top-p, and then draws 10,000 words
at each setting to show that the draws follow the probabilities.
What it needs: NumPy.
What to expect: temperature 0.5 gives 0.864, 0.117, 0.019; temperature 2 gives 0.502, 0.304, 0.194; top-k 2 gives 0.731, 0.269.
Run it: uv run python -m demos.d07_sampling
"""
import numpy as np

from lm_basics.report import run_and_save, save_results
from lm_basics.sampler import sample_next, softmax, top_k_filter, top_p_filter

WORDS = ["checks", "greets", "reads"]
LOGITS = [2.0, 1.0, 0.1]


def show(label, probs):
    """Print one row of probabilities."""
    print(f"  {label:30s}" + "".join(f"{p:8.3f}" for p in probs))


def main():
    print(f"  {'':30s}" + "".join(f"{w:>8s}" for w in WORDS))
    show("scores", LOGITS)
    show("temperature 1 (softmax)", softmax(LOGITS, 1.0))
    show("temperature 0.5", softmax(LOGITS, 0.5))
    show("temperature 2", softmax(LOGITS, 2.0))
    base = softmax(LOGITS, 1.0)
    show("top-k, k=2 (temperature 1)", top_k_filter(base, 2))
    show("top-p, p=0.9 (temperature 1)", top_p_filter(base, 0.9))
    show("top-p, p=0.6 (temperature 1)", top_p_filter(base, 0.6))
    print("\n  top-p adapts to how sure the model is (p = 0.9):")
    sure, unsure = [0.95, 0.03, 0.02], [0.35, 0.30, 0.20, 0.10, 0.05]
    print("   sure model   ", sure, "->", [round(float(x), 3) for x in top_p_filter(sure, 0.9)])
    print("   unsure model ", unsure, "->", [round(float(x), 3) for x in top_p_filter(unsure, 0.9)])
    print("\n  greedy always picks:", WORDS[sample_next(LOGITS, temperature=0)])
    print("\n  10,000 draws per setting (seed 0): share of each word")
    results = {}
    settings = {"temperature 0.5": dict(temperature=0.5), "temperature 1": dict(temperature=1.0),
                "temperature 2": dict(temperature=2.0), "top-k 2": dict(top_k=2), "top-p 0.9": dict(top_p=0.9)}
    for label, kwargs in settings.items():
        rng = np.random.default_rng(0)
        draws = [sample_next(LOGITS, rng=rng, **kwargs) for _ in range(10000)]
        share = np.bincount(draws, minlength=3) / len(draws)
        show(label, share)
        results[label] = share
    save_results("d07_sampling", results)


if __name__ == "__main__":
    run_and_save("d07_sampling", main)
