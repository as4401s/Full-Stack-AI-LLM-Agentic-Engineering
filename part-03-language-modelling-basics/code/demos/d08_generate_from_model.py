"""Demo 8 - write sentences with the trained neural model under different sampling settings.

What it does: starts every sentence after a full stop, asks the trained model for scores for the next word, picks a word with
greedy, low temperature, temperature 1, high temperature, or top-p, and stops at the next full stop. It writes six
sentences per setting and counts how many different sentences appear.
What it needs: PyTorch.
What to expect: greedy writes the same sentence six times; low temperature repeats a few favourites; a high temperature
writes more varied (and sometimes odder) sentences.
Run it: uv run python -m demos.d08_generate_from_model
"""
import numpy as np

from demos.common import trained_model
from lm_basics.mlp_lm import logits_for
from lm_basics.report import run_and_save, save_results
from lm_basics.sampler import sample_next


def write_sentence(model, words, index, rng, **settings):
    """Write one sentence word by word; each new word comes from sampling the model's scores."""
    context = ["."] * model.context_size
    out = []
    for _ in range(12):
        logits = logits_for(model, context, index)
        word = words[sample_next(logits, rng=rng, **settings)]
        out.append(word)
        context = context[1:] + [word]
        if word == ".":
            break
    return " ".join(out)


def main():
    model, words, index, _, _ = trained_model()
    settings = {"greedy (temperature 0)": dict(temperature=0), "temperature 0.5": dict(temperature=0.5),
                "temperature 1": dict(temperature=1.0), "temperature 2": dict(temperature=2.0),
                "temperature 1, top-p 0.9": dict(temperature=1.0, top_p=0.9)}
    written = {}
    for label, kwargs in settings.items():
        rng = np.random.default_rng(1)
        sentences = [write_sentence(model, words, index, rng, **kwargs) for _ in range(6)]
        written[label] = sentences
        print(f"{label}: {len(set(sentences))} different sentences out of 6")
        for s in sentences:
            print("   ", s)
        print()
    save_results("d08_generate_from_model", written)


if __name__ == "__main__":
    run_and_save("d08_generate_from_model", main)
