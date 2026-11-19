"""Demo 11 - keyboard-style suggestions: the three most likely next words after a few words of text.

What it does: gives the counting model (trigram) and the trained neural model the last two words of a sentence and
prints their three most likely next words with probabilities, like the suggestion bar on a phone keyboard.
What it needs: PyTorch.
What to expect: after 'checks the' both models suggest 'chart' and 'prescription' first; the neural model also
keeps a little probability for words the training text never put there.
Run it: uv run python -m demos.d11_suggestions
"""
from demos.common import trained_model
from lm_basics.mlp_lm import next_word_probs
from lm_basics.ngram import NGramModel
from lm_basics.report import run_and_save, save_results


def top3(probs):
    """The three most probable (word, probability) pairs."""
    return sorted(probs.items(), key=lambda kv: -kv[1])[:3]


def main():
    model, words, index, train_s, _ = trained_model()
    trigram = NGramModel(order=3).fit(train_s)
    saved = {}
    for context in (("checks", "the"), ("updates", "the"), ("the", "nurse"), ("the", "doctor")):
        typed = "the nurse checks the" if context == ("checks", "the") else " ".join(context)
        counts = top3(trigram.next_word_probs(context))
        neural = top3(next_word_probs(model, list(context), words, index))
        print(f"after '{' '.join(context)}':")
        print("   counting model:", ", ".join(f"{w} {p:.3f}" for w, p in counts))
        print("   neural model:  ", ", ".join(f"{w} {p:.3f}" for w, p in neural))
        saved[" ".join(context)] = {"counting": counts, "neural": neural}
    save_results("d11_suggestions", saved)


if __name__ == "__main__":
    run_and_save("d11_suggestions", main)
