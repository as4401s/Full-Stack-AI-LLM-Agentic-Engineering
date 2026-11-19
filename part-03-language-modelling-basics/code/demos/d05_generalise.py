"""Demo 5 - why the neural model copes with a pairing it never saw: similar words get similar embeddings.

What it does: prints the two-number embedding of every word, measures how close together the words of each kind sit,
and compares what the counting model and the neural model say about the next word after 'the doctor' (the training
text never shows 'doctor checks' or 'doctor updates').
What it needs: PyTorch.
What to expect: the seven 'thing' words (chart, patient, form ...) sit almost on top of each other, far closer than words
in general; the neural model gives 'checks' a clearly positive probability after 'the doctor', where plain counts give 0.
Run it: uv run python -m demos.d05_generalise
"""
import itertools
import math

from demos.common import trained_model
from lm_basics.mlp_lm import next_word_probs
from lm_basics.ngram import NGramModel
from lm_basics.report import run_and_save, save_results


def main():
    model, words, index, train_s, _ = trained_model()
    table = model.embed.weight.detach()
    print("the embedding table: two numbers for every word")
    for w in words:
        a, b = table[index[w]].tolist()
        print(f"  {w:13s} ({a:6.2f}, {b:6.2f})")
    groups = {
        "all 24 words": words,
        "the 4 job titles": ["nurse", "doctor", "clerk", "pharmacist"],
        "the 7 thing words": ["chart", "patient", "form", "schedule", "appointment", "prescription", "room"],
    }
    print("\naverage distance between the number lists of two words in a group (small = used in the same way):")
    spread = {}
    for name, members in groups.items():
        pairs = list(itertools.combinations(members, 2))
        spread[name] = sum(math.dist(table[index[a]].tolist(), table[index[b]].tolist()) for a, b in pairs) / len(pairs)
        print(f"  {name:20s} {spread[name]:.2f}")
    neural = next_word_probs(model, ["the", "doctor"], words, index)
    trigram = NGramModel(3).fit(train_s)
    smooth = NGramModel(3, 1.0).fit(train_s)
    print("\nthe next word after 'the doctor':")
    print(f"{'word':10s} {'counts':>8s} {'counts+1':>9s} {'neural':>8s}   (the training text has 'doctor reads', 'greets', 'signs', 'reviews')")
    for w in ("reads", "greets", "signs", "reviews", "checks", "updates", "prints"):
        print(f"{w:10s} {trigram.prob(w, ('the', 'doctor')):8.3f} {smooth.prob(w, ('the', 'doctor')):9.3f} {neural[w]:8.3f}")
    save_results("d05_generalise", {"embedding": {w: table[index[w]].tolist() for w in words},
                                    "average_distance_within_group": spread,
                                    "after_the_doctor": {w: neural[w] for w in words}})


if __name__ == "__main__":
    run_and_save("d05_generalise", main)
