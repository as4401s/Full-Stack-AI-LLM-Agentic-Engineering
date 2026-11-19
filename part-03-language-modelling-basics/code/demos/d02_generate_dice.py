"""Demo 2 - write new sentences by rolling weighted dice on a counting model.

What it does: trains a bigram model and a trigram model on 600 CareDesk training sentences, shows the
probabilities after 'the nurse', and writes ten sentences with each model using a fixed random seed.
What it needs: only the Python standard library.
What to expect: ten readable front-desk sentences from each model; the same ten every time you run it.
Run it: uv run python -m demos.d02_generate_dice
"""
import random

from lm_basics.corpus import load
from lm_basics.ngram import NGramModel
from lm_basics.report import run_and_save, save_results


def main():
    train = load("caredesk_train")
    bigram = NGramModel(order=2).fit(train)
    trigram = NGramModel(order=3).fit(train)
    print(f"training sentences: {len(train)}   words in the vocabulary: {len(bigram.vocab)}")
    print("\nbigram: the probabilities for the word after 'the' (a one-word context, so it must guess the noun):")
    for word, p in sorted(bigram.next_word_probs(("the",)).items(), key=lambda kv: -kv[1]):
        print(f"  {word:13s} {p:.3f}")
    print("\ntrigram: the probabilities after 'the nurse' (a two-word context):")
    for word, p in sorted(trigram.next_word_probs(("the", "nurse")).items(), key=lambda kv: -kv[1]):
        print(f"  {word:13s} {p:.3f}")
    written = {}
    for name, model in (("bigram", bigram), ("trigram", trigram)):
        rng = random.Random(3)
        print(f"\nten sentences written by the {name} model (seed 3):")
        written[name] = []
        for _ in range(10):
            sentence = " ".join(model.generate(rng))
            written[name].append(sentence)
            print("  ", sentence)
    save_results("d02_generate_dice", written)


if __name__ == "__main__":
    run_and_save("d02_generate_dice", main)
