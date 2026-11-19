"""Demo 1 - count the words that follow 'the', then score a whole sentence with the chain rule.

What it does: builds the bigram counts from the three toy sentences, prints P(word | the), multiplies one
probability per word to score 'the nurse checks the chart .', and turns that into a loss and a perplexity.
What it needs: only the Python standard library.
What to expect: P(nurse | the) = 0.333, sentence probability 0.0556 (that is 1/18), perplexity 1.619.
Run it: uv run python -m demos.d01_counts_chain_rule
"""
import math

from lm_basics.corpus import load
from lm_basics.ngram import NGramModel
from lm_basics.report import run_and_save, save_results


def main():
    toy = load("toy_corpus")
    print("toy sentences:")
    for s in toy:
        print("  ", " ".join(s))
    model = NGramModel(order=2).fit(toy)
    after_the = model.counts[("the",)]
    total = sum(after_the.values())
    print(f"\n'the' is followed by a word {total} times:")
    for word, count in after_the.items():
        print(f"  {word:8s} {count} of {total}  ->  P({word} | the) = {count / total:.3f}")
    sentence = "the nurse checks the chart .".split()
    probs = model.step_probs(sentence)
    print("\nthe chain rule, one probability per word (the sentence starts after a full stop):")
    previous = ["."] + sentence[:-1]
    for before, word, p in zip(previous, sentence, probs):
        print(f"  P({word} | {before}) = {p:.3f}")
    print(f"product of the six numbers = {math.prod(probs):.4f}   (1/18 = {1 / 18:.4f})")
    average_loss = sum(-math.log(p) for p in probs) / len(probs)
    print(f"average loss per word = {average_loss:.4f}   perplexity = e^loss = {math.exp(average_loss):.3f}")
    save_results("d01_counts_chain_rule", {"after_the": dict(after_the), "sentence_probability": math.prod(probs),
                                           "average_loss": average_loss, "perplexity": math.exp(average_loss)})


if __name__ == "__main__":
    run_and_save("d01_counts_chain_rule", main)
