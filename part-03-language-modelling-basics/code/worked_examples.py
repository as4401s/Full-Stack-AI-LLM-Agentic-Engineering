"""worked_examples.py - reproduces every small number calculated by hand in lesson 3.

What it does
    Recomputes, with plain Python and NumPy only, each calculation we do on a calculator in the video:
    counts and probabilities from three toy sentences, the chain rule, the weighted dice, the add-one fix,
    two-number embeddings and the scores they give, cross-entropy and perplexity, temperature, top-k, top-p,
    and a chat template.

What it needs
    NumPy. PyTorch is not needed.

What to expect
    Run:  python worked_examples.py
    Every line printed here is a number you can also get on a calculator. The saved output is in
    data/expected_output/worked_examples.txt.
"""
import math

import numpy as np

from lm_basics.chat_template import format_chat
from lm_basics.ngram import NGramModel
from lm_basics.sampler import softmax, top_k_filter, top_p_filter

TOY = ["the nurse checks the chart .", "the nurse greets the patient .", "the doctor reads the chart ."]


def r(x, n=3):
    """Round for printing."""
    return round(float(x), n)


def counts_and_chain_rule():
    """Counting words after 'the' and multiplying probabilities along a sentence."""
    print("== 1. Counting: what follows 'the'?")
    sentences = [s.split() for s in TOY]
    model = NGramModel(order=2).fit(sentences)
    after_the = model.counts[("the",)]
    total = sum(after_the.values())
    print("the appears", total, "times; the words after it:", dict(after_the))
    for word, c in after_the.items():
        print(f"  P({word} | the) = {c}/{total} = {r(c / total)}")
    print("== 2. The chain rule: multiply one probability per word")
    sentence = "the nurse checks the chart .".split()
    steps = model.step_probs(sentence)
    prev = ["."] + sentence[:-1]
    for before, word, p in zip(prev, sentence, steps):
        print(f"  P({word} | {before}) = {r(p)}")
    product = math.prod(steps)
    print(f"  product = 1 x 1/3 x 1/2 x 1 x 1/3 x 1 = 1/18 = {r(product, 4)}")
    return model


def dice(model):
    """Turn the probabilities after 'the' into slices of a number line and roll once."""
    print("== 3. Weighted dice: slices of the number line from 0 to 1")
    running = 0.0
    slices = []
    for word, p in model.next_word_probs(("the",)).items():
        slices.append((word, running, running + p))
        print(f"  {word}: from {r(running)} up to {r(running + p)}")
        running += p
    roll = 0.52
    chosen = next(w for w, lo, hi in slices if lo <= roll < hi)
    print(f"  a roll of {roll} lands in the slice of '{chosen}'")


def unseen_and_add_one(model):
    """A pair never seen in training gets probability 0; adding 1 to every count repairs that."""
    print("== 4. A pair the text never showed")
    print("  P(greets | doctor) =", r(model.prob("greets", ("doctor",))), "so the whole sentence 'the doctor greets the patient .' gets",
          r(model.sentence_prob("the doctor greets the patient .".split())))
    vocab_size = len(model.vocab)
    print("  words in the toy vocabulary:", vocab_size)
    smooth = NGramModel(order=2, smoothing=1.0).fit([s.split() for s in TOY])
    print(f"  add one to every count: P(greets | doctor) = (0 + 1) / (1 + {vocab_size}) = {r(smooth.prob('greets', ('doctor',)))}")
    print(f"  add one to every count: P(reads | doctor)  = (1 + 1) / (1 + {vocab_size}) = {r(smooth.prob('reads', ('doctor',)))}")
    total = sum(smooth.prob(w, ("doctor",)) for w in smooth.vocab)
    print("  the probabilities after 'doctor' still add up to", r(total))


def embeddings():
    """Two numbers per word, one multiply-and-add per candidate word, then softmax: lesson 2 again."""
    print("== 5. Embeddings: two numbers for every word")
    emb = {"nurse": np.array([1.0, 0.5]), "doctor": np.array([0.9, 0.6]), "chart": np.array([-0.7, 0.9])}
    weights = {"checks": np.array([1.5, 1.0]), "greets": np.array([1.0, 0.0]), "reads": np.array([-0.2, 0.6])}
    for name in ("nurse", "doctor"):
        scores = np.array([emb[name] @ w for w in weights.values()])
        probs = softmax(scores)
        shown = ", ".join(f"{k} {r(s, 2)}" for k, s in zip(weights, scores))
        print(f"  after '{name}': scores {shown}  ->  probabilities", [r(p) for p in probs])
    print("  distance nurse to doctor:", r(np.linalg.norm(emb['nurse'] - emb['doctor'])),
          "  nurse to chart:", r(np.linalg.norm(emb['nurse'] - emb['chart'])))


def cross_entropy_and_perplexity():
    """Surprise per word is -ln(probability); perplexity is e raised to the average surprise."""
    print("== 6. Cross-entropy and perplexity")
    for p in (1.0, 0.5, 0.25, 0.125, 0.1):
        print(f"  probability {p}: loss = -ln({p}) = {r(-math.log(p) + 0.0)}")
    probs = [0.5, 0.25, 0.5, 0.125]
    losses = [-math.log(p) for p in probs]
    mean = sum(losses) / len(losses)
    print("  four steps with probabilities", probs, "-> losses", [r(x) for x in losses], "-> average", r(mean, 4))
    print(f"  perplexity = e^{r(mean, 4)} = {r(math.exp(mean))}   (check: 128 to the power 1/4 = {r(128 ** 0.25)})")
    print(f"  a loss of 2.0 means perplexity e^2.0 = {r(math.exp(2.0))} equally likely choices")
    print(f"  a blind guess among 10 words: loss ln(10) = {r(math.log(10))}, perplexity {r(math.exp(math.log(10)))}")
    toy_probs = [1, 1 / 3, 1 / 2, 1, 1 / 3, 1]
    toy_mean = sum(-math.log(p) for p in toy_probs) / len(toy_probs)
    print(f"  the toy sentence: six steps, average loss {r(toy_mean, 4)}, perplexity {r(math.exp(toy_mean))}")


def sampling():
    """The same three scores, with temperature, top-k and top-p."""
    print("== 7. Sampling from the scores [2.0, 1.0, 0.1] for checks, greets, reads")
    logits = [2.0, 1.0, 0.1]
    print("  greedy picks the highest score: index", int(np.argmax(logits)), "(checks)")
    for t in (1.0, 0.5, 2.0):
        print(f"  temperature {t}: scores / {t} =", [r(x / t, 2) for x in logits], "-> probabilities", [r(p) for p in softmax(logits, t)])
    base = softmax(logits)
    print("  top-k with k=2 at temperature 1:", [r(p) for p in top_k_filter(base, 2)])
    print("  top-p with p=0.9 at temperature 1:", [r(p) for p in top_p_filter(base, 0.9)],
          "(the first two words already cover", r(base[0] + base[1]), ")")
    sure, unsure = [0.95, 0.03, 0.02], [0.35, 0.30, 0.20, 0.10, 0.05]
    print("  top-p 0.9 when the model is sure  ", sure, "->", [r(x) for x in top_p_filter(sure, 0.9)])
    print("  top-p 0.9 when the model is unsure", unsure, "->", [r(x) for x in top_p_filter(unsure, 0.9)])


def chat():
    """A conversation written as one piece of text."""
    print("== 8. A chat template")
    text = format_chat([
        {"role": "system", "content": "You are CareDesk, a helpful front-desk assistant."},
        {"role": "user", "content": "When does the clinic open?"},
    ])
    print(text, end="")


def main():
    model = counts_and_chain_rule()
    dice(model)
    unseen_and_add_one(model)
    embeddings()
    cross_entropy_and_perplexity()
    sampling()
    chat()


if __name__ == "__main__":
    main()
