"""Tests for the counting model: counts, the chain rule, add-one smoothing, perplexity and the dice."""
import math
import random

from lm_basics.corpus import load
from lm_basics.ngram import NGramModel


def toy_model(smoothing=0.0):
    return NGramModel(order=2, smoothing=smoothing).fit(load("toy_corpus"))


def test_probabilities_after_the():
    m = toy_model()
    assert math.isclose(m.prob("nurse", ("the",)), 2 / 6)
    assert math.isclose(m.prob("patient", ("the",)), 1 / 6)
    assert math.isclose(sum(m.next_word_probs(("the",)).values()), 1.0)


def test_chain_rule_sentence_probability_is_one_eighteenth():
    m = toy_model()
    assert math.isclose(m.sentence_prob("the nurse checks the chart .".split()), 1 / 18)


def test_unseen_pair_has_probability_zero_and_infinite_perplexity():
    m = toy_model()
    assert m.prob("greets", ("doctor",)) == 0.0
    assert m.perplexity([["the", "doctor", "greets", "the", "patient", "."]]) == math.inf


def test_add_one_gives_a_small_positive_probability_and_still_sums_to_one():
    m = toy_model(smoothing=1.0)
    assert math.isclose(m.prob("greets", ("doctor",)), 1 / 10)
    assert math.isclose(m.prob("reads", ("doctor",)), 2 / 10)
    assert math.isclose(sum(m.prob(w, ("doctor",)) for w in m.vocab), 1.0)


def test_perplexity_of_the_toy_sentence():
    m = toy_model()
    assert math.isclose(m.perplexity([["the", "nurse", "checks", "the", "chart", "."]]), 18 ** (1 / 6))


def test_generation_is_repeatable_and_ends_with_a_full_stop():
    m = NGramModel(order=3).fit(load("caredesk_train"))
    a = m.generate(random.Random(5))
    b = m.generate(random.Random(5))
    assert a == b and a[-1] == "."


def test_dice_follow_the_probabilities():
    m = toy_model()
    rng = random.Random(0)
    counts = {}
    for _ in range(6000):
        w = m.generate(rng, max_words=1, start=("the",))[0]
        counts[w] = counts.get(w, 0) + 1
    assert abs(counts["nurse"] / 6000 - 2 / 6) < 0.03
    assert abs(counts["doctor"] / 6000 - 1 / 6) < 0.03


def test_trigram_beats_bigram_on_usual_sentences():
    train, test = load("caredesk_train"), load("caredesk_test")[:20]
    bi = NGramModel(2, 1.0).fit(train).perplexity(test)
    tri = NGramModel(3, 0.1).fit(train).perplexity(test)
    assert tri < bi
