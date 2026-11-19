"""ngram.py - a language model made only of counts.

What it does
    Counts how often each word follows each short context (one word for a bigram model, two for a trigram
    model), turns the counts into probabilities, scores whole sentences with the chain rule, measures
    perplexity, and writes new text by rolling weighted dice.

What it needs
    Only the Python standard library.

What to expect
    On the three toy sentences, P(nurse | the) = 2/6 = 0.333 and the sentence 'the nurse checks the chart .'
    gets probability 1/18 = 0.0556.
"""
import math
from collections import Counter

from .corpus import BOUNDARY, padded


class NGramModel:
    """Counts of which word follows which context.

    order=2 is a bigram model (the context is the one previous word), order=3 is a trigram model.
    smoothing=k adds k to every count, so a pair that never appeared still gets a small probability.
    """

    def __init__(self, order=2, smoothing=0.0):
        self.order = order
        self.context_size = order - 1
        self.smoothing = smoothing
        self.counts = {}          # context (tuple of words) -> Counter of next words, kept in order of first appearance
        self.vocab = []           # every word seen in training

    def fit(self, sentences):
        """Count every (context, next word) pair in the training sentences."""
        seen = []
        for s in sentences:
            p = padded(s, self.context_size)
            for i in range(self.context_size, len(p)):
                context = tuple(p[i - self.context_size:i])
                self.counts.setdefault(context, Counter())[p[i]] += 1
                if p[i] not in seen:
                    seen.append(p[i])
        self.vocab = sorted(set(seen) | {BOUNDARY})
        return self

    def prob(self, word, context):
        """Probability of `word` after `context`: its count divided by the total count for that context."""
        context = tuple(context)
        following = self.counts.get(context, Counter())
        total = sum(following.values())
        k = self.smoothing
        if total + k * len(self.vocab) == 0:
            return 0.0                       # a context never seen in training, and no smoothing to fall back on
        return (following.get(word, 0) + k) / (total + k * len(self.vocab))

    def next_word_probs(self, context):
        """Return {word: probability} for every word that followed this context in training (in order of appearance)."""
        context = tuple(context)
        following = self.counts.get(context, Counter())
        return {w: self.prob(w, context) for w in following}

    def step_probs(self, sentence):
        """Return the probability the model gave to each word of the sentence, one number per word."""
        p = padded(sentence, self.context_size)
        return [self.prob(p[i], p[i - self.context_size:i]) for i in range(self.context_size, len(p))]

    def sentence_prob(self, sentence):
        """The chain rule: multiply the probability of every word given the words before it."""
        result = 1.0
        for q in self.step_probs(sentence):
            result *= q
        return result

    def perplexity(self, sentences):
        """e raised to the average of -ln(probability of the right word). Infinite if any word got probability 0."""
        total_loss, n = 0.0, 0
        for s in sentences:
            for q in self.step_probs(s):
                if q == 0.0:
                    return math.inf
                total_loss += -math.log(q)
                n += 1
        return math.exp(total_loss / n)

    def generate(self, rng, max_words=12, start=None):
        """Write a sentence by rolling a weighted die for each next word; stop at the full stop.

        `rng` is a random.Random, so the same seed always writes the same sentence.
        """
        context = tuple(start) if start else (BOUNDARY,) * self.context_size
        out = []
        for _ in range(max_words):
            choices = self.next_word_probs(context)
            if not choices:
                break                          # this context never appeared in training
            roll = rng.random()                # a number between 0 and 1
            running = 0.0
            chosen = list(choices)[-1]
            for word, p in choices.items():    # walk along the number line, one slice per word
                running += p
                if roll < running:
                    chosen = word
                    break
            out.append(chosen)
            context = context[1:] + (chosen,)
            if chosen == BOUNDARY:
                break
        return out
