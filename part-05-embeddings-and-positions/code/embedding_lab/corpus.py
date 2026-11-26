"""corpus.py - read the invented clinical sentences and turn them into training pairs.

What it does
    Small helpers shared by the demos: read the sentence files from ../data, build the vocabulary (every distinct word gets
    an id), and make "neighbour pairs": for every word in a sentence, every other word of the same sentence.
What it needs
    NumPy (see pyproject.toml).
What to expect
    sentences() returns a list of word lists; neighbour_pairs() returns two integer arrays of equal length (centre word id,
    neighbour word id) that the training loop in skipgram.py reads.
"""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data"


def sentences(name="clinical_words_train"):
    """Return data/<name>.txt as a list of sentences, each one a list of words."""
    text = (DATA / f"{name}.txt").read_text(encoding="utf-8")
    return [line.split() for line in text.splitlines() if line.strip()]


def vocabulary(all_sentences):
    """Return (words, index): the sorted list of distinct words and a dict from word to its id (its row in the table)."""
    words = sorted({w for s in all_sentences for w in s})
    return words, {w: i for i, w in enumerate(words)}


def word_groups():
    """Return {word: {'group': body system, 'role': organ|staff|conditions|drugs|tests|clinic}} for the content words."""
    return json.loads((DATA / "word_groups.json").read_text(encoding="utf-8"))


def neighbour_pairs(all_sentences, index):
    """Return (centre_ids, neighbour_ids): every ordered pair of different positions inside each sentence.

    Why pairs: the model is asked "this word is in a sentence; which other word is in the same sentence?". Words that keep
    the same company then end up with similar rows in the table.
    """
    centre, neighbour = [], []
    for sentence in all_sentences:
        ids = [index[w] for w in sentence]
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                if i != j:
                    centre.append(a)
                    neighbour.append(b)
    return np.array(centre, dtype=np.int64), np.array(neighbour, dtype=np.int64)
