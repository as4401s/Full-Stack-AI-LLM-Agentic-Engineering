"""corpus.py - read the practice text and turn words into numbers.

What it does
    * reads a text file with one sentence per line,
    * splits each sentence into words (in this lesson one word is one token; lesson 4 builds real tokens),
    * gives every distinct word a number (the vocabulary), and
    * cuts the text into (context, next word) examples that the models learn from.

What it needs
    Only the Python standard library.
"""
from pathlib import Path

BOUNDARY = "."  # the full stop marks the end of one sentence and so also the start of the next


def read_sentences(path):
    """Read a text file and return a list of sentences, each one a list of words (lowercase, ending in '.')."""
    sentences = []
    for line in Path(path).read_text().splitlines():
        words = line.strip().lower().split()
        if words:
            sentences.append(words)
    return sentences


def build_vocab(sentences):
    """Return (words, index): the sorted list of distinct words, and a dict mapping each word to its number."""
    words = sorted({w for s in sentences for w in s})
    index = {w: i for i, w in enumerate(words)}
    return words, index


def padded(sentence, context_size):
    """Put `context_size` boundary words in front of a sentence, so even the first word has a full context."""
    return [BOUNDARY] * context_size + list(sentence)


def make_examples(sentences, context_size):
    """Cut sentences into (context, next word) pairs of words.

    With context_size 2 the sentence 'the nurse checks the chart .' gives pairs such as
    ('.', '.') -> 'the' and ('.', 'the') -> 'nurse' and ('the', 'nurse') -> 'checks'.
    """
    examples = []
    for s in sentences:
        p = padded(s, context_size)
        for i in range(context_size, len(p)):
            examples.append((tuple(p[i - context_size:i]), p[i]))
    return examples


def to_ids(examples, index):
    """Turn word examples into two lists of numbers: the contexts (lists of ids) and the targets (ids)."""
    contexts = [[index[w] for w in ctx] for ctx, _ in examples]
    targets = [index[nxt] for _, nxt in examples]
    return contexts, targets


DATA_DIR = Path(__file__).resolve().parents[2] / "data"   # the part folder's data/ next to code/


def load(name):
    """Load one of the practice files by name: 'toy_corpus', 'caredesk_train' or 'caredesk_test'."""
    return read_sentences(DATA_DIR / f"{name}.txt")
