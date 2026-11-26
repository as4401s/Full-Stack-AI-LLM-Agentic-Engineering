"""mapmaker.py - train (once per run) the small word map used by several demos.

What it does
    trained_map() trains the skip-gram model of skipgram.py on the invented clinical sentences and remembers the answer for the
    rest of the run, so demos 4, 5, 6 and 12 share one training run when they are started together by run_all.py.
What it needs
    PyTorch and NumPy.
What to expect
    A dict with the sorted word list, the word -> id dict, the learned table (words x dim) and the loss numbers.
"""
from functools import lru_cache

from embedding_lab import corpus, skipgram


@lru_cache(maxsize=None)
def trained_map(dim=8, epochs=20, seed=0):
    """Train on data/clinical_words_train.txt, check on data/clinical_words_test.txt, and return the results as a dict."""
    train_sentences = corpus.sentences("clinical_words_train")
    words, index = corpus.vocabulary(train_sentences)
    centre, neighbour = corpus.neighbour_pairs(train_sentences, index)
    test_pairs = corpus.neighbour_pairs(corpus.sentences("clinical_words_test"), index)
    result = skipgram.train(centre, neighbour, len(words), dim, epochs=epochs, seed=seed, test_pairs=test_pairs)
    return {"words": words, "index": index, "pairs": len(centre), **result}
