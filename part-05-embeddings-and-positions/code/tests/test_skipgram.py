"""Tests for skipgram.py (need PyTorch; skipped when it is not installed)."""
import math

import pytest

pytest.importorskip("torch")

from embedding_lab import corpus, pca, skipgram  # noqa: E402


def small_pairs():
    sentences = corpus.sentences("clinical_words_train")[:600]
    words, index = corpus.vocabulary(corpus.sentences("clinical_words_train"))
    centre, neighbour = corpus.neighbour_pairs(sentences, index)
    return words, centre, neighbour


def test_training_lowers_the_loss_below_random_guessing():
    words, centre, neighbour = small_pairs()
    run = skipgram.train(centre, neighbour, len(words), dim=8, epochs=3, seed=0)
    assert run["history"][-1] < run["history"][0] < math.log(len(words)) + 0.5
    assert run["history"][-1] < math.log(len(words))
    assert run["table"].shape == (len(words), 8)


def test_the_same_seed_gives_the_same_table():
    words, centre, neighbour = small_pairs()
    a = skipgram.train(centre, neighbour, len(words), dim=4, epochs=1, seed=3)
    b = skipgram.train(centre, neighbour, len(words), dim=4, epochs=1, seed=3)
    assert (a["table"] == b["table"]).all()


def test_lookup_returns_the_row():
    import torch
    layer = skipgram.SkipGram(10, 4)
    ids = torch.tensor([3, 7])
    assert torch.equal(layer.table(ids), layer.table.weight[ids])


def test_words_of_one_body_system_group_after_training():
    sentences = corpus.sentences("clinical_words_train")
    words, index = corpus.vocabulary(sentences)
    centre, neighbour = corpus.neighbour_pairs(sentences, index)
    run = skipgram.train(centre, neighbour, len(words), dim=8, epochs=20, lr=0.02, seed=0)
    groups = corpus.word_groups()
    content = [w for w in words if w in groups]
    matrix = run["table"][[index[w] for w in content]]
    assert pca.neighbour_purity(matrix, [groups[w]["group"] for w in content]) > 0.8
