"""Tests for corpus.py and pca.py: the data files, the training pairs, and the map helpers."""
import numpy as np

from embedding_lab import corpus, pca


def test_corpus_size_and_vocabulary():
    train = corpus.sentences("clinical_words_train")
    words, index = corpus.vocabulary(train)
    assert len(train) == 3000 and len(words) == 89 and index[words[5]] == 5


def test_test_sentences_use_only_known_words():
    words, index = corpus.vocabulary(corpus.sentences("clinical_words_train"))
    assert all(w in index for s in corpus.sentences("clinical_words_test") for w in s)


def test_pairs_cover_every_other_word_of_a_sentence():
    words, index = corpus.vocabulary([["a", "b", "c"]])
    centre, neighbour = corpus.neighbour_pairs([["a", "b", "c"]], index)
    assert len(centre) == 6 and set(zip(centre.tolist(), neighbour.tolist())) == {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}


def test_every_group_has_ten_content_words():
    groups = corpus.word_groups()
    counts = {}
    for info in groups.values():
        counts[info["group"]] = counts.get(info["group"], 0) + 1
    assert len(counts) == 6 and set(counts.values()) == {10}


def test_pca_keeps_two_columns_and_is_repeatable():
    data = np.random.default_rng(0).normal(size=(20, 5))
    a, explained = pca.pca_2d(data)
    b, _ = pca.pca_2d(data)
    assert a.shape == (20, 2) and np.allclose(a, b) and 0 < explained[:2].sum() <= 1


def test_purity_is_one_for_separated_groups():
    group_a = np.random.default_rng(1).normal(size=(5, 3)) * 0.01 + [10, 0, 0]
    group_b = np.random.default_rng(2).normal(size=(5, 3)) * 0.01 + [0, 10, 0]
    data = np.vstack([group_a, group_b])
    assert pca.neighbour_purity(data, ["a"] * 5 + ["b"] * 5) == 1.0
    assert pca.neighbour_purity(data, ["a"] * 5 + ["b"] * 5, metric="euclid") == 1.0
