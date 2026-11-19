"""Tests for reading the practice text and cutting it into examples."""
from lm_basics.corpus import build_vocab, load, make_examples, padded, to_ids


def test_files_have_expected_sizes():
    assert len(load("toy_corpus")) == 3
    assert len(load("caredesk_train")) == 600
    assert len(load("caredesk_test")) == 40


def test_every_sentence_ends_with_a_full_stop():
    for name in ("toy_corpus", "caredesk_train", "caredesk_test"):
        assert all(s[-1] == "." for s in load(name))


def test_vocabulary_has_24_words_and_every_test_word_is_known():
    words, index = build_vocab(load("caredesk_train"))
    assert len(words) == 24
    assert all(w in index for s in load("caredesk_test") for w in s)


def test_padding_and_examples():
    assert padded(["the", "cat", "."], 2) == [".", ".", "the", "cat", "."]
    examples = make_examples([["the", "nurse", "."]], 2)
    assert examples == [((".", "."), "the"), ((".", "the"), "nurse"), (("the", "nurse"), ".")]


def test_ids_round_trip():
    words, index = build_vocab(load("toy_corpus"))
    contexts, targets = to_ids(make_examples(load("toy_corpus"), 2), index)
    assert len(contexts) == len(targets) == 18
    assert all(len(c) == 2 for c in contexts)


def test_test_sentences_use_pairings_missing_from_training():
    train_pairs = {(a, b) for s in load("caredesk_train") for a, b in zip(s, s[1:])}
    new_pairs = {(a, b) for s in load("caredesk_test") for a, b in zip(s, s[1:])} - train_pairs
    assert ("doctor", "checks") in new_pairs and ("clerk", "signs") in new_pairs
