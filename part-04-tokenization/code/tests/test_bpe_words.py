"""Tests for byte-pair encoding by hand on the four-word corpus."""
from tokenizer_lab import bpe_words as bw

CORPUS = {"charted": 7, "dosed": 6, "ward": 8, "checked": 5}


def test_pair_counts_match_the_video():
    counts = bw.count_pairs(bw.split_into_letters(CORPUS))
    assert counts[("e", "d")] == 18 and counts[("a", "r")] == 15 and counts[("c", "h")] == 12
    assert len(counts) == 15


def test_five_merges():
    merges, steps = bw.train(CORPUS, 5)
    assert [a + b for a, b in merges] == ["ed", "ar", "ch", "war", "ward"]
    assert [s["count"] for s in steps] == [18, 15, 12, 8, 8]


def test_tie_goes_to_the_pair_met_first():
    words = bw.split_into_letters(CORPUS)
    for pair in [("e", "d"), ("a", "r"), ("c", "h")]:
        words = bw.merge(words, pair)
    counts = bw.count_pairs(words)
    assert counts[("w", "ar")] == counts[("ar", "d")] == 8
    assert bw.most_frequent_pair(words) == ("w", "ar")


def test_vocabulary_grows_from_eleven_letters_to_sixteen_pieces():
    merges, _ = bw.train(CORPUS, 5)
    vocab = bw.vocabulary(CORPUS, merges)
    assert len(vocab) == 16 and len([v for v in vocab if len(v) == 1]) == 11


def test_new_words_and_round_trip():
    merges, _ = bw.train(CORPUS, 5)
    assert bw.encode_word("warded", merges) == ["ward", "ed"]
    assert bw.encode_word("waxed", merges) == ["w", "a", "x", "ed"]
    for word in ("warded", "waxed", "charted", "dosed"):
        assert bw.decode_pieces(bw.encode_word(word, merges)) == word


def test_training_stops_when_nothing_can_merge():
    merges, _ = bw.train({"ab": 1}, 10)
    assert len(merges) == 1
