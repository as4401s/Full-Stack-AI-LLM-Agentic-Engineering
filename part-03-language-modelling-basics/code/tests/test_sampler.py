"""Tests for softmax, temperature, top-k, top-p and the sampling choice."""
import numpy as np

from lm_basics.sampler import sample_next, softmax, top_k_filter, top_p_filter

LOGITS = [2.0, 1.0, 0.1]


def test_softmax_matches_the_numbers_in_the_video():
    assert np.allclose(softmax(LOGITS), [0.659, 0.242, 0.099], atol=1e-3)
    assert np.allclose(softmax(LOGITS, 0.5), [0.864, 0.117, 0.019], atol=1e-3)
    assert np.allclose(softmax(LOGITS, 2.0), [0.502, 0.304, 0.194], atol=1e-3)


def test_softmax_is_stable_for_huge_scores():
    p = softmax([1000.0, 1001.0])
    assert np.isclose(p.sum(), 1.0) and np.all(np.isfinite(p))


def test_top_k_and_top_p():
    base = softmax(LOGITS)
    assert np.allclose(top_k_filter(base, 2), [0.731, 0.269, 0.0], atol=1e-3)
    assert np.allclose(top_p_filter(base, 0.9), [0.731, 0.269, 0.0], atol=1e-3)
    assert np.allclose(top_p_filter(base, 0.6), [1.0, 0.0, 0.0])
    assert np.allclose(top_p_filter(base, 0.999), base / base.sum())


def test_greedy_picks_the_highest_score():
    assert sample_next(LOGITS, temperature=0) == 0


def test_draws_follow_the_probabilities():
    rng = np.random.default_rng(0)
    draws = np.array([sample_next(LOGITS, temperature=2.0, rng=rng) for _ in range(5000)])
    share = np.bincount(draws, minlength=3) / 5000
    assert np.allclose(share, softmax(LOGITS, 2.0), atol=0.03)


def test_top_k_never_draws_outside_the_kept_words():
    rng = np.random.default_rng(1)
    draws = {sample_next(LOGITS, top_k=2, rng=rng) for _ in range(500)}
    assert draws <= {0, 1}


def test_top_p_keeps_few_words_when_sure_and_more_when_unsure():
    sure = top_p_filter([0.95, 0.03, 0.02], 0.9)
    unsure = top_p_filter([0.35, 0.30, 0.20, 0.10, 0.05], 0.9)
    assert np.count_nonzero(sure) == 1
    assert np.count_nonzero(unsure) == 4
