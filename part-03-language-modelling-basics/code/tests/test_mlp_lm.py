"""Tests for the neural language model (these need PyTorch)."""
import math

import pytest

torch = pytest.importorskip("torch")

from lm_basics.corpus import build_vocab, load
from lm_basics.mlp_lm import NeuralLM, count_parameters, evaluate, make_tensors, next_word_probs, train


def small_setup():
    train_s = load("caredesk_train")
    words, index = build_vocab(train_s)
    return train_s, words, index


def test_shapes_and_parameter_count():
    _, words, index = small_setup()
    model = NeuralLM(len(words), context_size=2, embed_dim=2, hidden=16)
    out = model(torch.zeros(5, 2, dtype=torch.long))
    assert out.shape == (5, 24)
    assert count_parameters(model) == 24 * 2 + (4 * 16 + 16) + (16 * 24 + 24)   # 536


def test_untrained_model_is_about_a_blind_guess():
    train_s, words, index = small_setup()
    torch.manual_seed(0)
    model = NeuralLM(len(words))
    _, ppl = evaluate(model, train_s, index)
    assert 10 < ppl < 60          # near the blind-guess perplexity of 24


def test_training_lowers_the_loss():
    train_s, words, index = small_setup()
    torch.manual_seed(0)
    model = NeuralLM(len(words))
    contexts, targets = make_tensors(train_s, index, 2)
    history = train(model, contexts, targets, steps=150, log_every=150)
    assert history[-1][1] < 0.5 * history[0][1]


def test_probabilities_add_up_to_one():
    train_s, words, index = small_setup()
    torch.manual_seed(0)
    model = NeuralLM(len(words))
    p = next_word_probs(model, ["the", "nurse"], words, index)
    assert math.isclose(sum(p.values()), 1.0, rel_tol=1e-5)


def test_trained_model_gives_an_unseen_pairing_a_real_probability():
    train_s, words, index = small_setup()
    torch.manual_seed(0)
    model = NeuralLM(len(words))
    contexts, targets = make_tensors(train_s, index, 2)
    train(model, contexts, targets, steps=300, weight_decay=1.0, log_every=300)
    p = next_word_probs(model, ["the", "doctor"], words, index)
    assert p["checks"] > 0.005          # plain counts give exactly 0 for 'doctor checks'
