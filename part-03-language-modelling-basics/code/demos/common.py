"""common.py - the one trained model that demos 5 to 9 share.

What it does: trains the small neural language model with the same settings and seed as demo 4, so every
demo that needs a trained model gets exactly the same one.
What it needs: PyTorch.
"""
import torch

from lm_basics.corpus import build_vocab, load
from lm_basics.mlp_lm import NeuralLM, make_tensors, train


def trained_model(seed=0, weight_decay=1.0, steps=300, context_size=2, embed_dim=2, hidden=16):
    """Return (model, words, index, train_sentences, test_sentences) after training with the lesson's settings."""
    torch.manual_seed(seed)
    train_s, test_s = load("caredesk_train"), load("caredesk_test")
    words, index = build_vocab(train_s)
    model = NeuralLM(len(words), context_size=context_size, embed_dim=embed_dim, hidden=hidden)
    contexts, targets = make_tensors(train_s, index, context_size)
    train(model, contexts, targets, steps=steps, lr=0.02, weight_decay=weight_decay, log_every=steps)
    model.eval()
    return model, words, index, train_s, test_s
