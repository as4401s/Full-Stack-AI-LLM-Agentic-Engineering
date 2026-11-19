"""Demo 4 - train a small neural language model and compare it with the counting models.

What it does: turns the training sentences into (two previous words -> next word) examples, trains the embedding-plus-hidden-layer
model with the four-move loop from lesson 2, and measures perplexity on test sentences the model has never seen,
next to the counting models.
What it needs: PyTorch (uv run installs it). Takes a few seconds on a laptop CPU.
What to expect: the loss falls from about ln(24) = 3.18 to under 1; the neural model's test perplexity is lower than
the add-one bigram model's, mostly thanks to the sentences that use pairings absent from the training text.
Run it: uv run python -m demos.d04_train_neural
"""
import math

import torch

from lm_basics.corpus import build_vocab, load
from lm_basics.mlp_lm import NeuralLM, count_parameters, evaluate, make_tensors, train
from lm_basics.ngram import NGramModel
from lm_basics.report import run_and_save, save_results


def main():
    torch.manual_seed(0)
    train_s, test_s = load("caredesk_train"), load("caredesk_test")
    held = test_s[20:]
    words, index = build_vocab(train_s)
    V = len(words)
    print(f"vocabulary: {V} words; a blind guess among {V} words has loss ln({V}) = {math.log(V):.3f} and perplexity {V}")
    contexts, targets = make_tensors(train_s, index, context_size=2)
    print(f"training examples: {len(targets)}   context shape {tuple(contexts.shape)}   target shape {tuple(targets.shape)}")
    model = NeuralLM(V, context_size=2, embed_dim=2, hidden=16)
    print(f"parameters: embedding {model.embed.weight.numel()} + hidden layer {sum(p.numel() for p in model.hidden.parameters())}"
          f" + output layer {sum(p.numel() for p in model.out.parameters())} = {count_parameters(model)}")
    history = train(model, contexts, targets, steps=300, lr=0.02, weight_decay=1.0, log_every=50)
    print("\nstep   training loss")
    for step, loss in history:
        print(f"{step:4d}   {loss:.3f}")
    model.eval()
    _, ppl_all = evaluate(model, test_s, index)
    _, ppl_usual = evaluate(model, test_s[:20], index)
    _, ppl_held = evaluate(model, held, index)
    print("\nperplexity on sentences the models never saw (lower is better):")
    print(f"{'model':28s} {'all 40':>8s} {'20 usual':>9s} {'20 new pairings':>16s}")
    rows = {"neural (2 words of context)": (ppl_all, ppl_usual, ppl_held)}
    for name, ngram in (("bigram, add one", NGramModel(2, 1.0)), ("trigram, add one", NGramModel(3, 1.0)),
                        ("trigram, add 0.1", NGramModel(3, 0.1))):
        ngram.fit(train_s)
        rows[name] = (ngram.perplexity(test_s), ngram.perplexity(test_s[:20]), ngram.perplexity(held))
    for name, (a, u, h) in rows.items():
        print(f"{name:28s} {a:8.3f} {u:9.3f} {h:16.3f}")
    save_results("d04_train_neural", {"vocab": V, "parameters": count_parameters(model), "loss_history": history,
                                      "perplexity": {k: list(v) for k, v in rows.items()}})


if __name__ == "__main__":
    run_and_save("d04_train_neural", main)
