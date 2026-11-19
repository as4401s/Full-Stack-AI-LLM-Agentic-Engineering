"""Demo 10 - the lab: how much weight decay helps the model on word pairings it has not seen?

What it does: trains the same small neural model five times with weight decay 0, 0.5, 1, 2 and 4 and prints the training loss
and the test perplexity (on all 40 test sentences and on the 20 new pairings).
What it needs: PyTorch. Takes about half a minute on a laptop CPU.
What to expect: with the settings below (context_size 2, embed_dim 2), no weight decay lets the model memorise the training
pairings, so it does worst on the new ones; a moderate value does best; a very large value shrinks the numbers so much
that even the training text is predicted poorly.
Run it: uv run python -m demos.d10_lab_weight_decay
"""
import torch

from lm_basics.corpus import build_vocab, load
from lm_basics.mlp_lm import NeuralLM, evaluate, make_tensors, train
from lm_basics.report import run_and_save, save_results


def main():
    context_size = 2  # how many earlier words the model sees (your turn: try 3)
    embed_dim = 2  # how many numbers describe each word (your turn: try 4)
    train_s, test_s = load("caredesk_train"), load("caredesk_test")
    words, index = build_vocab(train_s)
    contexts, targets = make_tensors(train_s, index, context_size)
    print(f"{'weight decay':>12s} {'train loss':>11s} {'test perplexity':>16s} {'on new pairings':>16s}")
    results = {}
    for wd in (0.0, 0.5, 1.0, 2.0, 4.0):
        torch.manual_seed(0)
        model = NeuralLM(len(words), context_size=context_size, embed_dim=embed_dim, hidden=16)
        history = train(model, contexts, targets, steps=300, lr=0.02, weight_decay=wd, log_every=300)
        model.eval()
        _, all_ppl = evaluate(model, test_s, index)
        _, held_ppl = evaluate(model, test_s[20:], index)
        print(f"{wd:12.1f} {history[-1][1]:11.3f} {all_ppl:16.3f} {held_ppl:16.3f}")
        results[str(wd)] = {"train_loss": history[-1][1], "perplexity_all": all_ppl, "perplexity_new_pairings": held_ppl}
    print("\nYour turn: change context_size from 2 to 3 (or embed_dim from 2 to 4) and run it again.")
    save_results("d10_lab_weight_decay", results)


if __name__ == "__main__":
    run_and_save("d10_lab_weight_decay", main)
