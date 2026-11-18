"""Demo 4 - from raw scores to probabilities, and the loss for one example.

What it does: turns three raw scores (logits) into probabilities with softmax, takes the cross-entropy
loss for the correct class, checks the numbers against PyTorch's own functions, and shows why the
log-sum-exp form is used for large scores.
What it needs: torch.
What to expect: probabilities 0.659, 0.242, 0.099 and a loss of 0.417 for class 0.
Run it: uv run python -m demos.d04_softmax
"""
import torch
import torch.nn.functional as F

from dl_refresher.report import run_and_save


def main():
    logits = torch.tensor([2.0, 1.0, 0.1])               # raw scores for three classes
    exps = torch.exp(logits)                             # make every score positive
    probs = exps / exps.sum()                            # divide by the total so they add up to 1
    print("scores       ", [round(v, 1) for v in logits.tolist()])
    print("exp(scores)  ", [round(v, 3) for v in exps.tolist()])
    print("probabilities", [round(v, 3) for v in probs.tolist()], " sum =", round(probs.sum().item(), 3))
    loss = -torch.log(probs[0])                          # cross-entropy when the correct class is 0
    print(f"loss for class 0 = -ln({probs[0].item():.3f}) = {loss.item():.3f}")
    print("matches F.softmax      :", torch.allclose(probs, F.softmax(logits, dim=0)))
    print("matches F.cross_entropy:", torch.allclose(loss, F.cross_entropy(logits[None], torch.tensor([0]))))
    big = torch.tensor([1000.0, 1001.0])
    print("naive exp of 1000      :", torch.exp(big[0]).item(), "(too large to store)")
    print("log-sum-exp of 1000,1001:", round(torch.logsumexp(big, dim=0).item(), 4), "(stays finite)")
    print("stable softmax         :", [round(v, 3) for v in F.softmax(big, dim=0).tolist()])


if __name__ == "__main__":
    run_and_save("d04_softmax", main)
