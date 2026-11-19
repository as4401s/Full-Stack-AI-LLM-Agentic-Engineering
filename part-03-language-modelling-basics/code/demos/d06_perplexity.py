"""Demo 6 - from probabilities to loss to perplexity.

What it does: shows -ln(p) for a few probabilities, the average loss and perplexity of a four-step example,
what a loss of 2.0 means, and the blind-guess baseline for our 24-word vocabulary.
What it needs: only the Python standard library.
What to expect: perplexity 3.364 for the four steps (0.5, 0.25, 0.5, 0.125); a loss of 2.0 means about 7.389 choices.
Run it: uv run python -m demos.d06_perplexity
"""
import math

from lm_basics.report import run_and_save, save_results


def main():
    print("surprise for one word = -ln(probability the model gave the right word)")
    for p in (1.0, 0.5, 0.25, 0.125, 0.1, 0.01):
        print(f"  p = {p:<6}  loss = {-math.log(p) + 0.0:.3f}")
    probs = [0.5, 0.25, 0.5, 0.125]
    losses = [-math.log(p) for p in probs]
    mean = sum(losses) / len(losses)
    print(f"\nfour steps, probabilities {probs}")
    print("  losses:", [round(x, 3) for x in losses], f" average = {mean:.4f}")
    print(f"  perplexity = e^{mean:.4f} = {math.exp(mean):.3f}   (the same as 128 ** 0.25 = {128 ** 0.25:.3f})")
    print("\nloss to perplexity:")
    for loss in (0.0, 0.693, 1.0, 2.0, 3.0, math.log(24)):
        print(f"  loss {loss:5.3f}  ->  perplexity {math.exp(loss):6.3f}")
    print(f"\nblind guess among 24 words: loss ln(24) = {math.log(24):.3f}, perplexity 24")
    save_results("d06_perplexity", {"four_step_perplexity": math.exp(mean), "perplexity_of_loss_2": math.exp(2.0)})


if __name__ == "__main__":
    run_and_save("d06_perplexity", main)
