"""Demo 1 - one parameter, one loss, walking downhill.

What it does: fits the rule y = 2x with a single parameter w from one example (x = 1, y = 2). The loss is
(w * x - y) squared, and PyTorch works out its gradient for us. It runs twice: with a gentle learning rate
(0.1) and with one that is too large (1.1), so we can watch both a smooth walk and an overshoot.
What it needs: torch.
What to expect: with 0.1 the parameter goes 0.4, 0.72, 0.976 ... towards 2; with 1.1 it jumps past 2 and
the swings grow bigger.
Run it: uv run python -m demos.d01_one_weight
"""
import torch

from dl_refresher.report import run_and_save, save_results

X, Y = 1.0, 2.0    # one training example for the rule y = 2x: input 1, target 2 (best parameter value is 2)
STEPS = 6          # how many downhill steps to take


def walk(lr: float):
    """Take STEPS gradient-descent steps on loss = (w * X - Y) ** 2 and return the parameter path and loss path."""
    w = torch.tensor(0.0, requires_grad=True)                  # the parameter, starting at 0
    w_path, loss_path = [w.item()], [((w * X - Y) ** 2).item()]
    for step in range(1, STEPS + 1):
        loss = (w * X - Y) ** 2                                # how far is the prediction w * x from the target y?
        loss.backward()                                        # gradient = 2 * x * (w * x - y)
        if step == 1:
            print(f"  gradient at the start: {w.grad.item():.1f}  (the loss falls fastest by moving up)")
        with torch.no_grad():
            w -= lr * w.grad                                   # new parameter = old parameter - learning rate * gradient
            w.grad = None                                      # clear for the next step
        w_path.append(w.item())
        loss_path.append(((w * X - Y) ** 2).item())
        print(f"  step {step}: w = {w.item():8.4f}   loss = {loss_path[-1]:9.4f}")
    return w_path, loss_path


def main():
    results = {}
    for lr in (0.1, 1.1):
        print(f"learning rate {lr}")
        w_path, loss_path = walk(lr)
        results[str(lr)] = {"w": w_path, "loss": loss_path}
    save_results("d01_one_weight", results)


if __name__ == "__main__":
    run_and_save("d01_one_weight", main)
