"""Demo 2 - backpropagation on a network small enough to check by hand.

What it does: builds a tiny network, h = relu(w1 * x + b1) and prediction = w2 * h + b2, with the loss
(prediction - y) squared. loss.backward() fills in one gradient per parameter. Then it walks the recorded chain
(grad_fn) that PyTorch kept, and checks every gradient by nudging the parameter a tiny amount.
What it needs: torch.
What to expect: gradients w1 = -12, b1 = -6, w2 = -18, b2 = -6 for the starting values below.
Run it: uv run python -m demos.d02_backprop
"""
import torch

from dl_refresher.report import run_and_save, save_results

X, Y = 2.0, 6.0                                         # one training example: input 2, target 6
START = {"w1": 1.0, "b1": 1.0, "w2": 1.0, "b2": 0.0}   # starting parameter values


def loss_for(parameters: dict) -> torch.Tensor:
    """Forward pass and loss for the tiny network."""
    hidden = torch.relu(parameters["w1"] * X + parameters["b1"])  # the hidden unit
    prediction = parameters["w2"] * hidden + parameters["b2"]     # the output
    return (prediction - Y) ** 2                        # squared error


def walk_chain(node, depth=0, seen=None):
    """Print the chain of operations PyTorch recorded, from the loss back towards the parameters."""
    seen = set() if seen is None else seen
    if node is None or id(node) in seen:
        return
    seen.add(id(node))
    print("  " * depth + type(node).__name__)
    for child, _ in node.next_functions:
        walk_chain(child, depth + 1, seen)


def main():
    parameters = {name: torch.tensor(value, requires_grad=True) for name, value in START.items()}
    loss = loss_for(parameters)
    print(f"loss = {loss.item():.1f}   (prediction 3, target 6, miss 3, squared 9)")
    print("the recorded chain, from the loss back to the parameters:")
    walk_chain(loss.grad_fn)
    loss.backward()                                     # fill in .grad for every parameter
    grads = {name: parameter.grad.item() for name, parameter in parameters.items()}
    print("gradients from backward():")
    for name, grad in grads.items():
        print(f"  d loss / d {name} = {grad:6.1f}")
    print("nudge check: move one parameter by 0.001 and see how the loss changes")
    base = loss.item()
    nudged = {}
    for name in START:
        moved = {k: torch.tensor(v + (0.001 if k == name else 0.0)) for k, v in START.items()}
        slope = (loss_for(moved).item() - base) / 0.001
        nudged[name] = slope
        print(f"  {name}: (new loss - old loss) / 0.001 = {slope:7.2f}   gradient = {grads[name]:6.1f}")
    save_results("d02_backprop", {"loss": base, "grads": grads, "nudged": nudged})


if __name__ == "__main__":
    run_and_save("d02_backprop", main)
