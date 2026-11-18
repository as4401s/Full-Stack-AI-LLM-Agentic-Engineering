"""Demo 5 - counting the parameters of a network.

What it does: builds two networks with the MLP class and lists every parameter tensor with its shape and
size, then adds them up.
What it needs: torch.
What to expect: 82 parameters for 2 -> 16 -> 2 and 17,154 parameters for 2 -> 128 -> 128 -> 2.
Run it: uv run python -m demos.d05_parameters
"""
import torch

from dl_refresher.module_mlp import MLP, count_parameters
from dl_refresher.report import run_and_save


def describe(name: str, model: MLP):
    """Print each parameter tensor and the total."""
    print(name)
    for param_name, p in model.named_parameters():
        print(f"  {param_name:<12} shape {str(tuple(p.shape)):<12} {p.numel():>6} parameters")
    print(f"  total: {count_parameters(model)} parameters")


def main():
    torch.manual_seed(0)
    describe("small network, 2 -> 16 -> 2", MLP((16,)))
    describe("wide network, 2 -> 128 -> 128 -> 2", MLP((128, 128)))


if __name__ == "__main__":
    run_and_save("d05_parameters", main)
