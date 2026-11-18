"""Demo 11 - the lab question: how many hidden units does the spiral job need?

What it does: trains the same network with 2, 4, 8, 16 and 64 hidden units (AdamW, learning rate 0.01,
500 steps each) and prints the number of knobs, the final training loss and the accuracy on the training and
validation points.
What it needs: torch.
What to expect: very small networks have too few bends to follow the spirals; around 16 hidden units is
plenty; 64 hidden units fit as well or slightly better with four times more knobs.
Run it: uv run python -m demos.d11_lab_width
"""
import torch

from dl_refresher.data import make_datasets
from dl_refresher.module_mlp import MLP, count_parameters, train_module
from dl_refresher.report import run_and_save, save_results


def main():
    train, val = make_datasets()
    print("hidden units: how many does the spiral job need? (AdamW, learning rate 0.01, 500 steps)")
    print(f"{'hidden':>7} {'knobs':>6} {'train loss':>11} {'train acc':>10} {'val acc':>8}")
    rows = {}
    for hidden in (2, 4, 8, 16, 64):
        torch.manual_seed(0)
        model = MLP((hidden,))
        h = train_module(train, val, model, optimizer="adamw", lr=0.01, steps=500)["history"]
        rows[str(hidden)] = {"knobs": count_parameters(model), "train_loss": h["train_loss"][-1],
                             "train_acc": h["train_acc"][-1], "val_acc": h["val_acc"][-1]}
        print(f"{hidden:>7} {count_parameters(model):>6} {h['train_loss'][-1]:>11.4f} {h['train_acc'][-1]:>10.3f} {h['val_acc'][-1]:>8.3f}")
    save_results("d11_lab_width", rows)


if __name__ == "__main__":
    run_and_save("d11_lab_width", main)
