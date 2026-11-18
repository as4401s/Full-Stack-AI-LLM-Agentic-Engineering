"""Demo 8 - overfitting, and two ways to calm it: weight decay and dropout.

What it does: gives a wide network (2 -> 128 -> 128 -> 2) only 25 training points per class, then trains
four copies: plain, with weight decay, with dropout, and with both. Validation data (200 points per
class) shows how well each copy handles points it has not studied.
What it needs: torch.
What to expect: the plain copy reaches a tiny training loss while its validation loss stops improving and
then rises; the calmer copies keep validation loss lower. A short weight-decay sweep follows.
Run it: uv run python -m demos.d08_overfit
"""
import torch

from dl_refresher.data import two_spirals
from dl_refresher.module_mlp import MLP, train_module
from dl_refresher.report import boundary_grid, points_to_lists, run_and_save, save_results, thin

NOISE = 0.12
STEPS = 600
WEIGHT_DECAY = 0.1
DROPOUT = 0.3
CONFIGS = {
    "plain": dict(weight_decay=0.0, dropout=0.0),
    "weight decay": dict(weight_decay=WEIGHT_DECAY, dropout=0.0),
    "dropout": dict(weight_decay=0.0, dropout=DROPOUT),
    "both": dict(weight_decay=WEIGHT_DECAY, dropout=DROPOUT),
}


def run_config(train, val, weight_decay: float, dropout: float, snapshot_steps=()):
    """Train one 2-128-128-2 network with the given settings and return the training result."""
    torch.manual_seed(0)
    model = MLP((128, 128), dropout=dropout)
    return train_module(train, val, model, optimizer="adamw", lr=0.01, weight_decay=weight_decay,
                        steps=STEPS, snapshot_steps=snapshot_steps), model


def main():
    train = two_spirals(25, noise=NOISE, seed=0)
    val = two_spirals(200, noise=NOISE, seed=1)
    print(f"{len(train[1])} training points, {len(val[1])} validation points, noise {NOISE}, {STEPS} steps")
    print(f"{'setting':<14} {'train loss':>10} {'train acc':>10} {'val loss':>9} {'val acc':>8} {'best val loss':>14} {'at step':>8}")
    results = {}
    for name, cfg in CONFIGS.items():
        out, model = run_config(train, val, **cfg)
        h = out["history"]
        best = min(range(len(h["val_loss"])), key=lambda i: h["val_loss"][i])
        print(f"{name:<14} {h['train_loss'][-1]:>10.4f} {h['train_acc'][-1]:>10.3f} {h['val_loss'][-1]:>9.4f} "
              f"{h['val_acc'][-1]:>8.3f} {h['val_loss'][best]:>14.4f} {h['step'][best]:>8}")
        model.eval()
        grid = boundary_grid(lambda pts, model=model: torch.softmax(model(pts), dim=1)[:, 1])
        results[name] = {"history": thin(h, 5), "grid": grid}
    print()
    print("weight decay sweep (AdamW, no dropout): how much is enough?")
    print(f"{'weight decay':>12} {'train loss':>11} {'val loss':>9} {'val acc':>8}")
    sweep = {}
    for wd in (0.0, 0.01, 0.1, 1.0):
        out, _ = run_config(train, val, weight_decay=wd, dropout=0.0)
        h = out["history"]
        sweep[str(wd)] = {"train_loss": h["train_loss"][-1], "val_loss": h["val_loss"][-1], "val_acc": h["val_acc"][-1]}
        print(f"{wd:>12} {h['train_loss'][-1]:>11.4f} {h['val_loss'][-1]:>9.4f} {h['val_acc'][-1]:>8.3f}")
    save_results("d08_overfit", {"configs": results, "sweep": sweep, "train_points": points_to_lists(*train),
                                 "val_points": points_to_lists(*val)})


if __name__ == "__main__":
    run_and_save("d08_overfit", main)
