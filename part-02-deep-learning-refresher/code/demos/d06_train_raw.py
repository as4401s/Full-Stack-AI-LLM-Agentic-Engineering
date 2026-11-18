"""Demo 6 - training a network by hand on the two spirals, with plain tensors only.

What it does: makes the spirals, trains the raw-tensor network (forward, loss, backward, manual step) with
plain gradient descent, and keeps the loss curve plus snapshots of what the network believes at chosen steps.
What it needs: torch.
What to expect: training loss falls from about 0.7 to about 0.06 and accuracy climbs to about 98 percent.
Run it: uv run python -m demos.d06_train_raw
"""
import torch

from dl_refresher.data import make_datasets
from dl_refresher.raw_mlp import forward, train_raw
from dl_refresher.report import boundary_grid, points_to_lists, run_and_save, save_results, thin

SNAPSHOT_STEPS = (0, 10, 50, 100, 300, 1000)


def main():
    train, val = make_datasets()
    out = train_raw(train, val, hidden=16, lr=1.0, steps=1000, snapshot_steps=SNAPSHOT_STEPS)
    h = out["history"]
    print("raw tensors, 2 -> 16 -> 2, plain gradient descent, learning rate 1.0")
    print(f"{'step':>5} {'train loss':>11} {'train acc':>10} {'val loss':>9} {'val acc':>8}")
    for step in SNAPSHOT_STEPS:
        i = h["step"].index(step)
        print(f"{step:>5} {h['train_loss'][i]:>11.4f} {h['train_acc'][i]:>10.3f} {h['val_loss'][i]:>9.4f} {h['val_acc'][i]:>8.3f}")
    print(f"time: {out['seconds']:.1f} seconds for 1000 steps")
    grids = {}
    for step, params in out["snapshots"].items():
        grids[step] = boundary_grid(lambda pts, params=params: torch.softmax(forward(params, pts), dim=1)[:, 1])
    save_results("d06_train_raw", {"history": thin(h, 5), "train_points": points_to_lists(*train),
                                   "val_points": points_to_lists(*val), "grids": grids, "seconds": out["seconds"]})


if __name__ == "__main__":
    run_and_save("d06_train_raw", main)
