"""Demo 9 - optimizers, learning rates and a warmup-then-cosine schedule.

What it does: trains the small network (2 -> 16 -> 2) with plain SGD, SGD with momentum and AdamW; sweeps the
AdamW learning rate; and shows how a warmup-then-cosine schedule changes the learning rate step by step.
What it needs: torch.
What to expect: AdamW falls fastest at its usual 0.01; a learning rate that is too small crawls and one that
is too large bounces; the schedule climbs for 20 steps, glides down to zero and rescues the bold rate of 1.0.
Run it: uv run python -m demos.d09_optimizers
"""
import torch

from dl_refresher.data import make_datasets
from dl_refresher.module_mlp import MLP, train_module, warmup_cosine
from dl_refresher.report import run_and_save, save_results, thin

STEPS = 300
CHECK_STEPS = (0, 10, 50, 100, 200, 300)


def fit(train, val, optimizer, lr, steps=STEPS, **extra):
    """Train a fresh small network with one optimizer setting and return the history."""
    torch.manual_seed(0)
    return train_module(train, val, MLP((16,)), optimizer=optimizer, lr=lr, steps=steps, **extra)["history"]


def main():
    train, val = make_datasets()
    print("three optimizers, same network and data, 300 steps (training loss)")
    print(f"{'optimizer':<26}" + "".join(f"{s:>8}" for s in CHECK_STEPS))
    runs = {}
    for label, optimizer, lr in (("sgd lr 0.1", "sgd", 0.1), ("momentum lr 0.1", "momentum", 0.1), ("adamw lr 0.01", "adamw", 0.01)):
        h = fit(train, val, optimizer, lr)
        runs[label] = thin(h, 5)
        print(f"{label:<26}" + "".join(f"{h['train_loss'][h['step'].index(s)]:>8.3f}" for s in CHECK_STEPS))
    print()
    print("learning rate sweep with AdamW, 300 steps")
    print(f"{'learning rate':>13} {'train loss':>11} {'val acc':>8}")
    sweep = {}
    for lr in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        h = fit(train, val, "adamw", lr)
        sweep[str(lr)] = {"train_loss": h["train_loss"][-1], "val_acc": h["val_acc"][-1]}
        print(f"{lr:>13} {h['train_loss'][-1]:>11.4f} {h['val_acc'][-1]:>8.3f}")
    print()
    factor = warmup_cosine(20, STEPS)
    print("warmup 20 steps, then cosine to zero over 300 steps, base learning rate 1.0")
    for s in (0, 10, 19, 20, 100, 200, 299):
        print(f"  step {s:>3}: learning rate = {1.0 * factor(s):.5f}")
    flat = fit(train, val, "adamw", 1.0)
    sched = fit(train, val, "adamw", 1.0, warmup=20, cosine=True)
    print(f"final train loss, constant 1.0 : {flat['train_loss'][-1]:.4f}   val acc {flat['val_acc'][-1]:.3f}")
    print(f"final train loss, with schedule: {sched['train_loss'][-1]:.4f}   val acc {sched['val_acc'][-1]:.3f}")
    save_results("d09_optimizers", {"runs": runs, "sweep": sweep, "schedule": thin(sched, 5), "flat": thin(flat, 5)})


if __name__ == "__main__":
    run_and_save("d09_optimizers", main)
