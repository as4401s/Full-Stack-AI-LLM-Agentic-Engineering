"""Demo 7 - the same job with nn.Module, the standard loop and the AdamW optimizer.

What it does: trains MLP(16) on the spirals with AdamW, shows the five lines of the loop at work, and keeps
the loss curve and decision-boundary snapshots.
What it needs: torch.
What to expect: 82 parameters, loss falling from about 0.7 to about 0.08 and accuracy near 98 percent within 500 steps.
Run it: uv run python -m demos.d07_train_module
"""
import torch

from dl_refresher.data import make_datasets
from dl_refresher.module_mlp import MLP, count_parameters, train_module
from dl_refresher.report import boundary_grid, run_and_save, save_results, thin

SNAPSHOT_STEPS = (0, 10, 50, 100, 500)


def main():
    train, val = make_datasets()
    torch.manual_seed(0)
    model = MLP((16,))
    print(f"nn.Module network with {count_parameters(model)} parameters, AdamW, learning rate 0.01")
    out = train_module(train, val, model, optimizer="adamw", lr=0.01, steps=500, snapshot_steps=SNAPSHOT_STEPS)
    h = out["history"]
    print(f"{'step':>5} {'train loss':>11} {'train acc':>10} {'val loss':>9} {'val acc':>8}")
    for step in SNAPSHOT_STEPS:
        i = h["step"].index(step)
        print(f"{step:>5} {h['train_loss'][i]:>11.4f} {h['train_acc'][i]:>10.3f} {h['val_loss'][i]:>9.4f} {h['val_acc'][i]:>8.3f}")
    print(f"time: {out['seconds']:.1f} seconds for 500 steps")
    grids = {}
    for step, state in out["snapshots"].items():
        probe = MLP((16,))
        probe.load_state_dict(state)
        probe.eval()
        grids[step] = boundary_grid(lambda pts, probe=probe: torch.softmax(probe(pts), dim=1)[:, 1])
    save_results("d07_train_module", {"history": thin(h, 5), "grids": grids, "seconds": out["seconds"]})


if __name__ == "__main__":
    run_and_save("d07_train_module", main)
