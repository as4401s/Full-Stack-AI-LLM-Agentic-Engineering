"""run_all.py - run all eleven demos in order and keep their outputs.

What it does: imports each demo module, runs it, and writes its printed output to
data/expected_output/<demo>.txt and its numbers to data/results/<demo>.json.
What it needs: torch (run it with: uv run python run_all.py).
What to expect: one file per demo in each data folder; every demo finishes within a few seconds on a laptop CPU.
"""
import importlib
import time

from dl_refresher.report import run_and_save

DEMOS = [
    "d01_one_weight", "d02_backprop", "d03_tensors_shapes", "d04_softmax", "d05_parameters",
    "d06_train_raw", "d07_train_module", "d08_overfit", "d09_optimizers", "d10_precision_device", "d11_lab_width",
]


def main():
    for name in DEMOS:
        module = importlib.import_module(f"demos.{name}")
        start = time.perf_counter()
        run_and_save(name, module.main)
        print(f"-- {name} finished in {time.perf_counter() - start:.1f} s\n", flush=True)


if __name__ == "__main__":
    main()
