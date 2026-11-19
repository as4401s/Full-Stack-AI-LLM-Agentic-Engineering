"""run_all.py - run all eleven demos in order and keep their outputs.

What it does: imports each demo module, runs it, and writes its printed output to
data/expected_output/<demo>.txt and its numbers to data/results/<demo>.json.
What it needs: torch and numpy (run it with: uv run python run_all.py).
What to expect: one file per demo in each data folder; every demo finishes within about half a minute on a laptop CPU.
"""
import importlib
import time

from lm_basics.report import run_and_save

DEMOS = [
    "d01_counts_chain_rule", "d02_generate_dice", "d03_unseen_pairs", "d04_train_neural", "d05_generalise",
    "d06_perplexity", "d07_sampling", "d08_generate_from_model", "d09_chat_template", "d10_lab_weight_decay",
    "d11_suggestions",
]


def main():
    for name in DEMOS:
        module = importlib.import_module(f"demos.{name}")
        start = time.perf_counter()
        run_and_save(name, module.main)
        print(f"-- {name} finished in {time.perf_counter() - start:.1f} s\n", flush=True)


if __name__ == "__main__":
    main()
