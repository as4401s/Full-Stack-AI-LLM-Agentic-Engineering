"""run_all.py - run all nine demos in order and keep their outputs.

What it does: imports each demo module, runs it, and writes its printed output to data/expected_output/<demo>.txt and its numbers to
data/results/<demo>.json (pictures go to data/results as PNG files).
What it needs: the packages in pyproject.toml (run it with: uv run python run_all.py). The first run downloads GPT-2 (about 550 MB);
demos 1, 2, 3, 4, 7, 8 and 9 run without any download.
What to expect: one file per demo in each data folder; the whole run takes a few minutes. The times printed by demo 7 depend on the computer.
"""
import importlib
import time

from attention_lab.report import run_and_save

DEMOS = [
    "d01_attention_from_scratch", "d02_check_builtin", "d03_multihead_module", "d04_toy_maps", "d05_gpt2_map",
    "d06_head_roles", "d07_cost_curve", "d08_tiled_check", "d09_lab_question",
]


def main():
    for name in DEMOS:
        module = importlib.import_module(f"demos.{name}")
        start = time.perf_counter()
        run_and_save(name, module.main)
        print(f"-- {name} finished in {time.perf_counter() - start:.1f} s\n", flush=True)


if __name__ == "__main__":
    main()
