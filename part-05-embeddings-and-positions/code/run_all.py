"""run_all.py - run all thirteen demos in order and keep their outputs.

What it does: imports each demo module, runs it, and writes its printed output to data/expected_output/<demo>.txt and its numbers to
data/results/<demo>.json. Training runs of the word map are shared between demos 4, 5, 6 and 12 (see embedding_lab/mapmaker.py).
What it needs: the packages in pyproject.toml (run it with: uv run python run_all.py). The first run downloads the small public models and
vectors (about 200 MB in total); demos 2, 4, 6, 8, 9 and 12 run without any download, and demos 11 and 13 read only config files of a few kilobytes.
What to expect: one file per demo in each data folder; the whole run takes a few minutes.
"""
import importlib
import time

from embedding_lab.report import run_and_save

DEMOS = [
    "d01_search_hook", "d02_table_lookup", "d03_real_tables", "d04_train_map", "d05_analogies", "d06_dot_cosine",
    "d07_sentence_vs_token", "d08_bag_of_tokens", "d09_sinusoidal", "d10_learned_positions", "d11_rope", "d12_lab_dimension", "d13_rope_vs_qwen",
]


def main():
    for name in DEMOS:
        module = importlib.import_module(f"demos.{name}")
        start = time.perf_counter()
        run_and_save(name, module.main)
        print(f"-- {name} finished in {time.perf_counter() - start:.1f} s\n", flush=True)


if __name__ == "__main__":
    main()
