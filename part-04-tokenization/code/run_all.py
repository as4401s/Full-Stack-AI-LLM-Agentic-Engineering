"""run_all.py - run all ten demos in order and keep their outputs.

What it does: imports each demo module, runs it, and writes its printed output to
data/expected_output/<demo>.txt and its numbers to data/results/<demo>.json.
What it needs: the packages in pyproject.toml (run it with: uv run python run_all.py). The first run downloads a few
small tokenizer files; demos 5 to 8 need that download, demos 1 to 4, 9 and 10 run offline.
What to expect: one file per demo in each data folder; the whole run takes about a minute.
"""
import importlib
import time

from tokenizer_lab.report import run_and_save

DEMOS = [
    "d01_three_ways", "d02_bpe_by_hand", "d03_bytes_utf8", "d04_train_bpe", "d05_real_tokenizers",
    "d06_special_and_chat", "d07_tokens_and_money", "d08_quirks", "d09_lab_vocab_size", "d10_sentencepiece",
]


def main():
    for name in DEMOS:
        module = importlib.import_module(f"demos.{name}")
        start = time.perf_counter()
        run_and_save(name, module.main)
        print(f"-- {name} finished in {time.perf_counter() - start:.1f} s\n", flush=True)


if __name__ == "__main__":
    main()
