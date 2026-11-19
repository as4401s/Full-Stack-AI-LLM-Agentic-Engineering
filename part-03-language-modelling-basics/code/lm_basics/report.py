"""report.py - small helpers that keep every demo's output and results in files.

What it does
    Runs a demo while copying everything it prints into data/expected_output/<name>.txt, and saves
    rounded numbers as JSON in data/results/<name>.json.

What it needs
    Only the Python standard library (NumPy values are converted to plain lists).

What to expect
    After run_all.py, the two data folders hold one file per demo.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

PART_ROOT = Path(__file__).resolve().parents[2]        # the part folder: code/ and data/ sit side by side
DATA_DIR = PART_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
EXPECTED_DIR = DATA_DIR / "expected_output"


def rounded(obj, digits=4):
    """Round every float inside lists, dicts, tuples and arrays so saved files stay small and tidy."""
    if hasattr(obj, "tolist"):                          # NumPy arrays and PyTorch tensors
        return rounded(obj.tolist(), digits)
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {str(k): rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v, digits) for v in obj]
    return obj


def save_results(name, obj, digits=4):
    """Write data/results/<name>.json with compact separators."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(rounded(obj, digits), separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def run_and_save(name, fn):
    """Run fn(), show what it prints, and keep a copy in data/expected_output/<name>.txt."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn()
    text = buffer.getvalue()
    sys.stdout.write(text)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    (EXPECTED_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
