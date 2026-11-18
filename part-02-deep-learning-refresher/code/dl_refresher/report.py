"""report.py - small helpers that keep every demo's output and results in files.

What it does: runs a demo while copying everything it prints into data/expected_output/<name>.txt,
rounds and saves numbers as JSON in data/results/<name>.json, and builds the little grids of
predictions used to draw decision boundaries.
What it needs: torch (only for the grid helper and tensor conversion).
What to expect: after run_all.py, the two data folders hold one file per demo.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import torch

PART_ROOT = Path(__file__).resolve().parents[2]        # the part folder: code/ and data/ sit side by side
RESULTS_DIR = PART_ROOT / "data" / "results"
EXPECTED_DIR = PART_ROOT / "data" / "expected_output"


def rounded(obj, digits: int = 4):
    """Round every float inside lists, dicts, tuples and tensors so saved files stay small and tidy."""
    if isinstance(obj, torch.Tensor):
        return rounded(obj.detach().cpu().tolist(), digits)
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {str(k): rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v, digits) for v in obj]
    return obj


def thin(history: dict, every: int) -> dict:
    """Keep every n-th entry of a history dict (the last entry is always kept)."""
    n = len(history["step"])
    keep = sorted(set(range(0, n, every)) | {n - 1})
    return {key: [values[i] for i in keep] for key, values in history.items()}


def save_results(name: str, obj, digits: int = 4) -> Path:
    """Write data/results/<name>.json with compact separators."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(rounded(obj, digits), separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def run_and_save(name: str, fn) -> None:
    """Run fn(), show what it prints, and keep a copy in data/expected_output/<name>.txt."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn()
    text = buffer.getvalue()
    sys.stdout.write(text)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    (EXPECTED_DIR / f"{name}.txt").write_text(text, encoding="utf-8")


def boundary_grid(prob_class1, n: int = 32, extent: float = 1.3) -> dict:
    """Probability of class 1 on an n-by-n grid of points (used to draw decision boundaries).

    prob_class1 maps a (N, 2) tensor to N probabilities. Row 0 is the lowest y value.
    """
    axis = torch.linspace(-extent, extent, n)
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    with torch.no_grad():
        probs = prob_class1(points).reshape(n, n)
    return {"extent": extent, "n": n, "p1": probs.tolist()}


def points_to_lists(X, y) -> dict:
    """Points and labels as plain lists, ready for JSON."""
    return {"x": X[:, 0].tolist(), "y": X[:, 1].tolist(), "label": y.tolist()}
