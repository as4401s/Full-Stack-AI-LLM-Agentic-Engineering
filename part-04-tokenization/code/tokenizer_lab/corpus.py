"""corpus.py - read the invented text files and cut them into words, lines and letters.

What it does
    Small helpers shared by the demos: read a text file from ../data, split it into words or lines, and measure it.
What it needs
    Only the Python standard library.
"""
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"


def read(name):
    """Return the whole text of data/<name>.txt."""
    return (DATA / f"{name}.txt").read_text(encoding="utf-8")


def lines(name):
    """Return the non-empty lines of data/<name>.txt."""
    return [ln for ln in read(name).splitlines() if ln.strip()]


def letters():
    """Return the three invented discharge letters as a dict: {'english': ..., 'german': ..., 'python': ...}."""
    out, key = {}, None
    for ln in read("letters").splitlines():
        if ln.startswith("### "):
            key = ln[4:].lower()
            out[key] = []
        elif key:
            out[key].append(ln)
    return {k: "\n".join(v).strip("\n") for k, v in out.items()}


def words(text):
    """Split on whitespace: the one-word-per-token shortcut we used in lesson 3."""
    return text.split()
