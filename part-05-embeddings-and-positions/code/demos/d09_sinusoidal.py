"""Demo 9 - sinusoidal position codes: a table that needs no learning.

What it does: prints the sinusoidal table for 8 positions and 8 numbers, how long each pair of numbers takes to turn once, and the dot product
between the codes of two positions, which depends only on the distance between them.
What it needs: NumPy only.
What to expect: every row is different; fast pairs change at every step, slow pairs change over thousands of positions; dot(p, p+k) is the same for every p.
Run it: uv run python -m demos.d09_sinusoidal
"""
import numpy as np

from embedding_lab import positions
from embedding_lab.report import run_and_save, save_results


def main():
    table = positions.sinusoidal_table(8, 8)
    print("sinusoidal table, 8 positions x 8 numbers (pairs: sin, cos):")
    for p, row in enumerate(table):
        print(f"  position {p}: {' '.join(f'{x:6.3f}' for x in row)}")
    print("\nhow far a pair has to travel before it repeats (wavelength = 2 pi / angle per position):")
    for i in range(4):
        step = 10000.0 ** (-2 * i / 8)
        print(f"  pair {i}: {step:.4f} radians per position -> repeats every {2 * np.pi / step:,.1f} positions")
    long_table = positions.sinusoidal_table(40, 8)
    print("\ndot product between the codes of two positions, by distance k (same for every start p):")
    results = {}
    for k in (1, 2, 5, 10):
        values = [float(long_table[p] @ long_table[p + k]) for p in range(0, 30)]
        print(f"  k = {k:2d}: p = 0 -> {values[0]:.4f}, p = 7 -> {values[7]:.4f}, p = 29 -> {values[29]:.4f}   spread {max(values) - min(values):.1e}")
        results[f"k{k}"] = values[0]
    print(f"\nsame position: dot = {float(long_table[3] @ long_table[3]):.1f} (one 1 for every pair, so 4 pairs = 4)")
    save_results("d09_sinusoidal", results)


if __name__ == "__main__":
    run_and_save("d09_sinusoidal", main)
