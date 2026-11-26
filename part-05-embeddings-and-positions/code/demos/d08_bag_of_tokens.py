"""Demo 8 - why order needs a code: attention alone sees a bag of tokens.

What it does: runs "dog bites man" and "man bites dog" through one attention head (plain NumPy), first with only word vectors, then with
sinusoidal position codes added.
What it needs: NumPy only.
What to expect: without position codes every word gets exactly the same output in both sentences; with position codes the outputs differ.
Run it: uv run python -m demos.d08_bag_of_tokens
"""
import numpy as np

from embedding_lab import positions
from embedding_lab.report import run_and_save, save_results


def main():
    result = positions.bag_demo()
    a, b = result["dog bites man"], result["man bites dog"]
    print("output vector for 'bites' (first 4 of 8 numbers)")
    for setting in ("plain", "with_positions"):
        label = "word vectors only" if setting == "plain" else "word vectors + position codes"
        va, vb = a[setting]["bites"], b[setting]["bites"]
        print(f"  {label}")
        print(f"    in 'dog bites man': {np.round(va[:4], 3).tolist()}")
        print(f"    in 'man bites dog': {np.round(vb[:4], 3).tolist()}   identical: {bool(np.allclose(va, vb))}")
    same_dog = np.allclose(a["plain"]["dog"], b["plain"]["dog"])
    print(f"\nwithout position codes the output for 'dog' is the same whether the dog comes first or last: {same_dog}")
    print(f"largest difference with position codes for 'dog': {float(np.abs(a['with_positions']['dog'] - b['with_positions']['dog']).max()):.3f}")
    save_results("d08_bag_of_tokens", {"plain_identical": bool(np.allclose(a["plain"]["bites"], b["plain"]["bites"])),
                                       "positions_identical": bool(np.allclose(a["with_positions"]["bites"], b["with_positions"]["bites"]))})


if __name__ == "__main__":
    run_and_save("d08_bag_of_tokens", main)
