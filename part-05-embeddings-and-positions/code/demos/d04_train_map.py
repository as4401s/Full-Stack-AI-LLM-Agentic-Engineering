"""Demo 4 - train our own embedding table and draw its 2-D map.

What it does: trains the skip-gram model (skipgram.py) on 3000 invented clinical sentences about six body systems, prints the loss curve,
checks it on 300 unseen sentences, lists nearest neighbours, squashes the 8 numbers per word into 2 with PCA and saves the map.
What it needs: torch and NumPy; matplotlib is optional (it draws a PNG next to the saved map).
What to expect: the loss falls below the random-guess level ln(89); words of one body system sit together on the map.
Run it: uv run python -m demos.d04_train_map
"""
import math

import numpy as np
import torch

from embedding_lab import corpus, pca, vectors
from embedding_lab.mapmaker import trained_map
from embedding_lab.report import RESULTS_DIR, run_and_save, save_results


def main():
    print("torch", torch.__version__)
    run = trained_map()
    words, index, table = run["words"], run["index"], run["table"]
    groups = corpus.word_groups()
    print(f"vocabulary: {len(words)} words, {run['pairs']:,} (word, neighbour) pairs, {table.shape[1]} numbers per word "
          f"-> a table of {table.size} numbers")
    print(f"random guessing among {len(words)} words costs ln({len(words)}) = {math.log(len(words)):.3f} per pair\n")
    print("loss after each epoch (average over all pairs):")
    for epoch, loss in enumerate(run["history"], start=1):
        if epoch in (1, 2, 5, 10, 15, 20):
            print(f"  epoch {epoch:2d}: {loss:.4f}")
    print(f"train loss {run['train_loss']:.4f}, loss on 300 unseen sentences {run['test_loss']:.4f}\n")

    content = [w for w in words if w in groups]
    matrix = table[[index[w] for w in content]]
    labels = [groups[w]["group"] for w in content]
    purity = pca.neighbour_purity(matrix, labels)
    print(f"{len(content)} content words in {len(set(labels))} body systems; nearest neighbour is from the same system for {purity:.0%} of them\n")
    for word in ("kidney", "heart", "ramipril", "nephritis"):
        near = ", ".join(f"{w} {c:.2f}" for w, c in vectors.nearest(word, words, index, table, k=4))
        print(f"  closest to {word!r}: {near}")

    coords, explained = pca.pca_2d(matrix)
    flat = pca.neighbour_purity(coords, labels, metric="euclid")
    print(f"\non the 2-D map the nearest neighbour is from the same body system for {flat:.0%} of the words")
    print(f"\nPCA keeps {explained[0]:.0%} + {explained[1]:.0%} = {explained[0] + explained[1]:.0%} of the spread in two numbers per word")
    print("centre of each body system on the 2-D map (x, y):")
    summary = {}
    for group in sorted(set(labels)):
        points = np.array([c for c, g in zip(coords, labels) if g == group])
        centre = points.mean(axis=0)
        summary[group] = centre.tolist()
        print(f"  {group:7s} ({len(points)} words): ({centre[0]:6.2f}, {centre[1]:6.2f})")

    save_results("d04_map", {"words": content, "group": labels, "role": [groups[w]["role"] for w in content],
                             "x": coords[:, 0].tolist(), "y": coords[:, 1].tolist(), "explained": explained[:3].tolist(),
                             "purity": purity, "purity_2d": flat, "history": run["history"], "train_loss": run["train_loss"], "test_loss": run["test_loss"],
                             "centres": summary}, digits=3)
    try:                                                         # a picture, when matplotlib is installed
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        for group in sorted(set(labels)):
            pts = np.array([c for c, g in zip(coords, labels) if g == group])
            ax.scatter(pts[:, 0], pts[:, 1], label=group)
        for (x, y), word in zip(coords, content):
            ax.annotate(word, (x, y), fontsize=6)
        ax.legend()
        ax.set_title("our word map: 8 numbers per word squashed to 2 with PCA")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(RESULTS_DIR / "d04_map.png", dpi=130)
        print("\nsaved data/results/d04_map.png")
    except ImportError:
        print("\n(matplotlib is not installed, so no picture was drawn; the numbers are in data/results/d04_map.json)")


if __name__ == "__main__":
    run_and_save("d04_train_map", main)
