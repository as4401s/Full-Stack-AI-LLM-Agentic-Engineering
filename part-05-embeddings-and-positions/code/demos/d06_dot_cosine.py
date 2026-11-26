"""Demo 6 - dot product and cosine, first on three numbers, then on our own word map.

What it does: repeats the by-hand example [1, 2, 0] and [2, 1, 0], shows that a vector scaled by 2 keeps its cosine while the straight-line distance
changes, and then measures cosines between words of our trained table.
What it needs: torch and NumPy (the map comes from demo 4's training run).
What to expect: cosine 0.8 for the hand example; close to 1 for words of the same body system; lower for words of different systems.
Run it: uv run python -m demos.d06_dot_cosine
"""
import numpy as np

from embedding_lab import vectors
from embedding_lab.mapmaker import trained_map
from embedding_lab.report import run_and_save, save_results


def main():
    a, b = np.array([1.0, 2.0, 0.0]), np.array([2.0, 1.0, 0.0])
    print(f"a = {a.tolist()}, b = {b.tolist()}")
    print(f"dot(a, b)    = 1*2 + 2*1 + 0*0 = {vectors.dot(a, b):.0f}")
    print(f"length(a) = length(b) = sqrt(5) = {vectors.length(a):.3f}")
    print(f"cosine(a, b) = 4 / 5 = {vectors.cosine(a, b):.2f}")
    print(f"cosine(a, a) = {vectors.cosine(a, a):.2f}  (a vector points exactly the way it points)")
    print(f"cosine(a, 2a) = {vectors.cosine(a, 2 * a):.2f}  but the distance between a and 2a is {vectors.euclidean(a, 2 * a):.3f} while a to b is {vectors.euclidean(a, b):.3f}")
    print(f"cosine([1,0,0], [0,1,0]) = {vectors.cosine([1, 0, 0], [0, 1, 0]):.2f}   cosine(a, -a) = {vectors.cosine(a, -a):.2f}\n")

    run = trained_map()
    words, index, table = run["words"], run["index"], run["table"]
    pairs = [("kidney", "nephrologist"), ("kidney", "furosemide"), ("kidney", "nephritis"), ("heart", "cardiologist"),
             ("kidney", "cardiologist"), ("kidney", "ramipril"), ("furosemide", "ramipril"), ("asthma", "stroke")]
    print("cosine between rows of our trained table:")
    results = {}
    for x, y in pairs:
        c = vectors.cosine(table[index[x]], table[index[y]])
        results[f"{x}|{y}"] = c
        print(f"  {x:12s} {y:13s} {c:5.2f}")
    print("\nthe same table, one word against every other word of its own body system and of all the others:")
    from embedding_lab import corpus
    groups = corpus.word_groups()
    unit = vectors.normalise_rows(table)
    same, other = [], []
    content = [w for w in words if w in groups]
    for i, x in enumerate(content):
        for y in content[i + 1:]:
            c = float(unit[index[x]] @ unit[index[y]])
            (same if groups[x]["group"] == groups[y]["group"] else other).append(c)
    print(f"  same body system : {len(same)} pairs, average cosine {np.mean(same):.2f}")
    print(f"  different systems: {len(other)} pairs, average cosine {np.mean(other):.2f}")
    results["average_same"], results["average_other"] = float(np.mean(same)), float(np.mean(other))
    save_results("d06_dot_cosine", results)


if __name__ == "__main__":
    run_and_save("d06_dot_cosine", main)
