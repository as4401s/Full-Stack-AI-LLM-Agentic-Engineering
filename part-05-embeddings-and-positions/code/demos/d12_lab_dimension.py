"""Demo 12 - lab: how many numbers per word does the map need?

What it does: trains the skip-gram model with 2, 4, 8, 16 and 32 numbers per word on the same sentences, and reports the loss on unseen
sentences, the number of table entries, how often a word's nearest neighbour comes from its own body system, and how often it is the same
kind of word (organ, specialist, condition, medicine, test or clinic).
What it needs: torch and NumPy; about a minute.
What to expect: two numbers per word are too few to separate six body systems (they sort words by kind instead); around eight are enough; more numbers add little here.
Run it: uv run python -m demos.d12_lab_dimension
"""
import math

from embedding_lab import corpus, pca
from embedding_lab.mapmaker import trained_map
from embedding_lab.report import run_and_save, save_results


def main():
    groups = corpus.word_groups()
    results = {}
    print(f"{'numbers per word':>16} {'table entries':>14} {'train loss':>11} {'test loss':>10} {'same system':>12} {'same system, 2-D':>17} {'same kind of word':>18}")
    for dim in (2, 4, 8, 16, 32):
        run = trained_map(dim=dim)
        content = [w for w in run["words"] if w in groups]
        matrix = run["table"][[run["index"][w] for w in content]]
        labels = [groups[w]["group"] for w in content]
        purity = pca.neighbour_purity(matrix, labels)
        flat = pca.neighbour_purity(pca.pca_2d(matrix)[0], labels, metric="euclid") if dim > 2 else purity
        kinds = pca.neighbour_purity(matrix, [groups[w]["role"] for w in content])      # organ, specialist, condition, medicine, test, clinic
        print(f"{dim:>16} {run['table'].size:>14,} {run['train_loss']:>11.3f} {run['test_loss']:>10.3f} {purity:>12.0%} {flat:>17.0%} {kinds:>18.0%}")
        results[dim] = {"table_entries": int(run["table"].size), "train_loss": run["train_loss"], "test_loss": run["test_loss"], "purity": purity, "purity_2d": flat, "same_kind": kinds}
    print(f"\nrandom guessing costs ln({len(run['words'])}) = {math.log(len(run['words'])):.3f}; choosing a neighbour at random would match the body system about 1 time in 6 ({1 / 6:.0%})")
    save_results("d12_lab_dimension", results)


if __name__ == "__main__":
    run_and_save("d12_lab_dimension", main)
