"""Demo 5 - analogies with vectors, and why they need care.

What it does: (1) on our own map, tests "specialist - organ + other organ" for all 30 pairs of body systems; (2) on 50-number GloVe vectors,
tries king - man + woman and a few more, skipping the input words, and shows one result that reveals a bias in the training text.
What it needs: torch, gensim and huggingface_hub (GloVe download is about 100 MB).
What to expect: our own map answers 30 of 30 because its sentences are perfectly regular; GloVe gets the famous one right and some others wrong.
Run it: uv run python -m demos.d05_analogies
"""
from embedding_lab import corpus, realmodels, vectors
from embedding_lab.mapmaker import trained_map
from embedding_lab.report import run_and_save, save_results


def main():
    run = trained_map()
    words, index, table = run["words"], run["index"], run["table"]
    groups = corpus.word_groups()
    systems = sorted({g["group"] for g in groups.values()})
    role = {(g["group"], g["role"]): w for w, g in groups.items() if g["role"] in ("organ", "staff", "clinic")}

    print("our own map: specialist(A) - organ(A) + organ(B) should give specialist(B)")
    shown, right, total = 0, 0, 0
    for a in systems:
        for b in systems:
            if a == b:
                continue
            start, minus, plus, want = role[(a, "staff")], role[(a, "organ")], role[(b, "organ")], role[(b, "staff")]
            target = table[index[start]] - table[index[minus]] + table[index[plus]]
            unit = table / (table ** 2).sum(axis=1, keepdims=True) ** 0.5
            scores = unit @ (target / (target ** 2).sum() ** 0.5)
            order = [i for i in scores.argsort()[::-1] if words[i] not in (start, minus, plus)]
            answer = words[order[0]]
            total += 1
            right += answer == want
            if shown < 3:
                print(f"  {start} - {minus} + {plus} -> {answer} ({scores[order[0]]:.2f}); expected {want}")
                shown += 1
    print(f"  all pairs: {right} of {total} correct (the sentences are regular and made by templates)\n")
    results = {"ours": {"correct": right, "total": total}}

    glove = realmodels.try_load("GloVe", realmodels.load_glove)
    if glove is not None:
        print(f"GloVe: {len(glove.key_to_index):,} words x {glove.vector_size} numbers, learned from Wikipedia and a news archive")
        tests = [(["king", "woman"], ["man"], "queen"), (["paris", "italy"], ["france"], "rome"), (["walking", "swam"], ["walked"], "swimming"),
                 (["doctor", "woman"], ["man"], "doctor")]
        for positive, negative, expected in tests:
            top = glove.most_similar(positive=positive, negative=negative, topn=3)       # gensim skips the input words by itself
            print(f"  {positive[0]} - {negative[0]} + {positive[1]} -> " + ", ".join(f"{w} {s:.2f}" for w, s in top) + f"   (a clean answer would be {expected})")
            results[f"{positive[0]}-{negative[0]}+{positive[1]}"] = [(w, float(s)) for w, s in top]
        print("\nwhat happens when the input words are not skipped (king - man + woman):")
        raw = glove["king"] - glove["man"] + glove["woman"]
        unit = glove.vectors / (glove.vectors ** 2).sum(axis=1, keepdims=True) ** 0.5
        scores = unit @ (raw / (raw ** 2).sum() ** 0.5)
        for i in scores.argsort()[::-1][:3]:
            print(f"  {glove.index_to_key[i]:8s} {scores[i]:.2f}")
    save_results("d05_analogies", results)


if __name__ == "__main__":
    run_and_save("d05_analogies", main)
