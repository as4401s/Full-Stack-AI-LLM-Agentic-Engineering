"""Demo 1 - the hook: a search for "kidney failure" that finds "renal insufficiency".

What it does: ranks eight invented clinical sentences for the query "kidney failure" two ways: (1) keyword search, which needs a shared
word, and (2) embedding search, which compares sentence vectors from the small public model all-MiniLM-L6-v2 by cosine.
What it needs: torch and transformers (the first run downloads about 90 MB).
What to expect: the keyword search misses the sentences that say the same thing in other words; the embedding search ranks them near the top.
Run it: uv run python -m demos.d01_search_hook
"""
from embedding_lab import realmodels
from embedding_lab.report import run_and_save, save_results

QUERY = "kidney failure"
DOCUMENTS = [
    "Renal insufficiency was noted at the follow-up visit.",
    "The patient reports pain in the left knee after a fall.",
    "Creatinine rose again, which suggests the kidneys are failing.",
    "Chronic kidney failure requires regular dialysis sessions.",
    "The flight to Berlin leaves at nine in the morning.",
    "Heart failure symptoms include swollen ankles and tiredness.",
    "A broken wrist needs a cast for six weeks.",
    "Dialysis replaces the work of the kidneys when they stop working.",
]


def keyword_hits(query, documents):
    """Documents that share at least one whole word with the query (lower case), with the shared words."""
    wanted = set(query.lower().split())
    hits = []
    for number, text in enumerate(documents):
        shared = sorted(wanted & {w.strip(".,").lower() for w in text.split()})
        if shared:
            hits.append((number, shared))
    return hits


def main():
    import numpy as np
    tokenizer, model = realmodels.load_minilm()
    print(f'query: "{QUERY}"\n')
    print("keyword search (a sentence must share a whole word with the query):")
    hits = keyword_hits(QUERY, DOCUMENTS)
    for number, shared in hits:
        print(f"  [{number}] shares {shared}: {DOCUMENTS[number]}")
    print(f"  -> {len(hits)} of {len(DOCUMENTS)} sentences found; "
          f"missed: {[n for n in range(len(DOCUMENTS)) if n not in {h[0] for h in hits}][:3]} and more\n")

    vectors = realmodels.sentence_vectors([QUERY] + DOCUMENTS, tokenizer, model)
    print(f"each sentence became {vectors.shape[1]} numbers, each vector has length {float(np.linalg.norm(vectors[0])):.3f}\n")
    scores = vectors[1:] @ vectors[0]                      # unit vectors: the dot product is the cosine
    order = np.argsort(-scores)
    print("embedding search (cosine between the query vector and each sentence vector):")
    for rank, number in enumerate(order, start=1):
        print(f"  {rank}. cosine {scores[number]:.2f}  [{number}] {DOCUMENTS[number]}")

    pair_a = float(vectors[0] @ realmodels.sentence_vectors(["renal insufficiency"], tokenizer, model)[0])
    pair_b = float(vectors[0] @ realmodels.sentence_vectors(["knee surgery"], tokenizer, model)[0])
    print(f'\ncosine("kidney failure", "renal insufficiency") = {pair_a:.2f}')
    print(f'cosine("kidney failure", "knee surgery")        = {pair_b:.2f}')
    save_results("d01_search_hook", {"keyword_hits": [h[0] for h in hits], "ranking": [int(n) for n in order],
                                     "scores": [float(scores[n]) for n in order], "pair_related": pair_a, "pair_unrelated": pair_b})


if __name__ == "__main__":
    run_and_save("d01_search_hook", main)
