"""Demo 2 - byte-pair encoding by hand on a four-word corpus: count pairs, merge the most frequent, repeat.

What it does: shows five merges on four words (charted, dosed, ward, checked), printing the pair counts each time,
then cuts a new word, 'warded', with the learned merges and joins the pieces back together.
What it needs: only the Python standard library.
What to expect: the merges ed, ar, ch, war, ward; the vocabulary grows from 11 letters to 16 pieces;
'warded' becomes ward + ed; 'waxed' has a letter (x) that the starting vocabulary lacks.
Run it: uv run python -m demos.d02_bpe_by_hand
"""
from tokenizer_lab import bpe_words as bw
from tokenizer_lab.report import run_and_save, save_results

CORPUS = {"charted": 7, "dosed": 6, "ward": 8, "checked": 5}


def main():
    print("the corpus (word: how often it occurs):", CORPUS)
    words = bw.split_into_letters(CORPUS)
    print("\nstart: every word is a row of single letters")
    for symbols, freq in words.items():
        print(f"  {' '.join(symbols):16s} x {freq}")
    merges, steps = bw.train(CORPUS, 5)
    for number, step in enumerate(steps, 1):
        top = sorted(step["table"].items(), key=lambda kv: -kv[1])[:5]
        print(f"\nmerge {number}: the most frequent pair is {step['pair'][0]} + {step['pair'][1]}  (count {step['count']})")
        print("  top pairs:", ", ".join(f"{a}+{b}={c}" for (a, b), c in top))
        for text, freq in step["words_after"].items():
            print(f"  {text:16s} x {freq}")
    vocab = bw.vocabulary(CORPUS, merges)
    print(f"\nthe vocabulary: {len(vocab)} pieces = 11 letters + 5 merges")
    print(" ", " ".join(vocab))
    print("\nthe merge list, in order:", [a + "+" + b for a, b in merges])
    for new in ("warded", "waxed"):
        pieces = bw.encode_word(new, merges)
        unknown = [p for p in pieces if p not in vocab]
        print(f"\nencode '{new}' with the same five merges: {' '.join(pieces)}" + (f"   (not in the vocabulary: {unknown})" if unknown else ""))
        print(f"  decode: {bw.decode_pieces(pieces)}")
    save_results("d02_bpe_by_hand", {"merges": [a + b for a, b in merges], "counts": [s["count"] for s in steps], "vocabulary": vocab,
                                     "warded": bw.encode_word("warded", merges), "first_table": steps[0]["table"] and {a + "+" + b: c for (a, b), c in steps[0]["table"].items()}})


if __name__ == "__main__":
    run_and_save("d02_bpe_by_hand", main)
