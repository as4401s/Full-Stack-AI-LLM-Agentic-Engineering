"""Demo 1 - three ways to cut text into pieces: letters, words, or pieces in between.

What it does: cuts the invented CareDesk text in three ways and counts what each way costs.
  * letters: a tiny vocabulary, but very long token lists;
  * whole words: one token per word, but a huge vocabulary and no answer for a word it has never seen;
  * pieces (byte-pair encoding, built in demo 4): in between.
What it needs: only the Python standard library.
What to expect: letters 5.77 tokens per word; the best 100 words leave 254 of 2308 test words unknown;
  pieces get 1.80 tokens per word with a vocabulary of 512.
Run it: uv run python -m demos.d01_three_ways
"""
from collections import Counter

from tokenizer_lab.bpe import ByteBPE
from tokenizer_lab.corpus import read
from tokenizer_lab.report import run_and_save, save_results


def main():
    train, test = read("clinical_train"), read("clinical_test")
    test_words = test.split()
    print(f"training text: {len(train.split())} words, {len(train)} letters; test text: {len(test_words)} words, {len(test)} letters")

    print("\none word, three ways:")
    word = "paracetamol"
    bpe = ByteBPE.train(train, 512)
    print(f"  letters : {' '.join(word)}   ({len(word)} tokens)")
    print(f"  one word: {word}   (1 token if the word is in the list, else 'unknown')")
    print(f"  pieces  : {' '.join(bpe.pieces(' ' + word))}   ({len(bpe.pieces(' ' + word))} tokens)")

    print("\nthe same word list on a word we never trained on:")
    new = "rivaroxaban"
    print(f"  letters : {' '.join(new)}   ({len(new)} tokens)")
    print(f"  one word: {new}   -> unknown (not in the list)")
    print(f"  pieces  : {' '.join(bpe.pieces(' ' + new))}   ({len(bpe.pieces(' ' + new))} tokens)")

    print("\nthe trade-off on the test text (2308 words):")
    print(f"  {'way':26s}{'vocabulary':>11s}{'tokens per word':>17s}{'unknown words':>15s}")
    rows = []
    letters_vocab = len(set(train))
    rows.append(("letters", letters_vocab, len(test) / len(test_words), 0))
    counts = Counter(train.lower().split())
    for n in (100, len(counts)):
        known = {w for w, _ in counts.most_common(n)}
        unknown = sum(1 for w in test.lower().split() if w not in known)
        rows.append((f"whole words (top {n})", n, 1.0, unknown))
    for size in (300, 400, 512):
        tok = ByteBPE.train(train, size)
        rows.append((f"pieces (vocabulary {size})", size, len(tok.encode(test)) / len(test_words), 0))
    for name, vocab, per_word, unknown in rows:
        print(f"  {name:26s}{vocab:11d}{per_word:17.2f}{unknown:15d}")

    print("\ntokens per word as the vocabulary grows (pieces, learned from the training text):")
    curve = []
    for size in (256, 300, 350, 400, 450, 512, 600, 695):
        tok = ByteBPE.train(train, size)
        per_word = len(tok.encode(test)) / len(test_words)
        curve.append((len(tok.vocab), per_word))
        print(f"  vocabulary {len(tok.vocab):4d}: {per_word:5.2f} tokens per word")
    save_results("d01_three_ways", {"rows": rows, "curve": curve, "test_words": len(test_words),
                                    "unknown_by_top_n": {n: sum(1 for w in test.lower().split() if w not in {x for x, _ in counts.most_common(n)}) for n in (10, 20, 50, 100, 200, len(counts))}})


if __name__ == "__main__":
    run_and_save("d01_three_ways", main)
