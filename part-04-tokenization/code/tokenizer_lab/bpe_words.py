"""bpe_words.py - byte-pair encoding by hand, on a tiny list of words.

What it does
    * starts from words written as single letters (every letter is one piece),
    * counts which two neighbouring pieces occur most often (weighted by how often each word occurs),
    * glues the most frequent pair into one new piece, and repeats.
    The list of glue steps (the "merges") is the whole tokenizer: apply the same steps, in the same order,
    to any new word and you get its pieces.

What it needs
    Only the Python standard library.

What to expect
    On the four-word corpus in demos/d02_bpe_by_hand.py five merges give: ed, ar, ch, war, ward.

How ties are settled
    When two pairs are equally frequent, the one met first (scanning the words in the order given, left to right) wins.
    That rule is arbitrary but fixed, so every run gives the same result.
"""
from collections import Counter


def split_into_letters(word_counts):
    """Turn {'ward': 8} into {('w', 'a', 'r', 'd'): 8}: every word becomes a tuple of one-letter pieces."""
    return {tuple(word): count for word, count in word_counts.items()}


def count_pairs(words):
    """Count every pair of neighbouring pieces, adding the word's count each time the pair occurs in that word."""
    pairs = Counter()
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def most_frequent_pair(words):
    """Return the neighbouring pair with the highest count (the first one met wins a tie)."""
    pairs = count_pairs(words)
    return max(pairs, key=pairs.get)


def merge(words, pair):
    """Glue every occurrence of `pair` into one piece, in every word, and return the new word table."""
    merged = {}
    for symbols, freq in words.items():
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1])  # the new, longer piece
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged[tuple(out)] = freq
    return merged


def train(word_counts, num_merges):
    """Learn `num_merges` merges. Returns (merges, steps): the glue list and, for each step, what was counted."""
    words = split_into_letters(word_counts)
    merges, steps = [], []
    for _ in range(num_merges):
        pairs = count_pairs(words)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        steps.append({"pair": best, "count": pairs[best], "table": dict(pairs)})
        words = merge(words, best)
        steps[-1]["words_after"] = {" ".join(k): v for k, v in words.items()}
    return merges, steps


def encode_word(word, merges):
    """Cut a word into pieces by replaying the learned merges in the order they were learned."""
    symbols = tuple(word)
    for pair in merges:
        symbols = merge({symbols: 1}, pair).popitem()[0]
    return list(symbols)


def decode_pieces(pieces):
    """Join the pieces back together: decoding is plain concatenation, so nothing is lost."""
    return "".join(pieces)


def vocabulary(word_counts, merges):
    """The pieces the tokenizer knows: every starting letter, then one new piece per merge."""
    letters = sorted({ch for word in word_counts for ch in word})
    return letters + [a + b for a, b in merges]
