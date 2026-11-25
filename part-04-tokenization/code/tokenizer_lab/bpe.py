"""bpe.py - a small byte-level BPE tokenizer: train it on text, then encode and decode.

What it does
    * cuts text into chunks first (a word with its leading space, a short run of digits, a run of punctuation),
    * turns every chunk into UTF-8 bytes, so the starting vocabulary is just the 256 possible bytes (ids 0 to 255),
    * learns merges: the most frequent neighbouring pair of pieces becomes one new piece with the next free id,
    * encodes new text by replaying the merges, and decodes ids back to exactly the text that went in.
    This is the same recipe that GPT-2 and tiktoken use, in about a hundred lines.

What it needs
    Only the Python standard library.

What to expect
    Training on the invented CareDesk text (demos/d04_train_bpe.py) learns merges such as ' t' + 'he'.
    Because every byte is in the starting vocabulary, no text is ever "unknown", and decode(encode(text)) == text.
"""
import json
import re
from collections import Counter

# A chunk is: optional space + letters | optional space + up to 3 digits | optional space + punctuation | spaces.
# Merges never cross a chunk border, so a word never glues itself to the word next to it.
CHUNK_PATTERN = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?[^\w\s]+|\s+(?!\S)|\s+")


def pretokenize(text):
    """Cut text into chunks, for example 'the 2026 chart' -> ['the', ' ', ...] style pieces that keep their leading space."""
    return CHUNK_PATTERN.findall(text)


def most_frequent_pair(words):
    """Return the most frequent neighbouring pair (weighted by chunk counts); the first one met wins a tie."""
    pairs = Counter()
    for symbols, freq in words.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return max(pairs, key=pairs.get), pairs


def merge(words, pair):
    """Glue every occurrence of `pair` into one piece (the concatenated bytes) in every chunk."""
    merged = {}
    for symbols, freq in words.items():
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged[tuple(out)] = freq
    return merged


class ByteBPE:
    """A byte-level BPE tokenizer.

    Attributes
        merges   list of (left_bytes, right_bytes) pairs, in the order they were learned
        vocab    list of byte strings; the id of a piece is its position (0-255 are single bytes)
        special  dict {text: id} of special tokens, which are never cut into pieces
    """

    def __init__(self, merges=None, special=None):
        self.merges = list(merges or [])
        self.vocab = [bytes([i]) for i in range(256)] + [a + b for a, b in self.merges]
        self.special = dict(special or {})
        self._rank = {pair: i for i, pair in enumerate(self.merges)}  # earlier merge = higher priority
        self._id = {piece: i for i, piece in enumerate(self.vocab)}
        self._cache = {}

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, text, vocab_size):
        """Learn merges from `text` until the vocabulary holds `vocab_size` pieces (256 bytes + merges)."""
        counts = Counter(pretokenize(text))
        words = {tuple(bytes([b]) for b in chunk.encode("utf-8")): n for chunk, n in counts.items()}
        merges = []
        while 256 + len(merges) < vocab_size:
            if not any(len(s) > 1 for s in words):
                break  # every chunk is already a single piece: nothing left to merge
            pair, _ = most_frequent_pair(words)
            merges.append(pair)
            words = merge(words, pair)
        return cls(merges)

    # ------------------------------------------------------------------ encoding
    def _encode_chunk(self, chunk):
        """Encode one chunk: start from its bytes and repeatedly apply the earliest learned merge that fits."""
        if chunk in self._cache:
            return self._cache[chunk]
        symbols = [bytes([b]) for b in chunk.encode("utf-8")]
        while len(symbols) > 1:
            ranked = [(self._rank.get((a, b), None), i) for i, (a, b) in enumerate(zip(symbols, symbols[1:]))]
            ranked = [(r, i) for r, i in ranked if r is not None]
            if not ranked:
                break
            _, i = min(ranked)
            symbols[i:i + 2] = [symbols[i] + symbols[i + 1]]
        ids = [self._id[s] for s in symbols]
        self._cache[chunk] = ids
        return ids

    def encode(self, text, allow_special=True):
        """Turn text into a list of ids. Special tokens found in the text become their single reserved id."""
        parts = [text]
        if allow_special and self.special:
            pattern = "(" + "|".join(re.escape(s) for s in sorted(self.special, key=len, reverse=True)) + ")"
            parts = re.split(pattern, text)
        ids = []
        for part in parts:
            if part in self.special and allow_special:
                ids.append(self.special[part])
            else:
                for chunk in pretokenize(part):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    # ------------------------------------------------------------------ decoding
    def decode(self, ids):
        """Turn ids back into text: join the bytes of every piece, then read them as UTF-8."""
        reverse_special = {i: s.encode("utf-8") for s, i in self.special.items()}
        data = b"".join(reverse_special.get(i) or self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def pieces(self, text):
        """The pieces of a text as readable strings (bytes shown as text; a split-up character shows as escapes)."""
        names = {i: t for t, i in self.special.items()}
        return [names[i] if i in names else self.vocab[i].decode("utf-8", errors="backslashreplace") for i in self.encode(text)]

    # ------------------------------------------------------------------ saving
    def save(self, path):
        """Write the merges to a JSON file (each piece as hex so any byte is safe to store)."""
        data = {"merges": [[a.hex(), b.hex()] for a, b in self.merges], "special": self.special}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)

    @classmethod
    def load(cls, path):
        """Read a tokenizer saved by `save`."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls([(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges"]], data["special"])
