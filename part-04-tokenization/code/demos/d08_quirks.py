"""Demo 8 - tokenizer quirks that sit behind model mistakes: spaces, capitals, digits, rare words and letters.

What it does: prints the pieces that two OpenAI encodings give to small texts that differ in one detail
(a leading space, a capital letter, a date, a long number, a rare drug name) and shows why counting letters is hard for a model.
What it needs: tiktoken (a small download on the first run).
What to expect: '2026' and ' 2026' are different tokens; one number is cut into groups of up to three digits;
a rare drug name needs several tokens; a word is a handful of ids, so its letters are hidden inside the pieces.
Run it: uv run python -m demos.d08_quirks
"""
from tokenizer_lab.realtok import show, tiktoken_encoding
from tokenizer_lab.report import run_and_save, save_results


def row(label, tok, text):
    """Print one text with its pieces and ids and return the data."""
    pieces, ids = tok.pieces(text), tok.ids(text)
    print(f"  {label:14s}{text!r:22s} {len(ids):2d}  {' | '.join(show(p) for p in pieces)}   {ids}")
    return {"text": text, "pieces": pieces, "ids": ids}


def main():
    out = {}
    for name in ("cl100k_base", "o200k_base"):
        tok = tiktoken_encoding(name)
        print(f"== {name}")
        print(" spaces and capitals: the same word, different tokens")
        out[name] = {"spaces": [row("", tok, t) for t in ("nurse", " nurse", "Nurse", " Nurse", "NURSE", " nurse  ")]}
        print(" numbers")
        out[name]["numbers"] = [row("", tok, t) for t in ("2026", " 2026", "2026-03-14", "1234567", "12,345.67", "3.14159")]
        print(" common words and rare drug names (with a leading space)")
        out[name]["rare"] = [row("", tok, t) for t in (" the", " nurse", " prescription", " paracetamol", " rivaroxaban", " dapagliflozin", " hydroxychloroquine")]
        print(" long words")
        out[name]["long"] = [row("", tok, t) for t in (" cardiology", " Kardiologiestation", " electrocardiography")]
        print()
    print("== counting letters")
    tok = tiktoken_encoding("o200k_base")
    letters_out = []
    for word, letter in (("paracetamol", "a"), ("strawberry", "r")):
        pieces, ids = tok.pieces(" " + word), tok.ids(" " + word)
        print(f"  the model receives ' {word}' as {len(ids)} ids {ids}  (pieces: {' | '.join(show(p) for p in pieces)})")
        print(f"  a program counts the letter '{letter}' exactly: {word.count(letter)}   (code: '{word}'.count('{letter}'))")
        letters_out.append({"word": word, "letter": letter, "count": word.count(letter), "pieces": pieces, "ids": ids})
    out["letters"] = letters_out
    save_results("d08_quirks", out)


if __name__ == "__main__":
    run_and_save("d08_quirks", main)
