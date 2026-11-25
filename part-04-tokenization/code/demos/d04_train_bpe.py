"""Demo 4 - train a byte-level BPE tokenizer on the invented clinical text, then encode and decode.

What it does: trains with a vocabulary of 512 pieces (256 bytes + 256 merges), prints the first learned merges,
encodes a test sentence and a drug name the text never mentions, and checks that decode(encode(text)) gives the text back.
What it needs: only the Python standard library.
What to expect: the first merge is ' f'; 'rivaroxaban' (never seen) splits into 10 pieces and 'physiology' (never seen) reuses the learned ending 'iology'; the round trip is exact.
Run it: uv run python -m demos.d04_train_bpe
"""
from tokenizer_lab.bpe import ByteBPE, pretokenize
from tokenizer_lab.corpus import read
from tokenizer_lab.report import RESULTS_DIR, run_and_save, save_results


def show(piece):
    """Show a piece with visible spaces."""
    return "'" + piece.decode("utf-8", errors="backslashreplace").replace(" ", "_") + "'"


def main():
    train, test = read("clinical_train"), read("clinical_test")
    print("chunks of one line (a leading space stays with its word):")
    sample = "Room 26: levothyroxine 5 mg twice daily from 2026-09-18."
    print(" ", pretokenize(sample))
    tok = ByteBPE.train(train, 512)
    print(f"\ntrained: {len(tok.merges)} merges, vocabulary of {len(tok.vocab)} pieces (ids 0-255 are bytes; _ marks a space)")
    print("the first 15 merges (id: left + right = new piece):")
    for i, (a, b) in enumerate(tok.merges[:15]):
        print(f"  {256 + i}: {show(a)} + {show(b)} = {show(a + b)}")
    print("\nsome later merges:", " ".join(show(a + b) for a, b in tok.merges[100:112]))
    print("\na word the training text never mentions, cut with the pieces it learned:")
    for word in [" physiology"]:
        print(f"  {word!r}: {' | '.join(tok.pieces(word))}")
    for text in ["The nurse checks the chart for Mr. Berg in room 12.", "The pharmacist filled the prescription for rivaroxaban 500 mg."]:
        ids = tok.encode(text)
        print(f"\n{text}")
        print(f"  pieces: {' | '.join(tok.pieces(text))}")
        print(f"  ids   : {ids}")
        print(f"  {len(text.split())} words -> {len(ids)} tokens;  decode gives the text back: {tok.decode(ids) == text}")
    test_ids = tok.encode(test)
    print(f"\nwhole test text: {len(test.split())} words -> {len(test_ids)} tokens ({len(test_ids) / len(test.split()):.2f} per word), round trip exact: {tok.decode(test_ids) == test}")
    for text in ["Größe 😀 €", "Die Untersuchungen zeigten keinen Herzinfarkt."]:
        ids = tok.encode(text)
        print(f"  never-seen text {text!r}: {len(ids)} tokens, round trip exact: {tok.decode(ids) == text}")
    save_results("d04_train_bpe", {"vocab": len(tok.vocab), "first_merges": [(a + b).decode("utf-8", errors="backslashreplace") for a, b in tok.merges[:15]],
                                   "test_tokens": len(test_ids), "pieces_rivaroxaban": tok.pieces(" rivaroxaban")})
    tok.save(RESULTS_DIR / "clinical_bpe_512.json")


if __name__ == "__main__":
    run_and_save("d04_train_bpe", main)
