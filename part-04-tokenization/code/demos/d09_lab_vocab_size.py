"""Demo 9 - the lab: how does the vocabulary size change what a tokenizer does?

What it does: trains our byte-level BPE at five vocabulary sizes on the invented clinical text and prints, for each size,
the tokens per word on the training text and on the test text, the bytes per token, and how many pieces a drug name
the training text never mentioned needs.
What it needs: only the Python standard library.
What to expect: more merges mean fewer tokens per word, with a smaller gain each time; an unseen drug name stays at about ten
pieces however large the vocabulary grows, because our small text never taught the tokenizer the parts of long drug names.
Run it: uv run python -m demos.d09_lab_vocab_size
"""
from tokenizer_lab.bpe import ByteBPE
from tokenizer_lab.corpus import read
from tokenizer_lab.report import run_and_save, save_results

SIZES = (260, 300, 400, 512, 695)


def main():
    train, test = read("clinical_train"), read("clinical_test")
    tw, sw = len(train.split()), len(test.split())
    print(f"{'vocabulary':>10s}{'merges':>8s}{'train tokens/word':>19s}{'test tokens/word':>18s}{'bytes per token':>17s}{'rivaroxaban':>13s}{'dapagliflozin':>15s}")
    rows = []
    for size in SIZES:
        tok = ByteBPE.train(train, size)
        a, b = len(tok.encode(train)), len(tok.encode(test))
        r1, r2 = len(tok.pieces(" rivaroxaban")), len(tok.pieces(" dapagliflozin"))
        rows.append({"vocab": len(tok.vocab), "train_per_word": a / tw, "test_per_word": b / sw, "bytes_per_token": len(test.encode()) / b, "rivaroxaban": r1, "dapagliflozin": r2})
        print(f"{len(tok.vocab):10d}{len(tok.merges):8d}{a / tw:19.2f}{b / sw:18.2f}{len(test.encode()) / b:17.2f}{r1:13d}{r2:15d}")
    print("\nsmall extra gains: going from 512 to 695 pieces saves", f"{rows[3]['test_per_word'] - rows[4]['test_per_word']:.2f}", "tokens per word, while going from 260 to 300 saved", f"{rows[0]['test_per_word'] - rows[1]['test_per_word']:.2f}")
    save_results("d09_lab_vocab_size", rows)


if __name__ == "__main__":
    run_and_save("d09_lab_vocab_size", main)
