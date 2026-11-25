"""Demo 10 - SentencePiece: train a Unigram tokenizer and a BPE tokenizer on the same text and compare.

What it does: trains two SentencePiece models (Unigram and BPE) with a vocabulary of 300 on the invented clinical text,
then cuts three sentences with each. SentencePiece reads raw text and writes a space as the mark ▁.
What it needs: sentencepiece (see pyproject.toml). Nothing is downloaded: the models are trained here.
What to expect: the mark ▁ stands for a space. BPE reaches the 300 pieces we ask for; Unigram settles at 260 pieces on this small text
(the soft limit lets it stop early), so compare the two methods with care. Unknown characters become <unk>.
Run it: uv run python -m demos.d10_sentencepiece
"""
import tempfile
from pathlib import Path

import sentencepiece as spm

from tokenizer_lab.corpus import DATA
from tokenizer_lab.report import run_and_save, save_results

SENTENCES = ["The pharmacist filled the prescription for rivaroxaban 500 mg.", "The nurse checks the chart for Mr. Berg in room 12.", "Die Kardiologiestation hat heute zwei freie Betten."]


def main():
    out = {}
    with tempfile.TemporaryDirectory() as folder:
        for kind in ("unigram", "bpe"):
            prefix = str(Path(folder) / kind)
            spm.SentencePieceTrainer.train(input=str(DATA / "clinical_train.txt"), model_prefix=prefix, vocab_size=300, model_type=kind,
                                           character_coverage=1.0, hard_vocab_limit=False, minloglevel=2, seed_sentencepiece_size=100000)
            sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
            print(f"== SentencePiece {kind}, vocabulary {sp.get_piece_size()}")
            out[kind] = []
            for text in SENTENCES:
                pieces = sp.encode(text, out_type=str)
                print(f"  {len(pieces):2d} tokens: {' '.join(pieces)}")
                out[kind].append({"text": text, "pieces": pieces})
            test = (DATA / "clinical_test.txt").read_text(encoding="utf-8")
            count = sum(len(sp.encode(line)) for line in test.splitlines() if line.strip())
            print(f"  whole test text: {count} tokens ({count / len(test.split()):.2f} per word)\n")
            out[kind].append({"test_tokens": count})
    save_results("d10_sentencepiece", out)


if __name__ == "__main__":
    run_and_save("d10_sentencepiece", main)
