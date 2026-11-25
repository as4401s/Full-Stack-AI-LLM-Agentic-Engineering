"""Demo 5 - one sentence through six tokenizers: ours, GPT-2, two OpenAI encodings, BERT and T5.

What it does: cuts the same sentence with our own BPE (demo 4), GPT-2's BPE, OpenAI's cl100k_base and o200k_base
(through tiktoken), BERT's WordPiece and T5's Unigram, and prints the pieces, the ids and the counts.
What it needs: tiktoken and tokenizers (the first run downloads a few small tokenizer files).
What to expect: every tokenizer gives a different cut and a different count for the same sentence.
Run it: uv run python -m demos.d05_real_tokenizers
"""
import importlib.metadata as md

from tokenizer_lab.realtok import all_tokenizers, show
from tokenizer_lab.report import run_and_save, save_results

SENTENCE = "The pharmacist filled the prescription for rivaroxaban 500 mg on 2026-03-14."


def main():
    print("libraries:", ", ".join(f"{n} {md.version(n)}" for n in ("tiktoken", "tokenizers", "sentencepiece")))
    print(f"\nsentence: {SENTENCE}")
    print(f"({len(SENTENCE.split())} words, {len(SENTENCE)} characters)\n")
    results = {}
    for tok in all_tokenizers():
        ids, pieces = tok.ids(SENTENCE), tok.pieces(SENTENCE)
        print(f"{tok.name}  (vocabulary {tok.vocab_size:,})  ->  {len(ids)} tokens")
        print("  pieces:", " | ".join(show(p) for p in pieces))
        print("  ids   :", ids[:12], "..." if len(ids) > 12 else "")
        results[tok.name] = {"vocab": tok.vocab_size, "tokens": len(ids), "pieces": pieces, "ids": ids}
    import tiktoken
    print("\nwhich tiktoken encoding belongs to which model name (from the library's own table):")
    wanted = ("gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3")
    table = {**tiktoken.model.MODEL_TO_ENCODING}
    for key in wanted:
        if key in table:
            print(f"  {key:15s} -> {table[key]}")
    for prefix, enc in tiktoken.model.MODEL_PREFIX_TO_ENCODING.items():
        if prefix.rstrip("-") in wanted:
            print(f"  {prefix + '*':15s} -> {enc}")
    save_results("d05_real_tokenizers", results)


if __name__ == "__main__":
    run_and_save("d05_real_tokenizers", main)
