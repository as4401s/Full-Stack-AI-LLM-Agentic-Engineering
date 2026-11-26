"""Demo 7 - token vectors versus sentence vectors (they return in lesson 17 for retrieval).

What it does: (1) shows that the table row of "bank" is the same in every sentence while the model's output vector for "bank" changes with its
sentence; (2) compares a sentence vector made by averaging table rows with the real sentence vector of MiniLM.
What it needs: torch and transformers (MiniLM, about 90 MB on the first run).
What to expect: the table row never changes; the output vectors for "bank" in the two sentences differ; the real sentence vectors separate
related from unrelated sentences better than a plain average of table rows.
Run it: uv run python -m demos.d07_sentence_vs_token
"""
import numpy as np

from embedding_lab import realmodels, vectors
from embedding_lab.report import run_and_save, save_results


def main():
    tokenizer, model = realmodels.load_minilm()
    table = realmodels.token_table(model)
    river, money = "She sat on the bank of the river.", "He deposited money at the bank."
    results = {}
    outputs = {}
    for label, text in (("river", river), ("money", money)):
        tokens, hidden = realmodels.token_vectors(text, tokenizer, model)
        position = tokens.index("bank")
        outputs[label] = hidden[position]
        print(f"{text!r} -> tokens {tokens}")
        print(f"  'bank' sits at position {position}; its output vector starts {[round(float(v), 3) for v in hidden[position][:4]]}")
    bank_id = tokenizer("bank", add_special_tokens=False)["input_ids"][0]
    print(f"\ntable row of 'bank' (id {bank_id}) starts {[round(float(v), 3) for v in table[bank_id][:4]]} in both sentences: the lookup ignores the neighbours")
    cos_ctx = vectors.cosine(outputs["river"], outputs["money"])
    print(f"cosine between the two output vectors for 'bank': {cos_ctx:.2f} (the table row would give 1.00)")
    results["bank_context_cosine"] = cos_ctx

    print("\nsentence vectors: the model's own recipe versus a plain average of table rows")
    texts = ["kidney failure", "renal insufficiency", "knee surgery"]
    real = realmodels.sentence_vectors(texts, tokenizer, model)
    averaged = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        mean = table[ids].mean(axis=0)
        averaged.append(mean / np.linalg.norm(mean))
    for (i, j) in ((0, 1), (0, 2)):
        r, a = float(real[i] @ real[j]), float(averaged[i] @ averaged[j])
        print(f"  cosine({texts[i]!r}, {texts[j]!r}): model sentence vectors {r:.2f}, average of table rows {a:.2f}")
        results[f"{texts[i]}|{texts[j]}"] = {"model": r, "table_average": a}
    save_results("d07_sentence_vs_token", results)


if __name__ == "__main__":
    run_and_save("d07_sentence_vs_token", main)
