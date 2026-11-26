"""Demo 3 - the real tables: token ids, rows, and sizes in MiniLM, GPT-2 and Qwen.

What it does: cuts words into tokens with the MiniLM tokenizer, prints the first numbers of the table row of "kidney", compares table rows,
and reads the sizes of the token and position tables of three real models from their small config.json files.
What it needs: torch, transformers, huggingface_hub (downloads MiniLM about 90 MB, and a few kilobytes of config files).
What to expect: token ids as the row numbers; 30,522 x 384 numbers for MiniLM; the position table has 512 rows.
Run it: uv run python -m demos.d03_real_tables
"""
import numpy as np

from embedding_lab import realmodels, vectors
from embedding_lab.report import run_and_save, save_results


def main():
    tokenizer, model = realmodels.load_minilm()
    table, positions = realmodels.token_table(model), realmodels.position_table(model)
    print(f"MiniLM token table: {table.shape[0]:,} rows x {table.shape[1]} numbers = {table.size:,} numbers")
    print(f"MiniLM position table: {positions.shape[0]} rows x {positions.shape[1]} numbers = {positions.size:,} numbers\n")

    results = {}
    for word in ("kidney", "renal", "insufficiency", "failure", "knee", "the"):
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        print(f"{word!r}: ids {ids} -> pieces {tokenizer.convert_ids_to_tokens(ids)}")
        results[word] = ids
    encoded = tokenizer("kidney failure")["input_ids"]
    print(f"\n'kidney failure' with the start and end markers: ids {encoded} -> {tokenizer.convert_ids_to_tokens(encoded)}")

    kid = tokenizer("kidney", add_special_tokens=False)["input_ids"][0]
    print(f"\nrow {kid} (the token 'kidney'), first 6 of {table.shape[1]} numbers: {[round(float(v), 3) for v in table[kid][:6]]}")

    print("\ncosine between table rows (no context involved yet, just the lookup rows):")
    pairs = (("kidney", "renal"), ("kidney", "knee"), ("kidney", "the"))
    for a, b in pairs:
        ia, ib = tokenizer(a, add_special_tokens=False)["input_ids"][0], tokenizer(b, add_special_tokens=False)["input_ids"][0]
        print(f"  cosine({a}, {b}) = {vectors.cosine(table[ia], table[ib]):.2f}")
        results[f"cos_{a}_{b}"] = vectors.cosine(table[ia], table[ib])

    print("\nsizes read from the models' own config.json files:")
    gpt2 = realmodels.try_load("gpt2 config", realmodels.hub_config, "openai-community/gpt2")
    qwen = realmodels.try_load("qwen config", realmodels.hub_config, "Qwen/Qwen2.5-0.5B-Instruct")
    if gpt2:
        v, d, n = gpt2["vocab_size"], gpt2["n_embd"], gpt2["n_positions"]
        print(f"  GPT-2: token table {v:,} x {d} = {v * d:,}; learned position table {n:,} x {d} = {n * d:,}")
        results["gpt2"] = {"vocab": v, "dim": d, "positions": n, "token_numbers": v * d, "position_numbers": n * d}
    if qwen:
        v, d = qwen["vocab_size"], qwen["hidden_size"]
        theta = qwen.get("rope_theta") or qwen.get("rope_parameters", {}).get("rope_theta")      # older and newer config layouts
        print(f"  Qwen2.5-0.5B-Instruct: token table {v:,} x {d} = {v * d:,} numbers; positions: RoPE (no position table), "
              f"rope_theta {theta:,.0f}, max_position_embeddings {qwen['max_position_embeddings']:,}, tied input/output table: {qwen.get('tie_word_embeddings')}")
        results["qwen"] = {"vocab": v, "dim": d, "token_numbers": v * d, "rope_theta": theta,
                           "max_positions": qwen["max_position_embeddings"], "tied": qwen.get("tie_word_embeddings")}
    results["minilm"] = {"vocab": table.shape[0], "dim": table.shape[1], "positions": positions.shape[0], "token_numbers": int(table.size)}
    save_results("d03_real_tables", results)


if __name__ == "__main__":
    run_and_save("d03_real_tables", main)
