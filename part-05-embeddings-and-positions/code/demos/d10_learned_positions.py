"""Demo 10 - learned position tables: MiniLM and GPT-2.

What it does: looks inside the 512 x 384 position table that MiniLM learned during training: how similar are the rows of neighbouring positions
compared with far-apart positions? Then it reads the size of GPT-2's position table from its config.json.
What it needs: torch, transformers, huggingface_hub.
What to expect: nearby positions have more similar rows than far-apart ones; there is no row for position 512 or beyond.
Run it: uv run python -m demos.d10_learned_positions
"""
import numpy as np

from embedding_lab import realmodels, vectors
from embedding_lab.report import run_and_save, save_results


def main():
    tokenizer, model = realmodels.load_minilm()
    table = realmodels.position_table(model)
    print(f"learned position table: {table.shape[0]} rows x {table.shape[1]} numbers (one row per position 0..{table.shape[0] - 1})")
    print(f"the tokenizer cuts every text at {tokenizer.model_max_length} tokens, because no row exists for position {table.shape[0]}\n")
    results = {}
    print("cosine between the rows of two positions:")
    for p, q in ((10, 11), (10, 12), (10, 20), (10, 60), (10, 200), (10, 400)):
        c = vectors.cosine(table[p], table[q])
        print(f"  position {p:3d} and {q:3d}: {c:5.2f}")
        results[f"{p}_{q}"] = c
    near = np.mean([vectors.cosine(table[p], table[p + 1]) for p in range(0, 500)])
    far = np.mean([vectors.cosine(table[p], table[p + 100]) for p in range(0, 400)])
    print(f"\naverage over all positions: neighbours {near:.2f}, 100 apart {far:.2f}")
    results["average_neighbours"], results["average_100_apart"] = float(near), float(far)
    gpt2 = realmodels.try_load("gpt2 config", realmodels.hub_config, "openai-community/gpt2")
    if gpt2:
        n, d = gpt2["n_positions"], gpt2["n_embd"]
        print(f"\nGPT-2 learned position table: {n:,} rows x {d} numbers = {n * d:,} numbers (its context window is {n:,} tokens)")
        results["gpt2_positions"] = n * d
    save_results("d10_learned_positions", results)


if __name__ == "__main__":
    run_and_save("d10_learned_positions", main)
