"""Demo 2 - the embedding table is a lookup: token id 3 picks row 3.

What it does: builds a tiny table with PyTorch's nn.Embedding (six tokens, three numbers each), looks up a row, shows that the lookup equals a
one-hot vector times the table, and shows that only the rows used in a step receive a learning signal. Then it counts the numbers in real tables.
What it needs: torch.
What to expect: the same three numbers three ways; a gradient that is zero everywhere except at the looked-up row.
Run it: uv run python -m demos.d02_table_lookup
"""
import numpy as np
import torch
from torch import nn

from embedding_lab.report import run_and_save, save_results

TOKENS = ["the", "patient", "has", "kidney", "failure", "."]
ROWS = [[0.1, 0.0, 0.2], [0.9, 0.3, 0.1], [0.2, 0.1, 0.0], [0.2, 0.8, 0.7], [0.1, 0.6, 0.9], [0.0, 0.0, 0.1]]


def main():
    print("torch", torch.__version__)
    table = nn.Embedding(len(TOKENS), 3)
    with torch.no_grad():
        table.weight.copy_(torch.tensor(ROWS))                       # we write our own numbers so the example is easy to follow
    print(f"\na table of {len(TOKENS)} tokens x 3 numbers = {table.weight.numel()} numbers")
    for i, tok in enumerate(TOKENS):
        print(f"  row {i} ({tok!r}): {[round(x, 2) for x in table.weight[i].tolist()]}")

    ids = torch.tensor([0, 1, 2, 3, 4])                              # "the patient has kidney failure"
    looked_up = table(ids)
    print(f"\nlookup for ids {ids.tolist()} gives a block of shape {tuple(looked_up.shape)}:")
    for i, row in zip(ids.tolist(), looked_up.tolist()):
        print(f"  id {i} -> {[round(x, 2) for x in row]}")

    one_hot = torch.zeros(len(TOKENS))
    one_hot[3] = 1.0
    product = one_hot @ table.weight
    print(f"\none-hot vector for id 3: {one_hot.tolist()}")
    print(f"one-hot x table        : {[round(x, 2) for x in product.tolist()]}")
    print(f"table(3)               : {[round(x, 2) for x in table(torch.tensor(3)).tolist()]}   (same numbers, no multiplication needed)")

    table.zero_grad()
    target = torch.tensor([1.0, 0.0, 0.0])
    loss = ((table(torch.tensor(3)) - target) ** 2).sum()
    loss.backward()
    print("\nlearning signal (gradient) after one lookup of id 3:")
    for i, tok in enumerate(TOKENS):
        print(f"  row {i} ({tok!r}): {[round(x, 2) for x in table.weight.grad[i].tolist()]}")

    print("\nhow many numbers real tables hold (vocabulary x numbers per token):")
    sizes = {"our tiny table": (6, 3), "MiniLM token table": (30522, 384), "GPT-2 token table": (50257, 768), "Qwen2.5-0.5B token table": (151936, 896)}
    for name, (rows, width) in sizes.items():
        print(f"  {name:26s} {rows:>7,} x {width:<4} = {rows * width:>13,} numbers")
    save_results("d02_table_lookup", {"lookup_id3": ROWS[3], "gradient_rows": table.weight.grad.tolist(), "sizes": {k: v[0] * v[1] for k, v in sizes.items()}})


if __name__ == "__main__":
    run_and_save("d02_table_lookup", main)
