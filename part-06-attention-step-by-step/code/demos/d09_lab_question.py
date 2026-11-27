"""Demo 9 - the lab question: change the query and watch the weights move.

What it does: (1) swaps the query [1, 0] for [0, 1] in the three-token example and prints the new weights and output, (2) repeats the "divide by
the square root of d" test with PyTorch on 64 numbers per head.
What it needs: torch.
What to expect: query [0, 1] gives weights 0.1978, 0.4011, 0.4011 (keys 2 and 3 tie) and output [0.7967, 1.2033]; for d = 64 the plain scores give a
biggest weight close to 1, the scaled scores a much smaller one.
Run it: uv run python -m demos.d09_lab_question
"""
import math

import torch

from attention_lab.attention import attention
from attention_lab.report import save_results


def main():
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).unsqueeze(0).unsqueeze(0)
    values = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]).unsqueeze(0).unsqueeze(0)
    results = {}
    for query in ([1.0, 0.0], [0.0, 1.0]):
        q = torch.tensor([query]).unsqueeze(0).unsqueeze(0)
        out, w = attention(q, keys, values, causal=False)
        print(f"query {query}: scores {[int(s) for s in (q[0, 0] @ keys[0, 0].T)[0].tolist()]}  weights {[round(x, 4) for x in w[0, 0, 0].tolist()]}  output {[round(x, 4) for x in out[0, 0, 0].tolist()]}")
        results[str(query)] = {"weights": w[0, 0, 0], "output": out[0, 0, 0]}
    torch.manual_seed(1)
    d = 64
    q, k = torch.randn(1, 1, 1, d), torch.randn(1, 1, 8, d)
    raw = (q @ k.transpose(-2, -1))[0, 0, 0]
    print(f"\nd = {d}, eight keys: raw scores {[round(x, 1) for x in raw.tolist()]}")
    print(f"  softmax of the raw scores:    biggest weight {torch.softmax(raw, -1).max().item():.3f}")
    print(f"  softmax of scores / sqrt({d}):  biggest weight {torch.softmax(raw / math.sqrt(d), -1).max().item():.3f}")
    results["scaling"] = {"raw_max": torch.softmax(raw, -1).max().item(), "scaled_max": torch.softmax(raw / math.sqrt(d), -1).max().item()}
    save_results("d09_lab_question", results)
