"""Demo 7 - the cost of long text: attention time and score-table size as the number of tokens doubles.

What it does: times our plain attention() and PyTorch's built-in fused kernel for 128 up to 4,096 tokens (one head, 64 numbers per head, on the
computer's processor), and prints the size of the n x n score table that the plain version builds.
What it needs: torch.
What to expect: the plain version's score table grows exactly four times for every doubling of the tokens; its time follows the table once the
table is large (about four times per doubling at the big end, less at small sizes where fixed overheads dominate); the fused kernel never builds
the table and finishes in less than half the time at 4,096 tokens. The times depend on the computer, so only the pattern is the same from run to run.
Run it: uv run python -m demos.d07_cost_curve
"""
import statistics
import time

import torch
import torch.nn.functional as F

from attention_lab.attention import attention
from attention_lab.report import save_results


def median_ms(fn, repeats=5):
    """Run fn once to warm up, then return the median time of `repeats` runs in milliseconds."""
    fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main():
    torch.manual_seed(0)
    print(f"torch threads: {torch.get_num_threads()}; one head, 64 numbers per head, no mask, float32")
    print("     n   plain ms   built-in ms   score table (n x n x 4 bytes)   plain time vs previous row")
    rows, previous = {}, None
    for n in (128, 256, 512, 1024, 2048, 4096):
        q, k, v = (torch.randn(1, 1, n, 64) for _ in range(3))
        plain = median_ms(lambda: attention(q, k, v, causal=False))
        fused = median_ms(lambda: F.scaled_dot_product_attention(q, k, v))
        table_mb = n * n * 4 / 1e6
        ratio = f"{plain / previous:5.1f}x" if previous else "     -"
        print(f"{n:6d}   {plain:8.2f}   {fused:11.2f}   {table_mb:13.1f} MB                    {ratio}")
        rows[n] = {"plain_ms": plain, "builtin_ms": fused, "table_mb": table_mb}
        previous = plain
    print(f"\nthe score table at n = 4,096 holds {4096 * 4096:,} numbers; at n = 128 it holds {128 * 128:,}: {4096 * 4096 // (128 * 128):,} times more for {4096 // 128} times more text")
    save_results("d07_cost_curve", rows)
