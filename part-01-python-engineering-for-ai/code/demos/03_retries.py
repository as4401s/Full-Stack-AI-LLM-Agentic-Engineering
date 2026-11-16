"""03_retries.py - exponential backoff with jitter, on a provider that fails twice.

What it does: asks the "flaky" provider, which answers HTTP 503 twice before it succeeds. The demo
prints every failed attempt and the wait before the next one. The waits are scaled down
(base 0.25 s instead of 1 s) so the demo finishes quickly; the doubling pattern is the same.
What it needs: httpx and pydantic. Runs on any laptop.
What to expect: waits of about 0.25 s then 0.5 s, each with a small random extra.
Run it:  uv run python demos/03_retries.py
"""
import asyncio
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from caredesk_fanout.fanout import fan_out
from caredesk_fanout.mock_server import start_in_thread
from caredesk_fanout.models import ChatRequest
from caredesk_fanout.reliability import RetryPolicy, backoff_delay


async def main() -> None:
    # The retry helper writes a log line for every failed attempt; show those lines with a clock.
    logging.basicConfig(level=logging.INFO, format="%(relativeCreated)6.0f ms  %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)       # hide httpx's own per-request lines
    server, base_url = start_in_thread(flaky_failures=2)
    policy = RetryPolicy(max_retries=3, base_s=0.25, cap_s=30.0, jitter=0.5)
    print("asking the flaky provider (fails twice, then answers)")

    # Show what the policy plans, then let the real client follow the same random sequence.
    planner = random.Random(7)                                   # fixed seed: the same numbers every run
    plan = [backoff_delay(k, policy, planner) for k in range(3)]
    print("planned waits (s):", ", ".join(f"{w:.2f}" for w in plan))

    result = await fan_out(base_url, ["flaky"], ChatRequest(prompt="hello"), policy=policy, rng=random.Random(7))
    a = result.answers[0]
    print(f"success after {result.wall_s:.2f} s: {a.text}")
    server.shutdown()


asyncio.run(main())
