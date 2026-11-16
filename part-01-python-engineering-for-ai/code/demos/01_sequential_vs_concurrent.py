"""01_sequential_vs_concurrent.py - three model calls: one after another, then all at once.

What it does: starts the local mock model API (each call takes 3 seconds), asks three providers the
same question sequentially and then concurrently, and prints both wall-clock times.
What it needs: httpx and pydantic (uv sync). Runs on a laptop; no paid service and no GPU.
What to expect: about 9 s one after another, about 3 s all at once.
Run it:  uv run python demos/01_sequential_vs_concurrent.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # lets the demo import the package

from caredesk_fanout.fanout import fan_out, fan_out_sequential
from caredesk_fanout.mock_server import start_in_thread
from caredesk_fanout.models import ChatRequest

DELAY_S = 3.0
PROVIDERS = ["alpha", "beta", "gamma"]


async def main() -> None:
    server, base_url = start_in_thread(delay_s=DELAY_S)          # the stand-in model API
    request = ChatRequest(prompt="Summarise this synthetic referral note")
    print(f"Three model calls, {DELAY_S:.1f} s each (local mock server)")
    slow = await fan_out_sequential(base_url, PROVIDERS, request)
    print(f"one after another : {slow.wall_s:5.2f} s")
    fast = await fan_out(base_url, PROVIDERS, request)
    print(f"all at once       : {fast.wall_s:5.2f} s")
    print()
    for a in fast.answers:
        print(f"{a.source:<6} {a.latency_s:4.2f} s  {a.tokens:>2} tokens  {a.text}")
    server.shutdown()


asyncio.run(main())
