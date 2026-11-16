"""04_idempotency.py - why a retry needs an idempotency key.

What it does: calls the "billing" provider, whose first reply is lost (it arrives after the client
gave up). The client times out and retries. Run once without a key (the retry repeats the work and
the charge) and once with a key (the server recognises the repeat and charges once).
What it needs: httpx and pydantic. Runs on any laptop.
What to expect: 2 charges without a key, 1 charge with a key.
Run it:  uv run python demos/04_idempotency.py
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from caredesk_fanout.fanout import fetch_one
from caredesk_fanout.mock_server import start_in_thread
from caredesk_fanout.models import ChatRequest
from caredesk_fanout.reliability import RetryPolicy


async def run_once(base_url: str, use_key: bool) -> None:
    """One logical request whose first reply is lost; report the ledger afterwards."""
    async with httpx.AsyncClient() as client:
        await client.post(f"{base_url}/v1/billing/reset", json={"lose": 1})    # next reply is lost
        policy = RetryPolicy(max_retries=2, base_s=0.1, jitter=0.0)
        outcome = await fetch_one(client, base_url, "billing", ChatRequest(prompt="Generate the summary"),
                                  asyncio.Semaphore(1), policy, timeout_s=0.5, use_idempotency=use_key)
        ledger = (await client.get(f"{base_url}/v1/billing/ledger")).json()
    label = "with an idempotency key   " if use_key else "without an idempotency key"
    print(f"{label}: work done {ledger['executions']}x, charged {ledger['charges']}x  (client result: {type(outcome).__name__})")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="    log: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)       # hide httpx's own per-request lines
    server, base_url = start_in_thread(lose_reply_s=1.5)
    print("The first reply is lost, so the client times out after 0.5 s and retries.")
    await run_once(base_url, use_key=False)
    await run_once(base_url, use_key=True)
    server.shutdown()


asyncio.run(main())
