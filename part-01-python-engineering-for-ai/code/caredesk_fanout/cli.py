"""cli.py - a small command line for the fan-out client.

What it does: asks several providers one question and prints a compact table.
What it needs: the mock server running (python -m caredesk_fanout.mock_server) or any server with
the same URL shape. Settings come from environment variables (see settings.py).
What to expect: one line per provider and the total wall-clock time.
Example:  uv run python -m caredesk_fanout.cli "Summarise this referral" alpha beta gamma
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from .fanout import fan_out, fan_out_sequential
from .models import ChatRequest
from .reliability import RetryPolicy
from .settings import get_settings


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the fan-out, print the results."""
    ap = argparse.ArgumentParser(description="Ask several providers the same question at once")
    ap.add_argument("prompt", help="the question to ask")
    ap.add_argument("providers", nargs="+", help="provider names, for example alpha beta gamma")
    ap.add_argument("--sequential", action="store_true", help="ask one after another (slow, for comparison)")
    args = ap.parse_args(argv)

    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)       # httpx logs every request at INFO; keep the output short
    request = ChatRequest(prompt=args.prompt)
    policy = RetryPolicy(max_retries=s.max_retries, base_s=s.backoff_base_s, cap_s=s.backoff_cap_s)
    if args.sequential:
        result = asyncio.run(fan_out_sequential(s.base_url, args.providers, request, timeout_s=s.timeout_s, policy=policy))
    else:
        result = asyncio.run(fan_out(s.base_url, args.providers, request, max_concurrency=s.max_concurrency, timeout_s=s.timeout_s, policy=policy))

    for a in result.answers:
        print(f"{a.source:<8} {a.latency_s:>5.2f} s  {a.tokens:>3} tokens  {a.text}")
    for f in result.failures:
        print(f"{f.source:<8} FAILED  {f.error}")
    print(f"wall-clock time: {result.wall_s:.2f} s")


if __name__ == "__main__":
    main()
