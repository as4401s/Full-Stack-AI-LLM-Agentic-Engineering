"""fanout.py - ask several providers the same question at the same time.

What it does: sends one ChatRequest to many providers concurrently (at most N in flight),
retries temporary failures, checks every reply with Pydantic, and returns typed results.
What it needs: httpx and pydantic. A server that answers POST {base_url}/v1/<provider>/chat
(the lesson uses the local mock server in mock_server.py, so it is free).
What to expect: with three providers that each take 3 s, fan_out takes about 3 s, while
fan_out_sequential takes about 9 s.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable

import httpx
from pydantic import ValidationError

from .models import Answer, ChatRequest, FanOutResult, Failure, ModelReply
from .reliability import RetryPolicy, call_with_retries, new_idempotency_key

log = logging.getLogger(__name__)
Sleep = Callable[[float], Awaitable[None]]


async def fetch_one(
    client: httpx.AsyncClient,
    base_url: str,
    provider: str,
    request: ChatRequest,
    sem: asyncio.Semaphore,
    policy: RetryPolicy,
    *,
    timeout_s: float = 10.0,
    use_idempotency: bool = True,
    sleep: Sleep = asyncio.sleep,
    rng: random.Random | None = None,
) -> Answer | Failure:
    """Ask one provider. Never raises for provider trouble: it returns a Failure instead."""
    # One key per logical request, reused by every retry, so the server can spot repeats.
    key = new_idempotency_key() if use_idempotency else None
    started = time.perf_counter()

    async def send() -> httpx.Response:
        headers = {"Idempotency-Key": key} if key else {}
        async with sem:                                   # wait here when too many calls are in flight
            response = await client.post(
                f"{base_url}/v1/{provider}/chat",
                json=request.model_dump(),
                headers=headers,
                timeout=timeout_s,                        # never wait forever for one call
            )
        response.raise_for_status()                       # turn 4xx/5xx into an exception
        return response

    try:
        response = await call_with_retries(send, policy, sleep=sleep, rng=rng, label=provider)
        reply = ModelReply.model_validate_json(response.content)    # the customs desk
    except (httpx.HTTPError, ValidationError) as exc:
        log.error("%s failed: %s", provider, exc)
        return Failure(source=provider, error=f"{type(exc).__name__}: {exc}"[:200])
    return Answer(
        source=provider,
        text=reply.text,
        latency_s=round(time.perf_counter() - started, 3),
        tokens=reply.usage.total,
    )


async def fan_out(
    base_url: str,
    providers: list[str],
    request: ChatRequest,
    *,
    max_concurrency: int = 5,
    timeout_s: float = 10.0,
    policy: RetryPolicy | None = None,
    client: httpx.AsyncClient | None = None,
    sleep: Sleep = asyncio.sleep,
    rng: random.Random | None = None,
) -> FanOutResult:
    """Ask every provider at the same time, with at most `max_concurrency` calls in flight."""
    policy = policy or RetryPolicy()
    sem = asyncio.Semaphore(max_concurrency)                # the door host: lets N in at a time
    started = time.perf_counter()

    async def run(c: httpx.AsyncClient) -> list[Answer | Failure]:
        coros = (
            fetch_one(c, base_url, p, request, sem, policy, timeout_s=timeout_s, sleep=sleep, rng=rng)
            for p in providers
        )
        return await asyncio.gather(*coros)                 # start them all, wait for them all

    if client is None:
        async with httpx.AsyncClient() as own_client:       # one shared client reuses connections
            results = await run(own_client)
    else:
        results = await run(client)
    return _collect(results, time.perf_counter() - started)


async def fan_out_sequential(
    base_url: str,
    providers: list[str],
    request: ChatRequest,
    *,
    timeout_s: float = 10.0,
    policy: RetryPolicy | None = None,
    client: httpx.AsyncClient | None = None,
) -> FanOutResult:
    """The slow way, kept for comparison: ask one provider, wait, ask the next."""
    policy = policy or RetryPolicy()
    sem = asyncio.Semaphore(1)
    started = time.perf_counter()
    results: list[Answer | Failure] = []

    async def run(c: httpx.AsyncClient) -> None:
        for p in providers:
            results.append(await fetch_one(c, base_url, p, request, sem, policy, timeout_s=timeout_s))

    if client is None:
        async with httpx.AsyncClient() as own_client:
            await run(own_client)
    else:
        await run(client)
    return _collect(results, time.perf_counter() - started)


def _collect(results: list[Answer | Failure], wall_s: float) -> FanOutResult:
    """Split mixed results into answers and failures, in a stable order."""
    answers = sorted((r for r in results if isinstance(r, Answer)), key=lambda a: a.source)
    failures = sorted((r for r in results if isinstance(r, Failure)), key=lambda f: f.source)
    return FanOutResult(answers=answers, failures=failures, wall_s=round(wall_s, 3))
