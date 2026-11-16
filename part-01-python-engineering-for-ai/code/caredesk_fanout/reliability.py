"""reliability.py - timeouts, retries with exponential backoff and jitter, idempotency keys.

What it does: wraps one network call so that short-lived trouble (a timeout, a 503) is retried
politely, and gives every logical request a stable key so a retry never repeats its effect.
What it needs: httpx (only to recognise its error types). Runs anywhere Python runs.
What to expect: wait times of about base, 2x base, 4x base ... plus a random extra, up to a cap.
"""
from __future__ import annotations

import asyncio
import logging
import random
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

log = logging.getLogger(__name__)
T = TypeVar("T")

# HTTP status codes that usually mean "try again in a moment" (timeout, rate limit, server trouble).
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    """How patient we are. Frozen, so one policy can be shared safely."""

    max_retries: int = 3      # extra attempts after the first one
    base_s: float = 1.0       # first wait, in seconds
    cap_s: float = 30.0       # the longest single wait
    jitter: float = 0.5       # random extra of up to this fraction of the wait


def is_retryable(exc: BaseException) -> bool:
    """True when the failure looks temporary: network trouble, a timeout, or a retryable status."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)   # includes timeouts and connection errors


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    """Seconds to wait after failed attempt number `attempt` (0 for the first failure).

    The wait doubles each time (base, 2 x base, 4 x base ...), never exceeds the cap, and then a
    random extra of up to `jitter` times the wait is added, so many clients do not retry together.
    """
    wait = min(policy.cap_s, policy.base_s * 2**attempt)
    return wait + rng.uniform(0, wait * policy.jitter)


def describe_failure(exc: BaseException) -> str:
    """A short human-friendly name for a failure, for log lines (for example "HTTP 503" or "ReadTimeout")."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def new_idempotency_key() -> str:
    """A fresh random key for one logical request. Reuse it on every retry of that request."""
    return uuid.uuid4().hex


async def call_with_retries(
    send: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    label: str = "call",
) -> T:
    """Run `send()`; retry temporary failures with backoff; re-raise anything else.

    `sleep` and `rng` are parameters so tests can run instantly and repeatably.
    """
    rng = rng or random.Random()
    for attempt in range(policy.max_retries + 1):
        try:
            return await send()
        except Exception as exc:   # we decide below whether it is worth another try
            if attempt == policy.max_retries or not is_retryable(exc):
                raise
            delay = backoff_delay(attempt, policy, rng)
            log.warning("%s: attempt %d failed (%s); retrying in %.2f s", label, attempt + 1, describe_failure(exc), delay)
            if on_retry:
                on_retry(attempt + 1, exc, delay)
            await sleep(delay)
    raise AssertionError("unreachable")   # the loop always returns or raises
