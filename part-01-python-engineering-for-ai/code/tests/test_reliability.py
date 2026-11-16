"""Tests for reliability.py: backoff, jitter, retries and idempotency keys."""
import random

import httpx
import pytest

from caredesk_fanout.reliability import (
    RetryPolicy,
    backoff_delay,
    call_with_retries,
    is_retryable,
    new_idempotency_key,
)

pytestmark = pytest.mark.anyio


def test_backoff_doubles_then_hits_the_cap() -> None:
    policy = RetryPolicy(base_s=1.0, cap_s=30.0, jitter=0.0)          # no jitter: exact numbers
    rng = random.Random(0)
    assert [backoff_delay(k, policy, rng) for k in range(4)] == [1.0, 2.0, 4.0, 8.0]
    assert backoff_delay(10, policy, rng) == 30.0                       # 1024 would be capped to 30


def test_jitter_property_delay_stays_inside_its_bounds() -> None:
    """A property check: for many random cases, the delay is between the wait and wait x 1.5."""
    policy = RetryPolicy(base_s=1.0, cap_s=30.0, jitter=0.5)
    for seed in range(2000):
        rng = random.Random(seed)
        attempt = rng.randrange(0, 8)
        wait = min(30.0, 2**attempt)
        delay = backoff_delay(attempt, policy, rng)
        assert wait <= delay <= wait * 1.5


def test_retryable_and_non_retryable_failures() -> None:
    req = httpx.Request("POST", "http://x")
    assert is_retryable(httpx.ConnectError("boom"))
    assert is_retryable(httpx.HTTPStatusError("e", request=req, response=httpx.Response(503, request=req)))
    assert not is_retryable(httpx.HTTPStatusError("e", request=req, response=httpx.Response(404, request=req)))
    assert not is_retryable(ValueError("a bug is not worth retrying"))


async def test_retries_then_succeeds_without_real_waiting() -> None:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:        # records the wait instead of waiting
        waits.append(seconds)

    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("temporary")
        return "ok"

    policy = RetryPolicy(max_retries=3, base_s=1.0, jitter=0.0)
    assert await call_with_retries(flaky, policy, sleep=fake_sleep) == "ok"
    assert calls["n"] == 3 and waits == [1.0, 2.0]


async def test_gives_up_after_max_retries() -> None:
    async def no_sleep(_: float) -> None:
        pass

    calls = {"n": 0}

    async def always_down() -> str:
        calls["n"] += 1
        raise httpx.ConnectError("down")

    with pytest.raises(httpx.ConnectError):
        await call_with_retries(always_down, RetryPolicy(max_retries=2), sleep=no_sleep)
    assert calls["n"] == 3                                  # one try plus two retries


async def test_a_client_error_is_not_retried() -> None:
    req = httpx.Request("POST", "http://x")
    calls = {"n": 0}

    async def bad_request() -> str:
        calls["n"] += 1
        raise httpx.HTTPStatusError("bad", request=req, response=httpx.Response(400, request=req))

    with pytest.raises(httpx.HTTPStatusError):
        await call_with_retries(bad_request, RetryPolicy(max_retries=3))
    assert calls["n"] == 1


def test_idempotency_keys_are_unique() -> None:
    assert len({new_idempotency_key() for _ in range(1000)}) == 1000
