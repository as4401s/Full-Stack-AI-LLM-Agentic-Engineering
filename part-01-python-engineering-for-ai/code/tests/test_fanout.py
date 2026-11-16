"""Tests for fanout.py. No real network: httpx.MockTransport plays the provider inside the test."""
import asyncio
import json
import random
from pathlib import Path

import httpx
import pytest

from caredesk_fanout.fanout import fan_out, fan_out_sequential
from caredesk_fanout.models import ChatRequest
from caredesk_fanout.reliability import RetryPolicy

pytestmark = pytest.mark.anyio
BASE = "http://providers.test"
REQUEST = ChatRequest(prompt="Summarise this synthetic referral note")
SNAPSHOT = Path(__file__).parent / "snapshots" / "fanout_shape.json"


def reply(provider: str, text: str | None = None) -> dict:
    """A well-formed provider reply."""
    return {
        "model": f"{provider}-mini",
        "text": text or f"[{provider}] ok",
        "usage": {"prompt_tokens": 5, "completion_tokens": 12},
    }


def provider_of(request: httpx.Request) -> str:
    return request.url.path.split("/")[2]


async def no_sleep(_: float) -> None:
    """Stand-in for asyncio.sleep so retry tests finish instantly."""


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_concurrent_is_faster_than_sequential() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)                              # each provider "thinks" for 0.2 s
        return httpx.Response(200, json=reply(provider_of(request)))

    names = ["alpha", "beta", "gamma"]
    async with client_for(handler) as c:
        fast = await fan_out(BASE, names, REQUEST, client=c)
        slow = await fan_out_sequential(BASE, names, REQUEST, client=c)
    assert len(fast.answers) == len(slow.answers) == 3
    assert fast.wall_s < 0.45                                 # about 0.2 s: all at once
    assert slow.wall_s >= 0.6                                 # about 0.6 s: one after another


async def test_semaphore_never_allows_more_than_the_limit() -> None:
    state = {"now": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.05)
        state["now"] -= 1
        return httpx.Response(200, json=reply(provider_of(request)))

    names = [f"p{i}" for i in range(10)]
    async with client_for(handler) as c:
        result = await fan_out(BASE, names, REQUEST, max_concurrency=3, client=c)
    assert len(result.answers) == 10
    assert state["peak"] == 3                                 # reached the limit, never beyond it


async def test_one_failing_provider_does_not_sink_the_others() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if provider_of(request) == "bad":
            return httpx.Response(500, json={"error": "down"})
        return httpx.Response(200, json=reply(provider_of(request)))

    async with client_for(handler) as c:
        result = await fan_out(BASE, ["alpha", "bad", "gamma"], REQUEST, client=c,
                               policy=RetryPolicy(max_retries=1), sleep=no_sleep)
    assert [a.source for a in result.answers] == ["alpha", "gamma"]
    assert [f.source for f in result.failures] == ["bad"]


async def test_a_malformed_reply_is_stopped_at_the_border() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reply("alpha", text="") | {"text": ""})   # empty text

    async with client_for(handler) as c:
        result = await fan_out(BASE, ["alpha"], REQUEST, client=c)
    assert not result.answers and "ValidationError" in result.failures[0].error


async def test_idempotency_key_is_the_same_on_every_retry() -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Idempotency-Key"])
        if len(seen) < 3:
            return httpx.Response(503, json={"error": "try later"})
        return httpx.Response(200, json=reply("alpha"))

    async with client_for(handler) as c:
        result = await fan_out(BASE, ["alpha"], REQUEST, client=c, sleep=no_sleep, rng=random.Random(1))
    assert len(result.answers) == 1
    assert len(seen) == 3 and len(set(seen)) == 1             # three attempts, one key


async def test_varying_model_text_is_checked_by_contract_not_by_exact_words() -> None:
    """Model text changes from run to run, so we test what must always hold."""
    rng = random.Random()
    options = ["Looks fine.", "Please review the scan.", "Escalate to the on-call doctor."]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reply(provider_of(request), rng.choice(options)))

    names = ["alpha", "beta", "gamma", "delta"]
    async with client_for(handler) as c:
        result = await fan_out(BASE, names, REQUEST, client=c)
    assert len(result.answers) + len(result.failures) == len(names)      # nobody is lost
    for a in result.answers:
        assert a.text.strip() and a.tokens == 17 and a.source in names   # the contract


async def test_snapshot_of_the_result_shape() -> None:
    """A snapshot test: the structure (not the wording or the timing) must match the saved file."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reply(provider_of(request)))

    async with client_for(handler) as c:
        result = await fan_out(BASE, ["beta", "alpha"], REQUEST, client=c)
    shape = {"answers": [{"source": a.source, "tokens": a.tokens} for a in result.answers],
             "failures": [f.source for f in result.failures]}
    if not SNAPSHOT.exists():                                  # first run: write the snapshot
        SNAPSHOT.write_text(json.dumps(shape, indent=2) + "\n")
    assert shape == json.loads(SNAPSHOT.read_text())
