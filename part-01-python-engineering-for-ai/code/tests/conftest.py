"""Shared test setup. httpx brings anyio, whose pytest plugin runs `async def` tests for us."""
import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio only (anyio can also run on trio)."""
    return "asyncio"
