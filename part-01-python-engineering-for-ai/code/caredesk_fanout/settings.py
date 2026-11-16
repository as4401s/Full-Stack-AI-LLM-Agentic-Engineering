"""settings.py - every knob of the service in one typed place.

What it does: reads settings from environment variables (and an optional .env file) and checks
them with Pydantic, so a typo such as CAREDESK_MAX_CONCURRENCY=zero fails at start-up, loudly.
What it needs: pydantic-settings. Copy .env.example to .env to change values.
What to expect: Settings() returns an object with safe defaults for a local lab.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration, with names that start with CAREDESK_ in the environment."""

    model_config = SettingsConfigDict(env_prefix="CAREDESK_", env_file=".env", extra="ignore")

    base_url: str = "http://127.0.0.1:8765"          # where the (mock) model API lives
    max_concurrency: int = Field(default=5, ge=1, le=64)   # calls in flight at the same time
    timeout_s: float = Field(default=10.0, gt=0)     # give up on one HTTP call after this long
    max_retries: int = Field(default=3, ge=0, le=8)  # extra attempts after the first one
    backoff_base_s: float = Field(default=1.0, gt=0)  # first wait; it doubles each time
    backoff_cap_s: float = Field(default=30.0, gt=0)  # no wait is longer than this
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Build the settings once and reuse them (reading the environment is cheap but not free)."""
    return Settings()
