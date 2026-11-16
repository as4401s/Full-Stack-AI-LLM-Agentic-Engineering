"""models.py - the typed shapes of everything that crosses a border of our service.

What it does: defines the request we send, the reply we expect from a provider, and the answer
we hand back to our own callers. Pydantic checks every value when an object is built, like a
customs desk that opens every parcel at the border.
What it needs: pydantic 2. Runs anywhere Python runs (laptop, Colab, server).
What to expect: bad data raises pydantic.ValidationError with a clear list of what is wrong.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class ChatRequest(BaseModel):
    """The question we send to a provider."""

    prompt: str = Field(min_length=1, max_length=2000)    # an empty prompt is rejected
    max_tokens: int = Field(default=64, ge=1, le=512)     # keeps answers short and cheap


class Usage(BaseModel):
    """How many tokens a call used (a token is a small piece of text the model reads or writes)."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> int:
        """Prompt tokens plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens


class ModelReply(BaseModel):
    """What a provider sends back. We check it before trusting it."""

    model: str = Field(min_length=1)
    text: str = Field(min_length=1)     # an empty answer is treated as a failure
    usage: Usage


class Answer(BaseModel):
    """One checked answer from one provider."""

    source: str
    text: str
    latency_s: float = Field(ge=0)
    tokens: int = Field(ge=0)


class Failure(BaseModel):
    """One provider that could not give a usable answer, and why."""

    source: str
    error: str


class FanOutResult(BaseModel):
    """Everything one fan-out call returns: answers, failures and the wall-clock time."""

    answers: list[Answer]
    failures: list[Failure]
    wall_s: float = Field(ge=0)
