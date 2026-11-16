"""Tests for models.py: the customs desk accepts good parcels and stops bad ones."""
import pytest
from pydantic import ValidationError

from caredesk_fanout.models import ChatRequest, ModelReply

GOOD = {"model": "alpha-mini", "text": "All clear.", "usage": {"prompt_tokens": 5, "completion_tokens": 7}}


def test_good_reply_is_accepted_and_total_is_computed() -> None:
    reply = ModelReply.model_validate(GOOD)
    assert reply.usage.total == 12


def test_lax_mode_converts_a_numeric_string() -> None:
    """Pydantic converts "12" to 12 by default (lax mode); that is a choice worth knowing about."""
    reply = ModelReply.model_validate({**GOOD, "usage": {"prompt_tokens": "5", "completion_tokens": 7}})
    assert reply.usage.prompt_tokens == 5


@pytest.mark.parametrize(
    "bad",
    [
        {**GOOD, "text": ""},                                                      # empty answer
        {**GOOD, "usage": {"prompt_tokens": -3, "completion_tokens": 7}},          # negative count
        {**GOOD, "usage": {"prompt_tokens": 5, "completion_tokens": "twelve"}},    # not a number
        {"model": "alpha-mini", "usage": {"prompt_tokens": 5, "completion_tokens": 7}},  # missing text
    ],
)
def test_bad_replies_are_rejected(bad: dict) -> None:
    with pytest.raises(ValidationError):
        ModelReply.model_validate(bad)


def test_request_limits() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(prompt="")
    with pytest.raises(ValidationError):
        ChatRequest(prompt="hi", max_tokens=0)
    assert ChatRequest(prompt="hi").max_tokens == 64
