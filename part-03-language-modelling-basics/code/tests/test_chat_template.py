"""Tests for the chat template."""
import pytest

from lm_basics.chat_template import format_chat

CHAT = [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "Hi"}]


def test_template_text():
    assert format_chat(CHAT) == "<|system|>Be kind.<|end|>\n<|user|>Hi<|end|>\n<|assistant|>\n"


def test_without_generation_prompt_ends_after_last_message():
    assert format_chat(CHAT, add_generation_prompt=False).endswith("<|user|>Hi<|end|>\n")


def test_unknown_role_is_reported():
    with pytest.raises(ValueError):
        format_chat([{"role": "robot", "content": "x"}])
