"""Tests that use the real libraries. They are skipped when a library or its small download is not available."""
import pytest


def _encoding(name):
    tiktoken = pytest.importorskip("tiktoken")
    try:
        return tiktoken.get_encoding(name)
    except Exception as error:  # no network on the first run
        pytest.skip(f"{name} could not be loaded: {error}")


@pytest.mark.parametrize("name", ["cl100k_base", "o200k_base"])
def test_tiktoken_round_trip(name):
    enc = _encoding(name)
    text = "Die Untersuchungen zeigten keinen Herzinfarkt. 2026-03-14 😀"
    assert enc.decode(enc.encode(text)) == text


def test_leading_space_changes_the_token():
    enc = _encoding("o200k_base")
    assert enc.encode("nurse") != enc.encode(" nurse")
    assert len(enc.encode(" nurse")) == 1


def test_special_token_is_refused_in_user_text_by_default():
    enc = _encoding("cl100k_base")
    with pytest.raises(ValueError):
        enc.encode("<|endoftext|>")
    assert len(enc.encode("<|endoftext|>", allowed_special="all")) == 1
