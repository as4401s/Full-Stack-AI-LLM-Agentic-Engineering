"""Tests for the chat-template helpers (no download needed)."""
from tokenizer_lab.chat_template import render_published_template, render_simple

MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


def test_simple_template_has_markers_and_ends_with_the_assistant_cue():
    text = render_simple(MESSAGES)
    assert text == "<|start|>system\nBe brief.<|end|>\n<|start|>user\nHi<|end|>\n<|start|>assistant\n"
    assert render_simple(MESSAGES, add_generation_prompt=False).endswith("<|end|>\n")


def test_published_style_template_renders_with_jinja():
    config = {"chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"}
    text = render_published_template(config, MESSAGES)
    assert text.startswith("<|im_start|>system\nBe brief.<|im_end|>\n") and text.endswith("<|im_start|>assistant\n")
