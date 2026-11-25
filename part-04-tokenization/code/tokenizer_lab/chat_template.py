"""chat_template.py - turn a list of chat messages into the one long text a model reads.

What it does
    * `render_published_template` reads the chat template that a model publishes (a small Jinja program inside its
      tokenizer_config.json) and fills it with our messages,
    * `render_simple` is a tiny template of our own with the same idea, so the pattern is easy to see.
What it needs
    jinja2 (see pyproject.toml). The published template is downloaded from Hugging Face once.
What to expect
    A system message, the user and assistant turns, and a final marker that tells the model to write the next reply.
"""
import json
import urllib.request

from jinja2.sandbox import ImmutableSandboxedEnvironment

CONFIG_URL = "https://huggingface.co/{repo}/resolve/main/tokenizer_config.json"


def fetch_config(repo):
    """Download tokenizer_config.json of a Hugging Face repository (a small text file, not the model)."""
    with urllib.request.urlopen(CONFIG_URL.format(repo=repo), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def render_published_template(config, messages, add_generation_prompt=True):
    """Fill the model's own chat template with `messages` (a list of {'role': ..., 'content': ...})."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message):
        raise ValueError(message)

    env.globals["raise_exception"] = raise_exception
    template = env.from_string(config["chat_template"])
    return template.render(messages=messages, add_generation_prompt=add_generation_prompt,
                           bos_token=config.get("bos_token") or "", eos_token=config.get("eos_token") or "")


def render_simple(messages, add_generation_prompt=True):
    """Our own tiny template: <|start|>role newline content <|end|> newline for every message."""
    text = "".join(f"<|start|>{m['role']}\n{m['content']}<|end|>\n" for m in messages)
    return text + ("<|start|>assistant\n" if add_generation_prompt else "")
