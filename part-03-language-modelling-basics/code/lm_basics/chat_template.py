"""chat_template.py - turn a conversation into the single piece of text a language model continues.

What it does
    A chat is a list of messages, each with a role (system, user or assistant) and some text. A language
    model only continues text, so a chat template writes the whole conversation as one string, with a special
    marker in front of each message and one more marker that tells the model "the assistant speaks next".

What it needs
    Only the Python standard library.

What to expect
    format_chat([...]) returns a string such as:
        <|system|>You are CareDesk, a helpful front-desk assistant.<|end|>
        <|user|>When does the clinic open?<|end|>
        <|assistant|>
    Real models each have their own markers; the Hugging Face tokenizer's apply_chat_template method writes
    the right ones for each model (shown in the video, not run here, because it needs a model download).
"""

ROLES = ("system", "user", "assistant")
END_OF_MESSAGE = "<|end|>"


def format_chat(messages, add_generation_prompt=True):
    """Write a list of {'role': ..., 'content': ...} messages as one string.

    add_generation_prompt=True ends the text with the assistant marker, so the model's next words are the reply.
    """
    pieces = []
    for m in messages:
        if m["role"] not in ROLES:
            raise ValueError(f"unknown role {m['role']!r}; use one of {ROLES}")
        pieces.append(f"<|{m['role']}|>{m['content']}{END_OF_MESSAGE}\n")
    if add_generation_prompt:
        pieces.append("<|assistant|>\n")
    return "".join(pieces)
