"""Demo 6 - special tokens and a chat template, all the way down to token ids.

What it does:
  1. adds a special token to our own tokenizer and shows that it is one id, never cut into pieces;
  2. shows the special end-of-text token in OpenAI's two encodings (and what happens to user text that contains it);
  3. renders a short CareDesk chat through the published chat template of a Qwen model and prints the raw text and the token ids.
What it needs: tiktoken, tokenizers and jinja2 (the first run downloads two small files from Hugging Face).
What to expect: the markers <|im_start|> and <|im_end|> are single tokens; the whole chat is a few dozen tokens.
Run it: uv run python -m demos.d06_special_and_chat
"""
from tokenizer_lab.bpe import ByteBPE
from tokenizer_lab.chat_template import fetch_config, render_published_template, render_simple
from tokenizer_lab.corpus import read
from tokenizer_lab.realtok import show
from tokenizer_lab.report import run_and_save, save_results

MESSAGES = [
    {"role": "system", "content": "You are CareDesk, an assistant for the operations team of a fictional hospital."},
    {"role": "user", "content": "Which room is Mr. Berg in?"},
    {"role": "assistant", "content": "Mr. Berg is in room 12."},
    {"role": "user", "content": "Please book a follow-up visit."},
]


def main():
    print("1. a special token in our own tokenizer")
    tok = ByteBPE.train(read("clinical_train"), 512)
    tok.special = {"<|end|>": len(tok.vocab)}
    text = "Room 12.<|end|>"
    ids = tok.encode(text)
    print(f"  {text!r} -> {ids}   (the last id, {ids[-1]}, is the special token)")
    safe = tok.encode(text, allow_special=False)
    safe_pieces = [tok.vocab[i].decode("utf-8", errors="backslashreplace") for i in safe]
    print(f"  the same text with special tokens switched off -> {len(safe)} ordinary tokens: {' | '.join(show(p) for p in safe_pieces)}")
    print("  our own tiny chat template:")
    for line in render_simple(MESSAGES[:3]).splitlines():
        print("   ", line)

    print("\n2. the end-of-text token in OpenAI's encodings")
    import tiktoken
    special = {}
    for name in ("cl100k_base", "o200k_base"):
        enc = tiktoken.get_encoding(name)
        eot = enc.encode("<|endoftext|>", allowed_special="all")
        print(f"  {name}: '<|endoftext|>' with special tokens allowed -> {eot}")
        try:
            enc.encode("<|endoftext|>")
        except ValueError as error:
            print(f"  {name}: by default, user text that contains it is refused: {str(error)[:70]}...")
        plain = enc.encode("<|endoftext|>", disallowed_special=())
        print(f"  {name}: with special tokens switched off it is cut into ordinary pieces -> {len(plain)} ids {plain}")
        special[name] = {"allowed": eot, "plain": plain}

    print("\n3. a chat rendered through a published chat template (Qwen2.5)")
    from tokenizers import Tokenizer
    repo = "Qwen/Qwen2.5-0.5B-Instruct"
    config = fetch_config(repo)
    rendered = render_published_template(config, MESSAGES, add_generation_prompt=True)
    print("raw text the model reads:")
    for line in rendered.splitlines():
        print("  |", line)
    qwen = Tokenizer.from_pretrained(repo)
    enc = qwen.encode(rendered, add_special_tokens=False)
    print(f"\n{len(rendered)} characters -> {len(enc.ids)} tokens")
    markers = [(t, i) for t, i in zip(enc.tokens, enc.ids) if t.startswith("<|")]
    print("special tokens in this chat (each one token):", sorted(set(markers), key=lambda m: m[1]))
    print("first 20 tokens:", " | ".join(show(t) for t in enc.tokens[:20]))
    print("first 20 ids   :", enc.ids[:20])
    save_results("d06_special_and_chat", {"special": special, "rendered": rendered, "qwen_tokens": len(enc.ids), "qwen_chars": len(rendered),
                                          "markers": [list(m) for m in sorted(set(markers), key=lambda m: m[1])], "first_tokens": enc.tokens[:20], "first_ids": enc.ids[:20]})


if __name__ == "__main__":
    run_and_save("d06_special_and_chat", main)
