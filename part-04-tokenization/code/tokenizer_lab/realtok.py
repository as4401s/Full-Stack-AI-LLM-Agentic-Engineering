"""realtok.py - load the real tokenizers used in the lesson and show their pieces in a readable way.

What it does
    * tiktoken encodings (cl100k_base and o200k_base) published by OpenAI,
    * Hugging Face tokenizers: GPT-2 (byte-level BPE), BERT (WordPiece), T5 (Unigram), Qwen (byte-level BPE with chat markers),
    * a helper that turns every kind of tokenizer into the same two things: a list of readable pieces and a list of ids.
What it needs
    tiktoken and tokenizers (see pyproject.toml). The first run downloads a few small files (a few megabytes in total);
    later runs use the saved copies.
What to expect
    Each loader returns an object with .name, .vocab_size, .pieces(text), .ids(text) and .count(text).
    When a download is not possible the loader raises an error whose message says what was missing.
"""

# the Hugging Face repositories used in the lesson (only the small tokenizer files are downloaded, never the models)
HF_REPOS = {
    "gpt2": "gpt2",                                  # byte-level BPE, 50,257 pieces
    "bert": "bert-base-uncased",                     # WordPiece, lowercases everything
    "t5": "t5-small",                                # Unigram through SentencePiece
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",            # byte-level BPE with chat markers
}


class Wrapped:
    """One tokenizer behind one simple interface, whatever library it comes from."""

    def __init__(self, name, vocab_size, to_ids, to_piece):
        self.name, self.vocab_size, self._ids, self._piece = name, vocab_size, to_ids, to_piece

    def ids(self, text):
        """Token ids for a text (no start or end markers added)."""
        return self._ids(text)

    def pieces(self, text):
        """The pieces for a text as readable strings."""
        return [self._piece(i) for i in self._ids(text)]

    def count(self, text):
        """How many tokens a text costs."""
        return len(self._ids(text))


def tiktoken_encoding(name):
    """Load one of OpenAI's published encodings, for example 'cl100k_base' or 'o200k_base'."""
    import tiktoken
    enc = tiktoken.get_encoding(name)
    return Wrapped(name, enc.n_vocab, lambda t: enc.encode(t), lambda i: enc.decode_single_token_bytes(i).decode("utf-8", errors="backslashreplace"))


def hf_tokenizer(key):
    """Load a Hugging Face tokenizer by the short key in HF_REPOS (downloads the small tokenizer file once)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(HF_REPOS[key])
    return Wrapped(HF_REPOS[key], tok.get_vocab_size(), lambda t: tok.encode(t, add_special_tokens=False).ids, lambda i: tok.id_to_token(i))


def our_tokenizer(vocab_size=512):
    """Our own byte-level BPE from bpe.py, trained on the invented clinical text."""
    from tokenizer_lab.bpe import ByteBPE
    from tokenizer_lab.corpus import read
    tok = ByteBPE.train(read("clinical_train"), vocab_size)
    return Wrapped(f"our BPE ({len(tok.vocab)})", len(tok.vocab), tok.encode, lambda i: tok.vocab[i].decode("utf-8", errors="backslashreplace"))


def try_load(label, loader, *args):
    """Run a loader; on a failure return None and print one short line so the demo can carry on."""
    try:
        return loader(*args)
    except Exception as error:  # a download or an install can fail for many reasons; the demo reports it and moves on
        print(f"  [{label} unavailable: {type(error).__name__}: {str(error)[:90]}]")
        return None


def show(piece):
    """Make spaces visible in a printed piece."""
    return piece.replace(" ", "_").replace("\n", "\\n")


def all_tokenizers():
    """Return the six tokenizers compared in the lesson, skipping any that cannot be loaded."""
    loaded = [
        try_load("our BPE", our_tokenizer, 512),
        try_load("gpt2", hf_tokenizer, "gpt2"),
        try_load("cl100k_base", tiktoken_encoding, "cl100k_base"),
        try_load("o200k_base", tiktoken_encoding, "o200k_base"),
        try_load("bert", hf_tokenizer, "bert"),
        try_load("t5", hf_tokenizer, "t5"),
    ]
    return [t for t in loaded if t is not None]
