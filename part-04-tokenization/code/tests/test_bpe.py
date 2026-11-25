"""Tests for the byte-level BPE tokenizer."""
from tokenizer_lab.bpe import ByteBPE, pretokenize
from tokenizer_lab.corpus import read

TEXT = "the nurse checks the chart. the nurse greets the patient. " * 30


def test_pretokenize_keeps_leading_spaces_and_cuts_digits_in_threes():
    assert pretokenize("The nurse gave 500 mg") == ["The", " nurse", " gave", " 500", " mg"]
    assert pretokenize(" 2026-03-14") == [" 202", "6", "-", "03", "-", "14"]
    assert "".join(pretokenize("a  b\n\nc")) == "a  b\n\nc"


def test_vocabulary_starts_with_256_bytes_and_grows_by_one_per_merge():
    tok = ByteBPE.train(TEXT, 300)
    assert len(tok.vocab) == 256 + len(tok.merges) <= 300
    assert tok.vocab[:256] == [bytes([i]) for i in range(256)]


def test_round_trip_is_exact_for_any_text():
    tok = ByteBPE.train(TEXT, 300)
    for text in ["", "the nurse", "Größe café 😀 €", "日本語", "Die Untersuchungen zeigten keinen Herzinfarkt.", "tabs\tand\nnewlines  and  spaces "]:
        assert tok.decode(tok.encode(text)) == text


def test_more_merges_mean_fewer_tokens():
    train, test = read("clinical_train"), read("clinical_test")
    sizes = [260, 300, 400, 512]
    counts = [len(ByteBPE.train(train, s).encode(test)) for s in sizes]
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1]


def test_training_is_deterministic():
    a, b = ByteBPE.train(TEXT, 300), ByteBPE.train(TEXT, 300)
    assert a.merges == b.merges


def test_merges_never_cross_a_chunk_border():
    tok = ByteBPE.train(TEXT, 300)
    assert all(b" " not in piece[1:] for piece in tok.vocab[256:])  # a space only sits at the start of a piece


def test_special_tokens_are_one_id_and_can_be_switched_off():
    tok = ByteBPE.train(TEXT, 300)
    tok.special = {"<|end|>": len(tok.vocab)}
    ids = tok.encode("the nurse<|end|>")
    assert ids[-1] == len(tok.vocab) and tok.decode(ids) == "the nurse<|end|>"
    plain = tok.encode("<|end|>", allow_special=False)
    assert len(plain) > 1 and len(tok.vocab) not in plain


def test_save_and_load(tmp_path):
    tok = ByteBPE.train(TEXT, 300)
    tok.save(tmp_path / "t.json")
    again = ByteBPE.load(tmp_path / "t.json")
    assert again.merges == tok.merges and again.encode("the nurse") == tok.encode("the nurse")


def test_unseen_word_still_encodes_with_bytes_only():
    tok = ByteBPE.train(read("clinical_train"), 512)
    pieces = tok.pieces(" rivaroxaban")
    assert len(pieces) == 10 and "".join(pieces) == " rivaroxaban"
