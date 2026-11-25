"""worked_examples.py - reproduces every small number calculated by hand in lesson 4.

What it does
    Recomputes, with plain Python only, each calculation we do on a calculator in the video:
    pair counts and merges on four words, cutting a new word with the merges, how UTF-8 turns a letter into bytes,
    tokens per word and bytes per token, the cost of a prompt at an EXAMPLE price, and a context-window budget.

What it needs
    Only the Python standard library.

What to expect
    Run:  python worked_examples.py
    Every line printed here is a number you can also get on a calculator. The saved output is in
    data/expected_output/worked_examples.txt.
"""
from tokenizer_lab import bpe_words as bw
from tokenizer_lab.bpe import ByteBPE
from tokenizer_lab.corpus import read

CORPUS = {"charted": 7, "dosed": 6, "ward": 8, "checked": 5}

# numbers measured in demo 7 with the o200k_base encoding (see data/expected_output/d07_tokens_and_money.txt) and used in the video
SYSTEM_TOKENS, CHAT_TOKENS, QUESTION_TOKENS = 61, 80, 18   # system prompt, chat so far, the question
NOTES_TOKENS, NOTES_COUNT = 765, 40                         # forty clinical notes cost 765 tokens together
EXAMPLE_PRICE_IN, EXAMPLE_PRICE_OUT = 2.0, 8.0     # EXAMPLE dollars per million tokens, chosen for easy arithmetic
GERMAN_FACTOR = 1.2                                  # about how many times more tokens the German letter needs with o200k_base (measured 1.18 in demo 7)


def pair_counts_by_hand():
    """Count every neighbouring pair on the four-word corpus, then glue the most frequent one."""
    print("== 1. Counting pairs on four words (word: how often it occurs)", CORPUS)
    words = bw.split_into_letters(CORPUS)
    counts = bw.count_pairs(words)
    for (a, b), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {a}+{b}: {n}")
    print("  e+d occurs in charted (7) + dosed (6) + checked (5) = 7 + 6 + 5 =", 7 + 6 + 5)
    print("  a+r occurs in charted (7) + ward (8) = 7 + 8 =", 7 + 8)
    print("  c+h occurs in charted (7) + checked (5) = 7 + 5 =", 7 + 5)


def five_merges():
    """Run five merges and show the vocabulary growing from 11 letters to 16 pieces."""
    print("== 2. Five merges")
    merges, steps = bw.train(CORPUS, 5)
    for i, step in enumerate(steps, 1):
        a, b = step["pair"]
        print(f"  merge {i}: {a} + {b} -> {a + b}   (count {step['count']})")
    vocab = bw.vocabulary(CORPUS, merges)
    print(f"  pieces: {len([v for v in vocab if len(v) == 1])} letters + {len(merges)} merges = {len(vocab)}")
    print("  a tie: after merge 3, w+ar and ar+d both occur 8 times; the pair met first (w+ar) wins")
    return merges


def cut_a_new_word(merges):
    """Replay the merges, in order, on words the corpus never contained."""
    print("== 3. Cutting new words with the same five merges")
    for word in ("warded", "waxed"):
        print(f"  {word}: {' + '.join(bw.encode_word(word, merges))}")
    print("  decode joins the pieces: ward + ed =", bw.decode_pieces(["ward", "ed"]))


def utf8_by_hand():
    """Write the letter u-umlaut as bytes: its number is 252, which does not fit in seven bits."""
    print("== 4. UTF-8 by hand for the letter ü")
    code = ord("ü")
    bits = format(code, "011b")
    first, second = "110" + bits[:5], "10" + bits[5:]
    print(f"  number of the letter: {code} = binary {bits}")
    print(f"  two bytes: {first} {second} = {int(first, 2):02x} {int(second, 2):02x}")
    print("  Python agrees:", "ü".encode("utf-8").hex(" "))


def tokens_per_word():
    """Tokens per word and bytes per token for the 512-piece tokenizer on the test text."""
    print("== 5. Tokens per word and bytes per token")
    train, test = read("clinical_train"), read("clinical_test")
    tok = ByteBPE.train(train, 512)
    n_tokens, n_words, n_bytes = len(tok.encode(test)), len(test.split()), len(test.encode("utf-8"))
    print(f"  {n_tokens} tokens / {n_words} words = {n_tokens / n_words:.2f} tokens per word")
    print(f"  {n_bytes} bytes / {n_tokens} tokens = {n_bytes / n_tokens:.2f} bytes per token")
    print(f"  letters only would need {n_bytes} tokens = {n_bytes / n_words:.2f} per word")


def money():
    """Cost of a day of requests at an EXAMPLE price: tokens x price per million."""
    print("== 6. Tokens and money (EXAMPLE price: $2 per million tokens in, $8 per million out)")
    requests, prompt, reply = 10_000, 1_500, 300
    cost_in = requests * prompt * EXAMPLE_PRICE_IN / 1e6
    cost_out = requests * reply * EXAMPLE_PRICE_OUT / 1e6
    print(f"  English: {requests:,} x {prompt:,} = {requests * prompt:,} tokens in -> ${cost_in:.2f}; {requests:,} x {reply} = {requests * reply:,} tokens out -> ${cost_out:.2f}; total ${cost_in + cost_out:.2f}")
    german = round(prompt * GERMAN_FACTOR)
    cost_de = requests * german * EXAMPLE_PRICE_IN / 1e6
    print(f"  German at {GERMAN_FACTOR} times the tokens: {german:,} per prompt -> ${cost_de:.2f} in; total ${cost_de + cost_out:.2f}, which is ${cost_de + cost_out - cost_in - cost_out:.2f} more per day")


def budget():
    """A context window is a shared budget: whatever the reply needs is taken off first."""
    print("== 7. A context window as a budget (window size is an EXAMPLE)")
    window, reserve = 8_000, 1_000
    left = window - reserve - SYSTEM_TOKENS - CHAT_TOKENS - QUESTION_TOKENS
    print(f"  {window:,} - {reserve:,} (reply) - {SYSTEM_TOKENS} (system prompt) - {CHAT_TOKENS} (chat so far) - {QUESTION_TOKENS} (question) = {left:,} tokens left for notes")
    per_note = NOTES_TOKENS / NOTES_COUNT
    print(f"  one note costs {NOTES_TOKENS} / {NOTES_COUNT} = {per_note:.3f} tokens on average")
    print(f"  {left:,} / {per_note:.3f} = {left / per_note:.1f}, so {int(left // per_note)} notes fit")


def main():
    pair_counts_by_hand()
    merges = five_merges()
    cut_a_new_word(merges)
    utf8_by_hand()
    tokens_per_word()
    money()
    budget()


if __name__ == "__main__":
    main()
