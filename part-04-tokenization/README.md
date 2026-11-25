# Video 4 - Tokenization

Video: [VIDEO LINK, to be added when the video is published]

**Learning goal:** see how a model reads text. A model receives numbers, never letters, so something has to cut every text into pieces and give each piece a number. We build that cutter ourselves, step by step, on a few small words first and then on a larger invented clinical text. Then we compare it with the real tools that people use, count what pieces cost in money and in space, and look at the odd mistakes that pieces can cause.

This lesson assumes basic coding skills and a little familiarity with machine learning. Every idea is introduced before it is used, with a small example you can repeat on a calculator.

## What the lesson covers

The video has five parts:

1. **Cutting text into pieces.** What a model really receives, and a first comparison of three ways to cut: by letter, by word and by pieces in between. Each way has a price in vocabulary size, in number of tokens and in words it has never seen.
2. **Gluing letters into pieces, by hand.** Start from single letters on four small words, count which neighbouring pair occurs most, glue it into one piece, and repeat. Five merges, every count on screen, then cutting a brand-new word with the saved merges and joining the pieces back.
3. **Every text in every language.** How a computer stores letters from any alphabet as bytes, how the cutter starts from those 256 bytes so that nothing is ever unknown, training it on the clinical text, and the real tools people use: tiktoken, Hugging Face tokenizers and SentencePiece. The special markers that turn a conversation into one long text close this part.
4. **What pieces cost.** The same letter in English, German and Python, German compound words, a price per million tokens, a context window as a budget you can calculate, and the quirks that follow: capitals, spaces, numbers, rare drug names and counting letters.
5. **The code and the lab.** The main files read side by side, the 25 tests, and an exercise: how does the vocabulary size change what a tokenizer does to a rare drug name?

The next lesson (lesson 5) gives every token a list of numbers, an embedding, and a place in the sentence.

## Folder map

| Path | What it holds |
|---|---|
| `code/tokenizer_lab/bpe_words.py` | Byte-pair encoding by hand on a tiny list of words: count pairs, merge the most frequent, repeat, then cut a new word with the saved merges |
| `code/tokenizer_lab/bpe.py` | Our own byte-level tokenizer: it starts from the 256 byte values, cuts text into chunks (a space stays with its word, digits in groups of three), learns merges from the training text, and encodes and decodes with special tokens |
| `code/tokenizer_lab/corpus.py` | Reads the invented text files and cuts them into words, lines and letters |
| `code/tokenizer_lab/realtok.py` | Loads the real tokenizers (OpenAI's two encodings, GPT-2, BERT, T5 and Qwen) behind one small interface: pieces, ids and count |
| `code/tokenizer_lab/chat_template.py` | Our own tiny chat template, and the real Qwen template rendered with Jinja |
| `code/tokenizer_lab/report.py` | Small helpers that save every demo's printed output and numbers next to the code |
| `code/demos/` | Ten short demos, `d01` to `d10`, one per idea in the video |
| `code/tests/` | 25 pytest tests |
| `code/make_corpus.py` | Writes the invented CareDesk clinical text (training and test files) and the short letter used in the video |
| `code/worked_examples.py` | Reproduces every small number calculated by hand in the video (plain Python, no download) |
| `code/run_all.py` | Runs all ten demos in order and saves their output |
| `code/pyproject.toml`, `code/uv.lock` | The project recipe and the exact versions used in the recorded run |
| `data/clinical_train.txt`, `data/clinical_test.txt`, `data/letters.txt` | The invented practice text and the short letter in English, German and Python form (fictional, written for practice only) |
| `data/expected_output/` | The real printed output of every command below, from the recorded run |
| `data/results/` | The numbers behind the demos as JSON, plus the trained 512-piece tokenizer |

## Run plan

| Mode | What | Cost |
|---|---|---|
| Any laptop or desktop with Python 3.12+ | every demo and test in this folder (the CPU is enough) | free |
| Needs the internet | demos 5 to 8 download up to six small tokenizer files the first time (about 15.5 MB in total, listed in `data/expected_output/downloads.txt`) and reuse the saved copies afterwards; the Qwen chat-template settings (a small text file) are fetched again on every run of demo 6 | free |
| Shown, not run | (this lesson has none: every command below runs for real) | - |

The recorded run used uv 0.9.7, CPython 3.13.9, tiktoken 0.14.0, tokenizers 0.23.3, sentencepiece 0.2.3, jinja2 3.1.6 and pytest 9.2.0 (see `data/expected_output/environment.txt`). Only the small tokenizer files are downloaded, never any model.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the `code/` folder:

```bash
cd code
uv sync                 # creates .venv and installs the exact versions from uv.lock
```

## Run it

```bash
uv run python -m demos.d01_three_ways          # one word cut three ways, and the trade-off on the test text
uv run python -m demos.d02_bpe_by_hand         # five merges on four words, every pair count shown
uv run python -m demos.d03_bytes_utf8          # how many bytes each letter takes, and why 256 pieces can write any text
uv run python -m demos.d04_train_bpe           # train our byte-level tokenizer (vocabulary 512), then encode and decode
uv run python -m demos.d05_real_tokenizers     # one sentence through six tokenizers (needs the internet on the first run)
uv run python -m demos.d06_special_and_chat    # special tokens, OpenAI's end-of-text marker and a Qwen chat template
uv run python -m demos.d07_tokens_and_money    # English, German and Python, German compounds, price per day, context-window budget
uv run python -m demos.d08_quirks              # capitals, spaces, numbers, rare drug names, counting letters
uv run python -m demos.d09_lab_vocab_size      # the lab: five vocabulary sizes and two rare drug names
uv run python -m demos.d10_sentencepiece       # SentencePiece: a Unigram and a BPE tokenizer on the same text
uv run pytest -v                               # 25 tests
uv run python worked_examples.py               # the hand calculations from the video
uv run python run_all.py                       # all ten demos in one go; saves outputs to data/
```

The demos finish within a few seconds on a laptop CPU; demo 5 takes longer on the first run because of the downloads (see `data/expected_output/run_all.txt` for the recorded times).

## What you should see

The files in `data/expected_output/` hold the complete output. Highlights from the recorded run:

```text
$ uv run python -m demos.d01_three_ways
the trade-off on the test text (2308 words):
  way                        vocabulary  tokens per word  unknown words
  letters                            52             5.77              0
  whole words (top 100)             100             1.00            254
  whole words (top 508)             508             1.00             19
  pieces (vocabulary 300)           300             3.62              0
  pieces (vocabulary 400)           400             2.21              0
  pieces (vocabulary 512)           512             1.80              0
```

```text
$ uv run python -m demos.d02_bpe_by_hand
merge 1: the most frequent pair is e + d  (count 18)
merge 2: the most frequent pair is a + r  (count 15)
...
```

```text
$ uv run python -m demos.d03_bytes_utf8
text           characters  bytes   bytes in hex
e                       1      1   65
ü                       1      2   c3 bc
€                       1      3   e2 82 ac
😀                       1      4   f0 9f 98 80
```

```text
$ uv run python -m demos.d05_real_tokenizers
sentence: The pharmacist filled the prescription for rivaroxaban 500 mg on 2026-03-14.
our BPE (512)      ->  27 tokens
gpt2               ->  22 tokens
cl100k_base        ->  22 tokens
o200k_base         ->  22 tokens
bert-base-uncased  ->  24 tokens
t5-small           ->  21 tokens
```

```text
$ uv run python -m demos.d07_tokens_and_money
  tokenizer                   English   German   Python   German / English
  our BPE (512)                   213      349      398               1.64
  gpt2                             92      180      195               1.96
  cl100k_base                      99      142      148               1.43
  o200k_base                       98      116      148               1.18
  bert-base-uncased                93      169      174               1.82
  t5-small                         99      121      187               1.22

  window of   8,000: 6,841 tokens left for notes -> about 357 notes fit
  window of 128,000: 126,841 tokens left for notes -> about 6,632 notes fit
```

```text
$ uv run python -m demos.d09_lab_vocab_size
vocabulary  merges  train tokens/word  test tokens/word  bytes per token  rivaroxaban  dapagliflozin
       260       4               5.29              5.34             1.08           12             14
       300      44               3.60              3.62             1.60           10             12
       400     144               2.16              2.21             2.61           10             11
       512     256               1.77              1.80             3.20           10             11
       695     439               1.44              1.49             3.87           10             11
```

```text
$ uv run pytest -v
============================== 25 passed in 0.34 s ==============================
```

Timings depend on your machine. The numbers are repeatable, because the training text is fixed and the training steps contain no random choice. A newer version of a tokenizer library can still change a piece or an id.

### What `worked_examples.py` reproduces

The video calculates small numbers by hand. `worked_examples.py` prints exactly those numbers using plain Python and needs no download:

```text
== 1. Counting pairs on four words (word: how often it occurs) {'charted': 7, 'dosed': 6, 'ward': 8, 'checked': 5}
  e+d: 18
  a+r: 15
  c+h: 12
  r+d: 8
  w+a: 8
  h+a: 7
  r+t: 7
  t+e: 7
  d+o: 6
  o+s: 6
  s+e: 6
  c+k: 5
  e+c: 5
  h+e: 5
  k+e: 5
  e+d occurs in charted (7) + dosed (6) + checked (5) = 7 + 6 + 5 = 18
  a+r occurs in charted (7) + ward (8) = 7 + 8 = 15
  c+h occurs in charted (7) + checked (5) = 7 + 5 = 12
== 2. Five merges
  merge 1: e + d -> ed   (count 18)
  merge 2: a + r -> ar   (count 15)
  merge 3: c + h -> ch   (count 12)
  merge 4: w + ar -> war   (count 8)
  merge 5: war + d -> ward   (count 8)
  pieces: 11 letters + 5 merges = 16
  a tie: after merge 3, w+ar and ar+d both occur 8 times; the pair met first (w+ar) wins
== 3. Cutting new words with the same five merges
  warded: ward + ed
  waxed: w + a + x + ed
  decode joins the pieces: ward + ed = warded
== 4. UTF-8 by hand for the letter ü
  number of the letter: 252 = binary 00011111100
  two bytes: 11000011 10111100 = c3 bc
  Python agrees: c3 bc
== 5. Tokens per word and bytes per token
  4158 tokens / 2308 words = 1.80 tokens per word
  13313 bytes / 4158 tokens = 3.20 bytes per token
  letters only would need 13313 tokens = 5.77 per word
== 6. Tokens and money (EXAMPLE price: $2 per million tokens in, $8 per million out)
  English: 10,000 x 1,500 = 15,000,000 tokens in -> $30.00; 10,000 x 300 = 3,000,000 tokens out -> $24.00; total $54.00
  German at 1.2 times the tokens: 1,800 per prompt -> $36.00 in; total $60.00, which is $6.00 more per day
== 7. A context window as a budget (window size is an EXAMPLE)
  8,000 - 1,000 (reply) - 61 (system prompt) - 80 (chat so far) - 18 (question) = 6,841 tokens left for notes
  one note costs 765 / 40 = 19.125 tokens on average
  6,841 / 19.125 = 357.7, so 357 notes fit
```

The same output is saved in `data/expected_output/worked_examples.txt`.

## Honest notes

- **Our tokenizer is small on purpose.** It learns 256 merges from about 23,000 words of invented text, so it needs 1.80 tokens per word on the test text. The real tools learned from far more text and use between 30,000 and 200,000 pieces.
- **The Unigram and BPE comparison in demo 10 is not like for like.** The Unigram model settled at a vocabulary of 260 pieces and the BPE model was trained with 300 pieces, so the larger count of tokens per word for Unigram (2.63 against 1.62) mostly reflects the smaller vocabulary, not a verdict on the method.
- **The German factor of 1.2 in the money example is a teaching rounding** of the o200k_base result (1.18). The other tokenizers in demo 7 give 1.43 (cl100k_base) and 1.96 (GPT-2), so the extra cost of German depends on the tokenizer.
- **Prices and window sizes are examples.** The $2 and $8 per million tokens and the windows of 8,000 and 128,000 tokens teach the arithmetic; look up the current price and window of any model you plan to use.
- **The lab uses two words the tokenizer never saw.** Rivaroxaban and dapagliflozin are missing from `data/clinical_train.txt` (each appears once in the test text), so every piece in them comes from merges learned on other words. Another training text gives other numbers.
- **Real tokenizers change.** Demos 5 to 8 download the tokenizer files listed in `data/expected_output/downloads.txt` with their checksums; a later release of a library or a file can change a piece or an id.
