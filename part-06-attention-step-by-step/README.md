# Video 6 - Attention, step by step

Video: [VIDEO LINK, to be added when the video is published]

**Learning goal:** understand how every token in a sentence decides whom to listen to, and calculate it yourself: a score for each pair, a small fix for big numbers, shares that add up to one hundred percent, and a weighted blend of the values. Then see the same steps in PyTorch, in a real model (GPT-2), and in a rewrite that saves memory.

This lesson assumes basic coding skills and a little familiarity with machine learning. Every idea is introduced before it is used, with a small example you can repeat on a calculator. The same three-token example runs through the whole lesson, in the video, in `worked_examples.py` and in the tests.

## What the lesson covers

The video has five parts:

1. **Whom should a word listen to?** The sentence "the animal stayed home because it was tired" and the same sentence with "raining" at the end: one small word, two meanings. A meeting of colleagues where everyone decides whom to listen to, and how listening turns into a blend.
2. **Question, label and content.** A library with a question, a spine label and the content of a book: the query, the key and the value. A calculator example turns one token vector `[1, 2]` into the query `[3, 2]`, the key `[2, 1]` and the value `[2, 4]`, and shows where the three small tables come from.
3. **Attention by hand.** One question against three keys: the scores `[1, 0, 1]`, the division by the square root of d, softmax (shares of 40, 20 and 40 percent), and the blend `[1.2033, 0.7967]`. A think slide asks which key gets the smallest share. Then every token asks at once and the 3 x 3 table appears, followed by the whole recipe in one line.
4. **The rule for writing text, and many readers.** Switching off the future with minus infinity before softmax, the peek test with a huge vector, a think slide on counting the cells, and several heads that read their own slices, with the parameter count.
5. **Maps, cost, the code and the lab.** How to read an attention map, a real map from GPT-2, a think slide on which column lights up, the habits of real heads, what attention costs when the text grows, the tiled rewrite that saves memory, a tour of the main files, the tests, and a lab question with a mirror query.

The next lesson (lesson 7) wraps attention inside the Transformer block.

## Folder map

| Path | What it holds |
|---|---|
| `code/attention_lab/numpy_attention.py` | Softmax, attention for one sequence, the causal mask, several heads and the tiled version, all with plain NumPy |
| `code/attention_lab/attention.py` | The same attention in PyTorch for a batch of sequences and many heads at once |
| `code/attention_lab/multihead.py` | Multi-head attention as a small PyTorch module: four learned tables, a split into heads, a mix at the end |
| `code/attention_lab/gpt2_maps.py` | Runs the small public model GPT-2 and returns the real attention weights for any text |
| `code/attention_lab/maps.py` | Draws a table of weights as a picture, with the numbers inside the cells |
| `code/attention_lab/report.py` | Small helpers that save every demo's printed output and numbers next to the code |
| `code/demos/` | Nine short demos, `d01` to `d09`, one per idea in the video |
| `code/tests/` | 10 pytest tests (7 need only NumPy, 3 need PyTorch) |
| `code/worked_examples.py` | Reproduces every small number calculated by hand in the video (NumPy only, no download) |
| `code/run_all.py` | Runs all nine demos in order and saves their output |
| `code/pyproject.toml`, `code/uv.lock` | The project recipe and the exact versions used in the recorded run |
| `data/expected_output/` | The real printed output of every command below, from the recorded run |
| `data/results/` | The numbers behind the demos as JSON (the four attention-map pictures are drawn on your computer when you run demos 4 and 5) |

## Run plan

| Mode | What | Cost |
|---|---|---|
| Any laptop or desktop with Python 3.12+ | every demo and test in this folder (the CPU is enough) | free |
| Needs the internet | demos 5 and 6 download the small public model GPT-2 (about 550 MB) the first time and reuse the saved copy afterwards | free |
| Runs offline | demos 1, 2, 3, 4, 7, 8 and 9, the tests and `worked_examples.py` need no download (the tests of the PyTorch code need PyTorch and skip themselves when it is missing) | free |
| Shown, not run | (this lesson has none: every command below runs for real) | - |

The recorded run used uv 0.9.7, CPython 3.13.9, NumPy 2.5.5, PyTorch 2.15.0, transformers 5.24.1, huggingface_hub 2.5.0, matplotlib 3.11.2 and pytest 9.2.0 (see `data/expected_output/environment.txt`).

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the `code/` folder:

```bash
cd code
uv sync                 # creates .venv and installs the exact versions from uv.lock
```

## Run it

```bash
uv run python worked_examples.py                # the hand calculations from the video (NumPy only, starts here)
uv run python -m demos.d01_attention_from_scratch   # attention in PyTorch on our three tokens, compared with the NumPy numbers
uv run python -m demos.d02_check_builtin        # our attention() against PyTorch's built-in kernel, with and without the mask
uv run python -m demos.d03_multihead_module     # our multi-head module against torch.nn.MultiheadAttention, and the parameter count
uv run python -m demos.d04_toy_maps             # the 3 x 3 maps of the hand example, open and masked, as pictures
uv run python -m demos.d05_gpt2_map             # a real attention map: "it" in "the animal stayed home because it was tired" (GPT-2)
uv run python -m demos.d06_head_roles           # the habits of all 144 heads of GPT-2 on ten plain sentences
uv run python -m demos.d07_cost_curve           # time and score-table size from 128 to 4,096 tokens
uv run python -m demos.d08_tiled_check          # the tiled version against plain attention (NumPy only)
uv run python -m demos.d09_lab_question         # the lab question: the mirror query and the square root of d
uv run pytest -v                                # 10 tests
uv run python run_all.py                        # all nine demos in one go; saves outputs to data/
```

The offline demos finish within seconds on a laptop CPU. The first run of demos 5 and 6 takes longer because of the GPT-2 download (see `data/expected_output/downloads.txt` for the exact files and their checksums). The timings of demo 7 depend on your computer; the table sizes stay exact.

## What you should see

The files in `data/expected_output/` hold the complete output. Highlights from the recorded run:

```text
$ uv run python -m demos.d01_attention_from_scratch
one query [1, 0]: weights [0.4011, 0.1978, 0.4011]  output [1.2033, 0.7967]
  same numbers as NumPy: True

full table (every token asks):
  row 1: [0.4011, 0.1978, 0.4011]  sum 1.0000
  row 2: [0.1978, 0.4011, 0.4011]  sum 1.0000
  row 3: [0.2483, 0.2483, 0.5035]  sum 1.0000

causal table (the future is switched off):
  row 1: [1.0, 0.0, 0.0]  sum 1.0000
  row 2: [0.3302, 0.6698, 0.0]  sum 1.0000
  row 3: [0.2483, 0.2483, 0.5035]  sum 1.0000

largest weight above the diagonal in the causal table: 0.0
largest distance from 1 of any row sum: 0.0e+00
```

```text
$ uv run python -m demos.d02_check_builtin
causal=False: biggest difference from the built-in kernel 3.6e-07   rows add up to 1 within 1.2e-07   largest weight above the diagonal 0.8
             same answer within 1e-5: True
causal=True : biggest difference from the built-in kernel 2.4e-07   rows add up to 1 within 1.2e-07   largest weight above the diagonal 0.0
             same answer within 1e-5: True

shapes: q (2, 4, 16, 8) -> output (2, 4, 16, 8), weights (2, 4, 16, 16)
```

```text
$ uv run python -m demos.d03_multihead_module
input (2, 10, 32) -> output (2, 10, 32), weights (2, 4, 10, 10) (batch, heads, tokens, tokens)
biggest difference from torch.nn.MultiheadAttention: output 1.2e-07, weights 1.2e-07
same answer within 1e-5: True

parameters for d_model = 32 (four matrices of 32 x 32 plus four bias lists of 32):
  1 head(s) of 32 numbers each: 4,224 parameters
  2 head(s) of 16 numbers each: 4,224 parameters
  4 head(s) of 8 numbers each: 4,224 parameters
  8 head(s) of 4 numbers each: 4,224 parameters
  4 x (32 x 32 + 32) = 4,224

the heads look at different things even at the start (random weights): row 10 of each head, first 5 columns:
  head 1: [0.109, 0.117, 0.082, 0.141, 0.104]
  head 2: [0.097, 0.071, 0.092, 0.108, 0.068]
  head 3: [0.107, 0.127, 0.11, 0.09, 0.094]
  head 4: [0.088, 0.098, 0.126, 0.075, 0.094]
```

```text
$ uv run python -m demos.d04_toy_maps
map_toy_full: rows listen, columns are listened to
  token 1: [0.4, 0.2, 0.4]  sum 1.00
  token 2: [0.2, 0.4, 0.4]  sum 1.00
  token 3: [0.25, 0.25, 0.5]  sum 1.00
  picture saved: True
map_toy_causal: rows listen, columns are listened to
  token 1: [1.0, 0.0, 0.0]  sum 1.00
  token 2: [0.33, 0.67, 0.0]  sum 1.00
  token 3: [0.25, 0.25, 0.5]  sum 1.00
  picture saved: True
```

```text
$ uv run python -m demos.d05_gpt2_map
sentence: 'the animal stayed home because it was tired'
tokens (8): ['the', 'animal', 'stayed', 'home', 'because', 'it', 'was', 'tired']
weights tensor: 12 layers x 12 heads x 8 x 8 = 9,216 numbers
largest weight above the diagonal in any head: 0.0
largest distance from 1 of any row sum: 2.4e-07

the row of 'it' (position 6) listening to 'animal' (position 2), over all 144 heads:
  average weight 0.049; heads with weight above 0.20: 8; above 0.10: 14
  the five heads that listen most to 'animal' (layer, head: weight):
    layer 5, head 4: 0.696
    layer 6, head 11: 0.562
    layer 10, head 1: 0.357
    layer 1, head 7: 0.324
    layer 2, head 7: 0.294
  for comparison, the row of 'it' on the first token 'the': average 0.603; heads with weight above 0.5: 92 of 144

row of 'it' in the best head (layer 5, head 4):
       the 0.230
    animal 0.696
    stayed 0.023
      home 0.021
   because 0.010
        it 0.020
  biggest weight on a word other than the first token: animal (0.696)

row of 'it' averaged over all 144 heads:
       the 0.603
    animal 0.049
    stayed 0.067
      home 0.044
   because 0.140
        it 0.097

pictures saved: True
```

```text
$ uv run python -m demos.d06_head_roles
10 sentences, 12 layers x 12 heads = 144 heads; each number is an average over all tokens after the first

heads that listen most to the previous token:
  layer 5, head 12: 1.00
  layer 3, head 3: 0.61
  layer 4, head 3: 0.49
  heads with average weight above 0.5 on the previous token: 2 of 144

heads that listen most to the first token:
  layer 6, head 2: 1.00
  layer 8, head 3: 0.99
  layer 7, head 10: 0.98
  heads with average weight above 0.5 on the first token: 107 of 144

heads that listen most to their own token:
  layer 1, head 2: 0.94
  layer 1, head 4: 0.85
  layer 1, head 6: 0.85
  heads with average weight above 0.5 on their own token: 4 of 144

all heads together put on average 0.20 of their weight on the previous token, 0.65 on the first token, 0.09 on themselves

the strongest head of each habit, on 'the animal stayed home because it was tired': previous token = layer 5, head 12, first token = layer 6, head 2, own token = layer 1, head 2
```

```text
$ uv run python -m demos.d07_cost_curve
torch threads: 10; one head, 64 numbers per head, no mask, float32
     n   plain ms   built-in ms   score table (n x n x 4 bytes)   plain time vs previous row
   128       0.06          0.04             0.1 MB                         -
   256       0.12          0.05             0.3 MB                      1.9x
   512       0.16          0.06             1.0 MB                      1.4x
  1024       0.52          0.26             4.2 MB                      3.3x
  2048       1.29          0.66            16.8 MB                      2.5x
  4096       5.30          2.22            67.1 MB                      4.1x

the score table at n = 4,096 holds 16,777,216 numbers; at n = 128 it holds 16,384: 1,024 times more for 32 times more text
```

```text
$ uv run python -m demos.d08_tiled_check
n = 1024 tokens, 64 numbers per head; plain attention holds 1,048,576 scores at once
  causal=False block  64: biggest difference 8.6e-16   largest piece held 65,536 scores (6.2 percent of the full table)
  causal=False block 128: biggest difference 7.8e-16   largest piece held 131,072 scores (12.5 percent of the full table)
  causal=False block 256: biggest difference 8.0e-16   largest piece held 262,144 scores (25.0 percent of the full table)
  causal=True  block  64: biggest difference 7.2e-16   largest piece held 65,536 scores (6.2 percent of the full table)
  causal=True  block 128: biggest difference 8.3e-16   largest piece held 131,072 scores (12.5 percent of the full table)
  causal=True  block 256: biggest difference 8.3e-16   largest piece held 262,144 scores (25.0 percent of the full table)
```

```text
$ uv run python -m demos.d09_lab_question
query [1.0, 0.0]: scores [1, 0, 1]  weights [0.4011, 0.1978, 0.4011]  output [1.2033, 0.7967]
query [0.0, 1.0]: scores [0, 1, 1]  weights [0.1978, 0.4011, 0.4011]  output [0.7967, 1.2033]

d = 64, eight keys: raw scores [-10.8, -3.7, 8.7, 15.2, -0.4, -9.8, -4.0, -0.6]
  softmax of the raw scores:    biggest weight 0.999
  softmax of scores / sqrt(64):  biggest weight 0.502
```

```text
$ uv run pytest -v
============================= test session starts ==============================
collecting ... collected 10 items

tests/test_numpy_attention.py::test_softmax_rows_add_up_to_one_and_survive_large_scores PASSED [ 10%]
tests/test_numpy_attention.py::test_hand_example_weights_and_output PASSED [ 20%]
tests/test_numpy_attention.py::test_causal_table_is_lower_triangular_and_rows_add_up_to_one PASSED [ 30%]
tests/test_numpy_attention.py::test_causal_output_ignores_the_future_and_open_output_does_not PASSED [ 40%]
tests/test_numpy_attention.py::test_tiled_equals_plain_with_and_without_the_mask PASSED [ 50%]
tests/test_numpy_attention.py::test_multi_head_shapes_and_two_different_maps PASSED [ 60%]
tests/test_numpy_attention.py::test_worked_examples_print_the_video_numbers PASSED [ 70%]
tests/test_torch_attention.py::test_matches_the_built_in_kernel_with_and_without_the_mask PASSED [ 80%]
tests/test_torch_attention.py::test_causal_weights_are_zero_above_the_diagonal PASSED [ 90%]
tests/test_torch_attention.py::test_multihead_matches_torch_module_and_parameter_count_is_head_independent PASSED [100%]

============================== 10 passed in 0.59s ==============================
[exit 0]
```

Timings depend on your machine. The numbers repeat on the same versions, because the toy example is fixed, the random inputs use fixed seeds, and GPT-2 gives the same weights for the same text. A newer release of a library, or of the downloaded model files, can change the last digits.

### What `worked_examples.py` reproduces

The video calculates small numbers by hand. `worked_examples.py` prints exactly those numbers using NumPy only and needs no download:

```text
$ uv run python worked_examples.py
== 1. Where query, key and value come from (x times three tables)
  token vector x = [1, 2]
  query = x times table W_q = [1*1 + 2*1, 1*0 + 2*1] = [3, 2]
  key   = x times table W_k = [1*0 + 2*1, 1*1 + 2*0] = [2, 1]
  value = x times table W_v = [1*2 + 2*0, 1*0 + 2*2] = [2, 4]
  three tables of 2 x 2 = 12 learned numbers; GPT-2 small uses 768 x 768 per table = 589,824 numbers each
== 2. One question, three labels (d = 2)
  query q = [1, 0]; keys k1 = [1, 0], k2 = [0, 1], k3 = [1, 1]; values v1 = [2, 0], v2 = [0, 2], v3 = [1, 1]
  scores q.k = [1*1 + 0*0, 1*0 + 0*1, 1*1 + 0*1] = [1, 0, 1]
  divide by sqrt(2) = 1.4142: [0.7071, 0.0, 0.7071]
  exp of each: [2.0281, 1.0, 2.0281]   sum = 5.0562
  weights = exp / sum = [0.4011, 0.1978, 0.4011]  -> [40, 20, 40] percent, total 100 percent
  weighted sum = 0.4011*[2, 0] + 0.1978*[0, 2] + 0.4011*[1, 1] = [1.2033, 0.7967]
  with the rounded weights 0.40, 0.20, 0.40: [1.2, 0.8]   (the same to two decimals: [1.2, 0.8])

== 3. Why divide by the square root of d
  random vectors with ordinary-sized numbers (mean 0, spread 1), 20,000 pairs per row:
      d   spread of q.k   after dividing
      4            2.00             1.00
     16            4.01             1.00
     64            7.96             1.00
    256           16.06             1.00
  d = 64, eight keys, 10,000 random rows: the biggest weight in a row averages 0.87 without the division and 0.36 with it
  rows whose biggest weight is above 0.95: 53 percent without the division, 0 percent with it
  one example row, raw scores [5.4, 15.9, -4.0, 6.1, 4.9, 0.7, 7.7, -7.1]
    softmax of the raw scores    : biggest weight 1.000
    softmax after / sqrt(64) = 8 : biggest weight 0.405   scaled scores [0.68, 1.98, -0.5, 0.77, 0.61, 0.09, 0.96, -0.88]

== 4. Every token asks: the full 3 x 3 table (queries = keys = [[1, 0], [0, 1], [1, 1]])
  scores q.k: [[1, 0, 1], [0, 1, 1], [1, 1, 2]]
  row 1: weights [0.4011, 0.1978, 0.4011]  = [40, 20, 40] percent  (sum 1.00)
  row 2: weights [0.1978, 0.4011, 0.4011]  = [20, 40, 40] percent  (sum 1.00)
  row 3: weights [0.2483, 0.2483, 0.5035]  = [25, 25, 50] percent  (sum 1.00)
  outputs: [[1.203, 0.797], [0.797, 1.203], [1.0, 1.0]]

== 5. The causal mask (the upper triangle is switched off, then every row is shared out again)
  switched off (True = future): [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
  row 1 sees token 1 only: weights [1.0, 0.0, 0.0]
  row 2 sees tokens 1 and 2: exp [1.0, 2.0281], sum 3.0281, weights [0.3302, 0.6698]
  row 3 sees all three: weights [0.2483, 0.2483, 0.5035]  (nothing to switch off)
  all rows add up to 1: True
  replace token 3 by [9, 9] and look at the outputs of tokens 1 and 2:
    with the mask    : tokens 1 and 2 keep their outputs: True
    without the mask : token 1 changes from [1.203, 0.797] to [8.96, 8.957]  (it peeked at the future)

== 6. Two heads (3 tokens, d_model = 4, each head works on 2 numbers)
  x = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]]   (rows are tokens)
  head 1 reads columns 1-2, head 2 reads columns 3-4; the weight matrices here are the identity so the split is easy to see
  head 1 weights: [[0.4, 0.2, 0.4], [0.2, 0.4, 0.4], [0.25, 0.25, 0.5]]
  head 2 weights: [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.33, 0.33, 0.33]]
  head 1 row 3 = [0.25, 0.25, 0.5]  versus head 2 row 3 = [0.33, 0.33, 0.33]  (different readers, different focus)
  glued together and mixed by w_o: output shape (3, 4), first row [0.802, 0.599, 0.503, 0.248]
  parameters: four weight matrices of 4 x 4 = 64; the same for one head or many heads (the split costs nothing extra)

== 7. The n x n score table
  n =   1,000: 1,000 x 1,000 = 1,000,000 scores per head per layer; at 2 bytes each that is 2 MB
  n = 100,000: 100,000 x 100,000 = 10,000,000,000 scores per head per layer; at 2 bytes each that is 20 GB
  ratio: 10,000 times more entries for 100 times more text
  with 32 heads in one layer at n = 100,000 and 2 bytes: 640 GB

== 8. The running softmax (what the tiled version does for one row)
  scores [1.0, 0.0, 1.0, 2.0], values [2.0, 0.0, 1.0, 3.0]
  all at once: weights [0.1966, 0.0723, 0.1966, 0.5344], output 2.1932
  block 1: scores [1.0, 0.0]  new maximum m = 1  nothing to shrink yet  running sum l = 1.3679  running weighted sum = 2.0000
  block 2: scores [1.0, 2.0]  new maximum m = 2  shrink the old totals by exp(1 - 2) = 0.3679  running sum l = 1.8711  running weighted sum = 4.1036
  at the end: output = weighted sum / l = 4.1036 / 1.8711 = 2.1932  (the same as all at once)
[exit 0]
```

The same output is saved in `data/expected_output/worked_examples.txt`.

### Downloaded files (with checksums)

```text
model files used (source, file, bytes, sha256):
huggingface.co/gpt2  config.json  665  0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4
huggingface.co/gpt2  generation_config.json  124  ed0b32ac72c0f5f44a719abb2d7786ea5146c871f83717b7f2018065954de02b
huggingface.co/gpt2  merges.txt  456318  1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5
huggingface.co/gpt2  model.safetensors  548105171  248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707
huggingface.co/gpt2  tokenizer_config.json  26  5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f
huggingface.co/gpt2  tokenizer.json  1355256  8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6
huggingface.co/gpt2  vocab.json  1042301  196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783
```

## Honest notes

- **The main example is tiny on purpose.** Three tokens with two numbers each let you check every step on a calculator. GPT-2 small uses 768 numbers per token and 12 heads of 64 numbers; the steps are the same, and only the sizes grow.
- **Rounded numbers can differ in the last digit.** The video and the printout show four decimals. Dividing the two rounded totals of the tiled example (4.1036 by 1.8711) gives 2.1931 on a calculator, while the unrounded totals give 2.1932, the same as plain attention. Both values agree to three decimals.
- **The causal mask writes minus infinity before softmax.** A score of zero would be a weaker switch: in our hand example a zero score still earned a share of 20 percent. The peek test (replace token three by `[9, 9]`) shows that the masked outputs of tokens one and two stay the same, while the open table moves token one from `[1.203, 0.797]` to `[8.96, 8.957]`.
- **One sentence, one model.** The map of "the animal stayed home because it was tired" comes from GPT-2 small. Layer 5, head 4 gives the strongest link from "it" to "animal" (0.696). Averaged over all 144 heads the row of "it" puts about 60 percent on the first word and only about 5 percent on "animal", so one head tells a clearer story than the average. A map is a strong clue about what a head reads, and later lessons add further tests before anything is called an explanation.
- **Head habits come from ten plain sentences.** The averages in demo 6 (previous token, first token, own token) describe those ten sentences only. Many heads rest much of their weight on the first token, a habit often called an attention sink.
- **Timings differ from computer to computer.** The cost curve of demo 7 ran on the CPU of one computer, with 10 threads: the time grows by about 88 times from 128 to 4,096 tokens while the score table grows by 1,024 times, because small jobs are dominated by fixed overheads. The built-in kernel finished in less than half the time at 4,096 tokens. The table sizes stay exact on every computer.
- **The tiled version in `numpy_attention.py` is the idea, not a fast kernel.** It follows the running-maximum and running-sum recipe of FlashAttention (Dao and others, 2022) in plain NumPy to show that the answer stays exact. Real kernels run on the GPU with careful use of its memory.
- **GPT-2 is a public model with its own licence.** The files are downloaded from its Hugging Face page when you run demos 5 and 6 and are never part of this repository. Check the licence before you reuse a model in your own project.
- **Later lessons return to attention.** Lesson 7 wraps it in the Transformer block, lesson 8 adds the K V cache, and lesson 38 returns to the memory cost when many users share one server.
