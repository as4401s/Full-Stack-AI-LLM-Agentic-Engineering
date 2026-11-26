# Video 5 - Embeddings and positions

Video: [VIDEO LINK, to be added when the video is published]

**Learning goal:** understand how a model turns a token id into a list of numbers that carries meaning, how it measures how close two meanings are, and how it knows the order of the words. We look up rows in an embedding table, train a small table of our own on an invented clinical text, draw it as a map, measure closeness with the dot product and the cosine, add position codes, turn arrows with RoPE, and compare our toy with the tables of real models.

This lesson assumes basic coding skills and a little familiarity with machine learning. Every idea is introduced before it is used, with a small example you can repeat on a calculator.

## What the lesson covers

The video has five parts:

1. **Numbers for tokens.** A search for "kidney failure" that finds "renal insufficiency", the embedding table as a wall of numbered drawers, a tiny table of six tokens with three numbers each, how only the rows that were used receive a learning signal, and the size of real tables.
2. **Meaning as a place.** We train a small table on 3,000 invented clinical sentences, squash its 8 numbers per word into 2 with PCA, and read the map. Then the famous word arithmetic (king - man + woman) on our map and on public GloVe vectors, with three honest warnings, and the table rows of a real model.
3. **Measuring closeness.** Arrows with length and direction, the dot product, the length of an arrow, the cosine of [1, 2, 0] and [2, 1, 0] (4 / 5 = 0.8), a think slide on the cosine of a vector with itself, and the cosine on the table our model learned.
4. **Knowing the order.** Why a model that adds up word vectors cannot tell "dog bites man" from "man bites dog", sinusoidal position codes (turning clock hands), learned position tables, RoPE (turning each arrow by an angle that grows with the position), a think slide on relative position, many speeds per vector, and two ways to stretch a context window.
5. **Sentences, the code and the lab.** Token vectors versus sentence vectors (they return in lesson 17), the code side by side with the tests, and an exercise: how many numbers per word does the map need?

The next lesson (lesson 6) opens the Transformer block and shows attention, step by step.

## Folder map

| Path | What it holds |
|---|---|
| `code/embedding_lab/skipgram.py` | Our own embedding table with PyTorch's `nn.Embedding`: it learns word vectors by guessing which other word shares a sentence |
| `code/embedding_lab/vectors.py` | Dot product, length, cosine, distance and nearest neighbours with plain NumPy |
| `code/embedding_lab/pca.py` | PCA through the singular value decomposition: squashes 8 numbers per word into 2, and checks that related words stay together |
| `code/embedding_lab/positions.py` | The sinusoidal position table and one attention head in plain NumPy, to show that attention alone sees a bag of tokens |
| `code/embedding_lab/rope.py` | Rotary position embeddings (RoPE) with NumPy: `rotate`, the turning speeds, and two ways to stretch a window |
| `code/embedding_lab/realmodels.py` | Loads the small public models used in the lesson (MiniLM, GloVe, and the settings files of GPT-2 and Qwen) and reads their tables |
| `code/embedding_lab/corpus.py` | Reads the invented sentences, builds the vocabulary and the neighbour pairs |
| `code/embedding_lab/mapmaker.py` | Trains our word map once per run and shares it between demos 4, 5, 6 and 12 |
| `code/embedding_lab/report.py` | Small helpers that save every demo's printed output and numbers next to the code |
| `code/demos/` | Thirteen short demos, `d01` to `d13`, one per idea in the video |
| `code/tests/` | 31 pytest tests |
| `code/make_corpus.py` | Writes the invented CareDesk clinical sentences (training and test files) from fixed templates and a fixed seed |
| `code/worked_examples.py` | Reproduces every small number calculated by hand in the video (NumPy only, no download) |
| `code/run_all.py` | Runs all thirteen demos in order and saves their output |
| `code/pyproject.toml`, `code/uv.lock` | The project recipe and the exact versions used in the recorded run |
| `data/clinical_words_train.txt`, `data/clinical_words_test.txt`, `data/word_groups.json` | The invented practice sentences (3,000 for training, 300 kept aside as an exam) and the body system and kind of each word (fictional, written for practice only) |
| `data/expected_output/` | The real printed output of every command below, from the recorded run |
| `data/results/` | The numbers behind the demos as JSON (the map picture, `d04_map.png`, is drawn on your computer when you run demo 4) |

The syllabus calls the first code file `embedding.py`. In this folder the same job is shared by three small files, `skipgram.py` (the table and its training), `vectors.py` (closeness) and `pca.py` (the map), so each file holds one idea.

## Run plan

| Mode | What | Cost |
|---|---|---|
| Any laptop or desktop with Python 3.12+ | every demo and test in this folder (the CPU is enough) | free |
| Needs the internet | demos 1, 3, 5, 7, 10, 11 and 13 download small public files the first time and reuse the saved copies afterwards: the MiniLM model (about 91 MB), the GloVe vectors (about 101 MB) and three small settings files (listed with checksums in `data/expected_output/downloads.txt`) | free |
| Runs offline | demos 2, 4, 6, 8, 9 and 12, the tests and `worked_examples.py` need no download (the tests of the training code need PyTorch and skip themselves when it is missing) | free |
| Shown, not run | (this lesson has none: every command below runs for real) | - |

The recorded run used uv 0.9.7, CPython 3.13.9, NumPy 2.5.5, PyTorch 2.15.0, transformers 5.24.1, huggingface_hub 2.4.0, gensim 4.4.0, matplotlib 3.11.2 and pytest 9.2.0 (see `data/expected_output/environment.txt`). Only the Qwen settings file is read from Qwen's repository; its model weights are never downloaded.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the `code/` folder:

```bash
cd code
uv sync                 # creates .venv and installs the exact versions from uv.lock
```

## Run it

```bash
uv run python -m demos.d01_search_hook         # the hook: keyword search against search by meaning (MiniLM)
uv run python -m demos.d02_table_lookup        # a tiny nn.Embedding: row 3 is the word kidney; only that row gets a learning signal
uv run python -m demos.d03_real_tables         # token ids, table rows and table sizes in MiniLM, GPT-2 and Qwen
uv run python -m demos.d04_train_map           # train our own table (8 numbers per word), loss curve, neighbours, the PCA map
uv run python -m demos.d05_analogies           # word arithmetic on our map and on GloVe, with the caveats
uv run python -m demos.d06_dot_cosine          # dot product and cosine on [1, 2, 0] and [2, 1, 0], then on our table
uv run python -m demos.d07_sentence_vs_token   # "bank" in two sentences; sentence vectors against a plain average of table rows
uv run python -m demos.d08_bag_of_tokens       # "dog bites man" against "man bites dog", without and with position codes
uv run python -m demos.d09_sinusoidal          # the sinusoidal table, how fast each pair turns, dot product against distance
uv run python -m demos.d10_learned_positions   # the learned position table of MiniLM and the size of GPT-2's
uv run python -m demos.d11_rope                # rotate a query and a key, score against distance, two ways to stretch a window
uv run python -m demos.d12_lab_dimension       # the lab: 2, 4, 8, 16 and 32 numbers per word
uv run python -m demos.d13_rope_vs_qwen        # our rope.rotate against Qwen2.5's own rotary code
uv run pytest -v                               # 31 tests
uv run python worked_examples.py               # the hand calculations from the video
uv run python run_all.py                       # all thirteen demos in one go; saves outputs to data/
```

The offline demos finish within seconds on a laptop CPU. The first run of demo 1 takes longer because of the model download, and demo 5 because of the GloVe download (see `data/expected_output/run_all.txt` for the recorded times).

## What you should see

The files in `data/expected_output/` hold the complete output. Highlights from the recorded run:

```text
$ uv run python -m demos.d01_search_hook
keyword search (a sentence must share a whole word with the query):
  [3] shares ['failure', 'kidney']: Chronic kidney failure requires regular dialysis sessions.
  [5] shares ['failure']: Heart failure symptoms include swollen ankles and tiredness.
  -> 2 of 8 sentences found; missed: [0, 1, 2] and more

embedding search (cosine between the query vector and each sentence vector):
  1. cosine 0.66  [3] Chronic kidney failure requires regular dialysis sessions.
  2. cosine 0.65  [2] Creatinine rose again, which suggests the kidneys are failing.
  3. cosine 0.59  [0] Renal insufficiency was noted at the follow-up visit.
  4. cosine 0.56  [7] Dialysis replaces the work of the kidneys when they stop working.
  5. cosine 0.48  [5] Heart failure symptoms include swollen ankles and tiredness.
  6. cosine 0.26  [1] The patient reports pain in the left knee after a fall.
  7. cosine 0.24  [6] A broken wrist needs a cast for six weeks.
  8. cosine 0.07  [4] The flight to Berlin leaves at nine in the morning.

cosine("kidney failure", "renal insufficiency") = 0.70
cosine("kidney failure", "knee surgery")        = 0.25
```

```text
$ uv run python -m demos.d03_real_tables
MiniLM token table: 30,522 rows x 384 numbers = 11,720,448 numbers
MiniLM position table: 512 rows x 384 numbers = 196,608 numbers

'kidney': ids [14234] -> pieces ['kidney']
'renal': ids [25125] -> pieces ['renal']
'insufficiency': ids [16021, 16093, 8873, 29125] -> pieces ['ins', '##uf', '##fi', '##ciency']
'failure': ids [4945] -> pieces ['failure']
'knee': ids [6181] -> pieces ['knee']
'the': ids [1996] -> pieces ['the']

cosine between table rows (no context involved yet, just the lookup rows):
  cosine(kidney, renal) = 0.75
  cosine(kidney, knee) = 0.24
  cosine(kidney, the) = 0.06
```

```text
$ uv run python -m demos.d04_train_map
vocabulary: 89 words, 388,972 (word, neighbour) pairs, 8 numbers per word -> a table of 712 numbers
random guessing among 89 words costs ln(89) = 4.489 per pair

loss after each epoch (average over all pairs):
  epoch  1: 3.8396
  epoch  2: 3.5863
  epoch  5: 3.4695
  epoch 10: 3.4484
  epoch 15: 3.4434
  epoch 20: 3.4392
train loss 3.4377, loss on 300 unseen sentences 3.4475

60 content words in 6 body systems; nearest neighbour is from the same system for 100% of them

  closest to 'kidney': urinalysis 0.97, sevelamer 0.97, furosemide 0.97, creatinine 0.97
  closest to 'heart': bisoprolol 0.97, ramipril 0.97, infarction 0.96, troponin 0.96
  closest to 'ramipril': bisoprolol 1.00, arrhythmia 1.00, infarction 1.00, angina 1.00
  closest to 'nephritis': uraemia 1.00, cystitis 1.00, furosemide 1.00, sevelamer 1.00
```

```text
$ uv run python -m demos.d05_analogies
our own map: specialist(A) - organ(A) + organ(B) should give specialist(B)
  orthopaedist - bone + brain -> neurologist (0.98); expected neurologist
  orthopaedist - bone + heart -> cardiologist (0.97); expected cardiologist
  orthopaedist - bone + kidney -> urinalysis (0.97); expected nephrologist
  all pairs: 14 of 30 correct (the sentences are regular and made by templates)

GloVe: 400,000 words x 50 numbers, learned from Wikipedia and a news archive
  king - man + woman -> queen 0.85, throne 0.77, prince 0.76   (a clean answer would be queen)
  paris - france + italy -> rome 0.85, milan 0.78, turin 0.77   (a clean answer would be rome)
  walking - walked + swam -> swim 0.81, swims 0.79, swimming 0.74   (a clean answer would be swimming)
  doctor - man + woman -> nurse 0.84, child 0.77, pregnant 0.76   (a clean answer would be doctor)
```

```text
$ uv run python -m demos.d06_dot_cosine
a = [1.0, 2.0, 0.0], b = [2.0, 1.0, 0.0]
dot(a, b)    = 1*2 + 2*1 + 0*0 = 4
length(a) = length(b) = sqrt(5) = 2.236
cosine(a, b) = 4 / 5 = 0.80
cosine(a, a) = 1.00  (a vector points exactly the way it points)
cosine(a, 2a) = 1.00  but the distance between a and 2a is 2.236 while a to b is 1.414
cosine([1,0,0], [0,1,0]) = 0.00   cosine(a, -a) = -1.00

cosine between rows of our trained table:
  kidney       nephrologist   0.92
  kidney       furosemide     0.97
  kidney       nephritis      0.96
  heart        cardiologist   0.89
  kidney       cardiologist  -0.08
  kidney       ramipril      -0.01
  furosemide   ramipril       0.12
  asthma       stroke         0.14

the same table, one word against every other word of its own body system and of all the others:
  same body system : 270 pairs, average cosine 0.97
  different systems: 1500 pairs, average cosine 0.28
```

```text
$ uv run python -m demos.d08_bag_of_tokens
output vector for 'bites' (first 4 of 8 numbers)
  word vectors only
    in 'dog bites man': [0.248, 0.245, -0.827, -0.197]
    in 'man bites dog': [0.248, 0.245, -0.827, -0.197]   identical: True
  word vectors + position codes
    in 'dog bites man': [-0.856, 1.76, 0.048, 1.043]
    in 'man bites dog': [-1.066, 0.973, 0.639, 1.067]   identical: False

without position codes the output for 'dog' is the same whether the dog comes first or last: True
largest difference with position codes for 'dog': 1.576
```

```text
$ uv run python -m demos.d11_rope
length of the query: 1.986816; largest change in length over 286 positions: 2.2e-16

attention score numerator (dot of rotated query and rotated key) for several positions:
  query at    5, key at    2 (distance 3):   1.69336
  query at  105, key at  102 (distance 3):   1.69336
  query at 1005, key at 1002 (distance 3):   1.69336
  query at    7, key at    3 (distance 4):   1.55178
  query at  107, key at  103 (distance 4):   1.55178
largest change when both positions move together by the same amount: 8.9e-16

stretching a window trained for 4096 tokens to 8192 tokens:
  position interpolation: positions 100, 4096, 6000, 8191 become [50.0, 2048.0, 3000.0, 4095.5]  (all inside 0..4096)
  NTK-style base: 10000 -> 20,452; slowest pair turns 1.33e-04 -> 6.67e-05 radians per position, fastest 1.00 -> 1.00
```

```text
$ uv run python -m demos.d12_lab_dimension
numbers per word  table entries  train loss  test loss  same system  same system, 2-D  same kind of word
               2            178       3.656      3.665           7%                7%                57%
               4            356       3.570      3.580          17%                7%                92%
               8            712       3.438      3.448         100%              100%                70%
              16          1,424       3.374      3.385          87%               98%                83%
              32          2,848       3.368      3.380          83%               70%                87%

random guessing costs ln(89) = 4.489; choosing a neighbour at random would match the body system about 1 time in 6 (17%)
```

```text
$ uv run pytest -v
============================== 31 passed in 2.48 s ==============================
```

Timings depend on your machine. The numbers repeat on the same versions, because the sentences are fixed and every training run uses a fixed seed. A newer release of a library, or of a downloaded model file, can change the last digits.

### What `worked_examples.py` reproduces

The video calculates small numbers by hand. `worked_examples.py` prints exactly those numbers using NumPy only and needs no download:

```text
== 1. The table is a lookup (6 tokens x 3 numbers)
  table size: 6 x 3 = 18 numbers
  token id 3 is 'kidney': row 3 = [0.2, 0.8, 0.7]
  one-hot [0,0,0,1,0,0] times the table = [0.2, 0.8, 0.7]  (the same row)
  real sizes: 30,522 x 384 = 11,720,448   50,257 x 768 = 38,597,376

== 2. Dot product and cosine of a = [1, 2, 0] and b = [2, 1, 0]
  dot = 1*2 + 2*1 + 0*0 = 4
  length of a = sqrt(1 + 4 + 0) = sqrt(5) = 2.236; length of b = sqrt(5) = 2.236
  cosine = 4 / (sqrt(5) * sqrt(5)) = 4 / 5 = 0.80
  angle = 36.9 degrees
  a with itself: dot = 5, cosine = 5 / 5 = 1.00
  a with 2a = [2, 4, 0]: cosine = 1.00, but the straight-line distance is 2.236 (a to b: 1.414)
  right angle [1,0,0] and [0,1,0]: cosine = 0; opposite a and -a: cosine = -1

== 3. A bag of tokens: dog = [1, 0, 2], bites = [0, 3, 1], man = [2, 1, 0]
  'dog bites man' sum = [3, 4, 3]   'man bites dog' sum = [3, 4, 3]  (identical)
  with position vectors added: dog at position 1 = [2, 0, 2], dog at position 3 = [1, 0, 3]  (different now)
  man at position 3 = [2, 1, 1], man at position 1 = [3, 1, 0]

== 4. Sinusoidal codes with 4 numbers per position (two turning pairs: 1 radian and 0.01 radian per position)
  position 0: [0.0, 1.0, 0.0, 1.0]
  position 1: [0.841, 0.54, 0.01, 1.0]
  position 2: [0.909, -0.416, 0.02, 1.0]
  position 3: [0.141, -0.99, 0.03, 1.0]
  dot(position 0, position 1) = cos(1) + cos(0.01) = 1.5403   from the table: 1.5403
  dot(position 1, position 2) = 1.5403   dot(position 2, position 3) = 1.5403  (same distance, same dot product)

== 5. Rotation by 30 degrees per position: q = [2, 0], k = [1, 1]
  q at position 3 turns 90 degrees: [0.0, 2.0]; its length stays 2.000 (before: 2.000)
  k at position 1 turns 30 degrees: [0.366, 1.366]
  score(q at 3, k at 1) = 2.732   distance 2
  score(q at 5, k at 3) = 2.732   distance 2 again, same score
  score(q at 4, k at 1) = 2.000   distance 3, a different score
  the library function rope.rotate keeps lengths too: 5.477226 vs 5.477226

== 6. Stretching a window trained for 4,096 tokens to 8,192
  position interpolation: scale = 4096 / 8192 = 0.5; position 6000 becomes 3000; position 8191 becomes 4095.5
  NTK-style base for 64 numbers per head and a stretch of 2: 10000 x 2^(64/62) = 20,452
[exit 0]
```

The same output is saved in `data/expected_output/worked_examples.txt`.

### Downloaded files (with checksums)

```text
model and vector files used (source, file, bytes, sha256):
huggingface.co/fse/glove-wiki-gigaword-50  glove-wiki-gigaword-50.model  21494764  d64693ba82109842fe8c9b1078970fae99de992e374d1ee9eba141b121e605a6
huggingface.co/fse/glove-wiki-gigaword-50  glove-wiki-gigaword-50.model.vectors.npy  80000128  5cee3016e3af5c1a3675d6b1f6d467b0fc7a0ba3dccf745aded1984ccce3ad74
huggingface.co/openai-community/gpt2  config.json  665  0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4
huggingface.co/Qwen/Qwen2.5-0.5B-Instruct  config.json  659  18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45
huggingface.co/sentence-transformers/all-MiniLM-L6-v2  config.json  612  953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41
huggingface.co/sentence-transformers/all-MiniLM-L6-v2  model.safetensors  90868376  53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db
huggingface.co/sentence-transformers/all-MiniLM-L6-v2  special_tokens_map.json  112  303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3
huggingface.co/sentence-transformers/all-MiniLM-L6-v2  tokenizer_config.json  350  acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b
huggingface.co/sentence-transformers/all-MiniLM-L6-v2  tokenizer.json  466247  be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037
huggingface.co/sentence-transformers/all-MiniLM-L6-v2  vocab.txt  231508  07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3
```

## Honest notes

- **The practice text is invented.** The 3,000 training sentences and 300 exam sentences come from templates about six body systems (heart, kidney, lung, liver, brain, bone), ten clinical words each, and the whole text holds 89 different words. The map looks tidy because the sentences are regular. Real text is messier, so real maps are less tidy.
- **"Neighbouring words" means every other word of the same sentence.** Each sentence gives one practice question for every ordered pair of its words, 388,972 pairs in all. That choice suits short sentences and keeps the code small.
- **The word arithmetic works for some pairs only.** On our map the "specialist - organ + other organ" trick is right for 14 of 30 pairs (for kidney it lands on urinalysis, expected nephrologist). On GloVe, king - man + woman gives queen, and doctor - man + woman gives nurse, which shows how vectors copy the habits of their texts. The search skips the three input words; when it keeps them, king itself wins (demo 5).
- **A lower loss is not always a better map.** In the lab the loss keeps falling from 2 to 32 numbers per word, yet the nearest neighbour comes from the same body system for 100 % of the words at 8 numbers and for 83 % at 32. Random choice scores about 17 % (one in six). With 2 or 4 numbers the table groups words by their kind (92 % at 4) instead. One text, one seed and one training length give these numbers; another set can shift them.
- **The embedding tables of real models are trained inside much larger models.** MiniLM, GPT-2 and Qwen2.5 learned their tables together with all their layers. Demo 3 reads the rows and the sizes; it does not retrain anything.
- **MiniLM, GloVe, GPT-2 and Qwen2.5 are public models with their own licences.** The files are downloaded from their Hugging Face pages when you run the demos and are never part of this repository. Check each licence before you reuse a model in your own project.
- **Our `rope.rotate` pairs neighbouring numbers by default** (0 with 1, 2 with 3), as in the RoFormer paper. The Llama and Qwen code in the transformers library pairs the first half of the numbers with the second half, and `pairing="half"` does the same. Demo 13 uses `pairing="half"` and shows that it agrees with Qwen's own code to better than 0.001 on random vectors up to position 20,000 (the gap is the rounding between 32-bit and 64-bit arithmetic); the other pairing would differ by 4.88, because the numbers are paired in another order.
- **The window-stretching examples are overviews.** Position interpolation and the NTK-aware base are calculated on a window of 4,096 tokens stretched to 8,192; real systems also train a little on longer text afterwards. Later lessons return to long windows.
- **Sentence vectors return in lesson 17.** Demo 7 gives a first look at the gap between a token vector and a sentence vector; retrieval gets its own lesson.
