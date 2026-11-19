# Video 3 - Language modelling basics

Video: [VIDEO LINK, to be added when the video is published]

**Learning goal:** see how a language model guesses the next word, and build two of them side by side on the same small text. First a model made only of counts, which shows the idea in its simplest form (probabilities, the chain rule, weighted dice, perplexity), then a small neural network that gives every word a short list of numbers (an embedding) and handles word pairings the counts never saw. Every chat assistant in this course is built on this one skill: guessing the next word, again and again. We finish with the dials that steer the writing (temperature, top-k, top-p) and the chat template that turns a conversation into one long text.

This lesson assumes basic coding skills and a little familiarity with machine learning. Every idea is introduced before it is used, with a small example you can repeat on a calculator.

## What the lesson covers

The video has five parts:

1. **Guessing the next word by counting.** A phone keyboard as a language model, how a computer sees text (words as tokens; lesson 4 builds real tokens), probability from counts, the chain rule (multiply one probability per word), the log score (surprise) and writing sentences by rolling a weighted die.
2. **When counting runs short.** A sentence built from a word pair that the examples left out gets probability zero; adding one to every count repairs it, and longer contexts show why counting needs ever more text.
3. **A network that guesses the next word.** Embeddings (a short list of numbers for every word), a small network with 536 parameters, cross-entropy and perplexity as scores for the guesses, and a comparison with the counting models on 40 sentences the models have never seen, followed by the weight-decay sweep (how short a leash keeps the network honest).
4. **Choosing the next word.** Softmax again, greedy choice, temperature, top-k and top-p, and how a chat becomes one long text with role markers.
5. **The code and the lab.** The two models read side by side, the tests, and an exercise: how much weight decay helps on word pairings the model has not seen?

Tokenization has its own lesson (lesson 4), pretraining comes in lesson 10 and fine-tuning in lesson 36; this lesson gives the picture they build on.

## Folder map

| Path | What it holds |
|---|---|
| `code/lm_basics/corpus.py` | Reads the practice text, splits it into words, builds the vocabulary and cuts the text into (context, next word) examples |
| `code/lm_basics/ngram.py` | The counting model: bigram and trigram counts, add-one repair, chain-rule sentence scores, perplexity and weighted-dice writing |
| `code/lm_basics/mlp_lm.py` | The small neural model: an embedding table, one hidden layer, a score for every word, plus the training loop and evaluation |
| `code/lm_basics/sampler.py` | Softmax with temperature, greedy choice, top-k and top-p |
| `code/lm_basics/chat_template.py` | Writes a list of chat messages as one piece of text with role markers |
| `code/lm_basics/report.py` | Runs a demo and saves its printed output and numbers next to the code |
| `code/demos/` | Eleven short demos, `d01` to `d11`, one per idea in the video |
| `code/tests/` | 30 pytest tests |
| `code/make_corpus.py` | Writes the invented CareDesk practice text (the three toy sentences, 600 training sentences, 40 test sentences) |
| `code/worked_examples.py` | Reproduces every small number calculated by hand in the video (plain Python and NumPy; PyTorch is optional) |
| `code/run_all.py` | Runs all eleven demos in order and saves their output |
| `code/pyproject.toml`, `code/uv.lock` | The project recipe and the exact versions used in the recorded run |
| `data/caredesk_train.txt`, `data/caredesk_test.txt`, `data/toy_corpus.txt` | The invented practice text, one sentence per line (fictional, written for practice only) |
| `data/expected_output/` | The real printed output of every command below, from the recorded run |
| `data/results/` | The numbers behind the demos as JSON |

## Run plan

| Mode | What | Cost |
|---|---|---|
| Any laptop or desktop with Python 3.12+ | every demo and test in this folder (the CPU is enough) | free |
| Shown, not run | (this lesson has none: every command below runs for real on a laptop) | - |

The recorded run used uv 0.9.7, CPython 3.12.12, PyTorch 2.15.0, NumPy 2.5.5 and pytest 9.2.0 (see `data/expected_output/environment.txt`). Installing PyTorch downloads a few hundred megabytes the first time.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the `code/` folder:

```bash
cd code
uv sync                 # creates .venv and installs the exact versions from uv.lock
```

## Run it

```bash
uv run python -m demos.d01_counts_chain_rule   # counting three toy sentences: P(nurse | the) = 0.333, the six-step chain rule gives 1/18 = 0.0556
uv run python -m demos.d02_generate_dice       # bigram and trigram probabilities from 600 sentences, and ten sentences written by rolling weighted dice
uv run python -m demos.d03_unseen_pairs        # a word pair the text never showed: probability 0, perplexity infinity, then the add-one repair
uv run python -m demos.d04_train_neural        # the neural model: 536 parameters, loss 3.311 -> 0.722, perplexity on sentences it never saw
uv run python -m demos.d05_generalise          # the embedding table: two numbers per word, and which words end up close together
uv run python -m demos.d06_perplexity          # surprise = -ln(probability), and loss to perplexity
uv run python -m demos.d07_sampling            # softmax, temperature 0.5 / 1 / 2, top-k and top-p on the scores 2.0, 1.0, 0.1
uv run python -m demos.d08_generate_from_model # sentences written by the trained model at four temperatures
uv run python -m demos.d09_chat_template       # a chat written as one piece of text with role markers
uv run python -m demos.d10_lab_weight_decay    # the lab: five leash lengths (weight decay 0, 0.5, 1, 2, 4)
uv run python -m demos.d11_suggestions         # next-word suggestions from the counting model and the neural model side by side
uv run pytest -v                               # 30 tests
uv run python worked_examples.py               # the hand calculations from the video (no PyTorch needed)
uv run python run_all.py                       # all eleven demos in one go; saves outputs to data/
```

Every demo finishes within a few seconds on a laptop CPU (see `data/expected_output/run_all.txt` for the recorded times).

## What you should see

The files in `data/expected_output/` hold the complete output. Highlights from the recorded run:

```text
$ uv run python -m demos.d01_counts_chain_rule
'the' is followed by a word 6 times:
  nurse    2 of 6  ->  P(nurse | the) = 0.333
  chart    2 of 6  ->  P(chart | the) = 0.333
  patient  1 of 6  ->  P(patient | the) = 0.167
  doctor   1 of 6  ->  P(doctor | the) = 0.167
...
product of the six numbers = 0.0556   (1/18 = 0.0556)
average loss per word = 0.4817   perplexity = e^loss = 1.619
```

```text
$ uv run python -m demos.d03_unseen_pairs
bigram  add 0.0: perplexity on all 40 test sentences     inf   on the 20 new pairings     inf
bigram  add 1.0: perplexity on all 40 test sentences   4.076   on the 20 new pairings   5.357
trigram add 1.0: perplexity on all 40 test sentences   3.591   on the 20 new pairings   5.966
```

```text
$ uv run python -m demos.d04_train_neural
vocabulary: 24 words; a blind guess among 24 words has loss ln(24) = 3.178 and perplexity 24
parameters: embedding 48 + hidden layer 80 + output layer 408 = 536
step   training loss
   0   3.311
  100   0.953
  300   0.722

perplexity on sentences the models never saw (lower is better):
model                          all 40  20 usual  20 new pairings
neural (2 words of context)     2.466     2.038            2.969
bigram, add one                 4.076     3.081            5.357
trigram, add one                3.591     2.136            5.966
trigram, add 0.1                3.736     1.792            7.658
```

```text
$ uv run python -m demos.d07_sampling
                                  checks  greets   reads
  scores                           2.000   1.000   0.100
  temperature 1 (softmax)          0.659   0.242   0.099
  temperature 0.5                  0.864   0.117   0.019
  temperature 2                    0.502   0.304   0.194
  top-k, k=2 (temperature 1)       0.731   0.269   0.000
  top-p, p=0.9 (temperature 1)     0.731   0.269   0.000
```

```text
$ uv run pytest -v
============================== 30 passed in 1.20s ==============================
```

Timings depend on your machine. The numbers are repeatable, because every random choice uses a fixed seed; a different PyTorch version or processor can still change the last digit.

### What `worked_examples.py` reproduces

The video calculates small numbers by hand. `worked_examples.py` prints exactly those numbers using plain Python and NumPy:

```text
== 1. Counting: what follows 'the'?
the appears 6 times; the words after it: {'nurse': 2, 'chart': 2, 'patient': 1, 'doctor': 1}
  P(nurse | the) = 2/6 = 0.333
  P(chart | the) = 2/6 = 0.333
  P(patient | the) = 1/6 = 0.167
  P(doctor | the) = 1/6 = 0.167
== 2. The chain rule: multiply one probability per word
  P(the | .) = 1.0
  P(nurse | the) = 0.333
  P(checks | nurse) = 0.5
  P(the | checks) = 1.0
  P(chart | the) = 0.333
  P(. | chart) = 1.0
  product = 1 x 1/3 x 1/2 x 1 x 1/3 x 1 = 1/18 = 0.0556
== 3. Weighted dice: slices of the number line from 0 to 1
  nurse: from 0.0 up to 0.333
  chart: from 0.333 up to 0.667
  patient: from 0.667 up to 0.833
  doctor: from 0.833 up to 1.0
  a roll of 0.52 lands in the slice of 'chart'
== 4. A pair the text never showed
  P(greets | doctor) = 0.0 so the whole sentence 'the doctor greets the patient .' gets 0.0
  words in the toy vocabulary: 9
  add one to every count: P(greets | doctor) = (0 + 1) / (1 + 9) = 0.1
  add one to every count: P(reads | doctor)  = (1 + 1) / (1 + 9) = 0.2
  the probabilities after 'doctor' still add up to 1.0
== 5. Embeddings: two numbers for every word
  after 'nurse': scores checks 2.0, greets 1.0, reads 0.1  ->  probabilities [0.659, 0.242, 0.099]
  after 'doctor': scores checks 1.95, greets 0.9, reads 0.18  ->  probabilities [0.658, 0.23, 0.112]
  distance nurse to doctor: 0.141   nurse to chart: 1.746
== 6. Cross-entropy and perplexity
  probability 1.0: loss = -ln(1.0) = 0.0
  probability 0.5: loss = -ln(0.5) = 0.693
  probability 0.25: loss = -ln(0.25) = 1.386
  probability 0.125: loss = -ln(0.125) = 2.079
  probability 0.1: loss = -ln(0.1) = 2.303
  four steps with probabilities [0.5, 0.25, 0.5, 0.125] -> losses [0.693, 1.386, 0.693, 2.079] -> average 1.213
  perplexity = e^1.213 = 3.364   (check: 128 to the power 1/4 = 3.364)
  a loss of 2.0 means perplexity e^2.0 = 7.389 equally likely choices
  a blind guess among 10 words: loss ln(10) = 2.303, perplexity 10.0
  the toy sentence: six steps, average loss 0.4817, perplexity 1.619
== 7. Sampling from the scores [2.0, 1.0, 0.1] for checks, greets, reads
  greedy picks the highest score: index 0 (checks)
  temperature 1.0: scores / 1.0 = [2.0, 1.0, 0.1] -> probabilities [0.659, 0.242, 0.099]
  temperature 0.5: scores / 0.5 = [4.0, 2.0, 0.2] -> probabilities [0.864, 0.117, 0.019]
  temperature 2.0: scores / 2.0 = [1.0, 0.5, 0.05] -> probabilities [0.502, 0.304, 0.194]
  top-k with k=2 at temperature 1: [0.731, 0.269, 0.0]
  top-p with p=0.9 at temperature 1: [0.731, 0.269, 0.0] (the first two words already cover 0.901 )
  top-p 0.9 when the model is sure   [0.95, 0.03, 0.02] -> [1.0, 0.0, 0.0]
  top-p 0.9 when the model is unsure [0.35, 0.3, 0.2, 0.1, 0.05] -> [0.368, 0.316, 0.211, 0.105, 0.0]
== 8. A chat template
<|system|>You are CareDesk, a helpful front-desk assistant.<|end|>
<|user|>When does the clinic open?<|end|>
<|assistant|>
```

The same text is saved in `data/expected_output/worked_examples.txt`.

## Lab: check and extend

The exercise at the end of the video: weight decay is a short leash on the size of the network's numbers. Predict how the test perplexity changes with weight decay 0, 0.5, 1, 2 and 4, then run:

```bash
uv run python -m demos.d10_lab_weight_decay
```

The recorded run shows the memorising network (decay 0) doing worst on the 20 new word pairings (4.706), a moderate leash doing best (about 2.9 to 3.1), and a very short leash (decay 4) leaving the network too little room to learn (8.376).

```text
weight decay  train loss  test perplexity  on new pairings
         0.0       0.558            2.889            4.706
         0.5       0.617            2.401            3.110
         1.0       0.722            2.466            2.969
         2.0       1.156            2.993            2.925
         4.0       2.157            8.406            8.376
```

Then extend it: open `code/demos/d10_lab_weight_decay.py`, change `context_size` from 2 to 3 (or `embed_dim` from 2 to 4) and run it again. Which settings give the lowest perplexity on the new pairings now?

## Notes on reproducibility

- The practice text is invented and small on purpose, so every number can be checked by hand or by the tests. `code/make_corpus.py` rebuilds the three files in `data/` and writes identical text every time.
- Every random choice uses a fixed seed, so repeated runs print the same numbers on the same versions.
- The videos and the Medium article for this lesson show the same numbers as the files in `data/expected_output/`.
