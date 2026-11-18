# Video 2 - Deep learning refresher with PyTorch

Video: [VIDEO LINK, to be added when the video is published]

**Learning goal:** train a small neural network by hand and explain every line of the training loop. Every language model in this course is a neural network, so this lesson opens the box: what the parameters are (the adjustable numbers inside a network), how a score for the mistakes (the loss) guides them, how PyTorch works out every slope, and how to keep a network honest while it learns. We write the network twice on the same data: first with plain tensors, so every line of the learning loop is visible, then with PyTorch's building blocks (`nn.Module` and an optimizer), so we see what they save us.

This lesson assumes basic coding skills and a little familiarity with machine learning. Every idea is introduced before it is used, with a small example you can repeat on a calculator.

## What the lesson covers

The video has four parts:

1. **How a network learns.** Watch a small network learn two spirals; meet the parameters (pictured as knobs on a mixing desk), the score for the mistakes (the loss), gradient descent as walking downhill in fog, the learning rate as the size of a step, and backpropagation as tracing a dish back through its recipe, worked with real numbers and then in PyTorch's autograd.
2. **The building blocks in PyTorch.** Tensors and their shapes (and how to read a shape error), softmax (scores into probabilities), cross-entropy (how surprised were we?) and the four-move learning loop: forward, loss, backward, step.
3. **Learning well.** Training versus validation (studying versus sitting the exam), overfitting, dropout, weight decay, optimizers (SGD, momentum, AdamW), a learning-rate sweep, warmup and cosine schedules, number formats (float32, bfloat16, float16) and where the numbers are computed (CPU, CUDA, MPS).
4. **The code and the lab.** Two versions of the same network side by side, the tests, and an exercise: how many hidden units does the spiral job need?

## Folder map

| Path | What it holds |
|---|---|
| `code/dl_refresher/data.py` | The two-spirals data: number of points, noise level and seed in, identical points out |
| `code/dl_refresher/raw_mlp.py` | The network from four loose tensors: forward pass, softmax and cross-entropy by hand, the four-move loop with a manual update |
| `code/dl_refresher/module_mlp.py` | The same idea with `nn.Module`, dropout, an optimizer (SGD, momentum or AdamW) and an optional schedule |
| `code/dl_refresher/report.py` | Runs a demo and saves its printed output and numbers next to the code |
| `code/demos/` | Eleven short demos, `d01` to `d11`, one per idea in the video |
| `code/tests/` | 16 pytest tests |
| `code/worked_examples.py` | Reproduces every small number calculated by hand in the video (plain Python; PyTorch is optional) |
| `code/run_all.py` | Runs all eleven demos in order and saves their output |
| `code/pyproject.toml`, `code/uv.lock` | The project recipe and the exact versions used in the recorded run |
| `data/expected_output/` | The real printed output of every command below, from the recorded run |
| `data/results/` | The numbers behind the demos as JSON (loss curves, sweeps, timings) |

## Run plan

| Mode | What | Cost |
|---|---|---|
| Any laptop or desktop with Python 3.12+ | every demo and test in this folder (the CPU is enough) | free |
| Free Colab | optional: the device demo shows a faster result on a GPU | free |
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
uv run python -m demos.d01_one_weight          # gradient descent on one parameter: 0 -> 0.4 -> 0.72 -> 0.976, and a step size of 1.1 that overshoots
uv run python -m demos.d02_backprop            # the recipe traced backwards: slopes -12, -6, -18, -6, checked by nudging each parameter
uv run python -m demos.d03_tensors_shapes      # shapes, broadcasting and three shape mistakes caught on purpose
uv run python -m demos.d04_softmax             # scores 2.0, 1.0, 0.1 -> probabilities 0.659, 0.242, 0.099 and the loss 0.417
uv run python -m demos.d05_parameters          # counting the parameters: 82 in the small network, 17,154 in a wide one
uv run python -m demos.d06_train_raw           # training with plain tensors, 1000 steps
uv run python -m demos.d07_train_module        # training with nn.Module and AdamW, 500 steps
uv run python -m demos.d08_overfit             # memorizing 50 points; dropout and weight decay calm it down
uv run python -m demos.d09_optimizers          # SGD, momentum, AdamW; a learning-rate sweep; warmup and cosine
uv run python -m demos.d10_precision_device    # number formats and CPU versus GPU timings (timings vary per machine)
uv run python -m demos.d11_lab_width           # the lab: how many hidden units does the spiral job need?
uv run pytest -v                               # 16 tests
uv run python worked_examples.py               # the hand calculations from the video (plain Python)
uv run python run_all.py                       # all eleven demos in one go; saves outputs to data/
```

Every demo finishes within a few seconds on a laptop CPU (see `data/expected_output/run_all.txt` for the recorded times).

## What you should see

The files in `data/expected_output/` hold the complete output. Highlights from the recorded run:

```text
$ uv run python -m demos.d06_train_raw
raw tensors, 2 -> 16 -> 2, plain gradient descent, learning rate 1.0
 step  train loss  train acc  val loss  val acc
    0      0.6744      0.490    0.6598    0.502
   10      0.5297      0.640    0.5740    0.572
   50      0.3739      0.835    0.4077    0.822
  100      0.2316      0.942    0.2357    0.960
  300      0.0969      0.965    0.0850    0.975
 1000      0.0633      0.978    0.0550    0.983
```

```text
$ uv run python -m demos.d07_train_module
nn.Module network with 82 parameters, AdamW, learning rate 0.01
 step  train loss  train acc  val loss  val acc
    0      0.7102      0.500    0.7054    0.500
   10      0.6316      0.620    0.6308    0.623
   50      0.5018      0.658    0.5438    0.580
  100      0.3806      0.788    0.4052    0.755
  500      0.0819      0.970    0.0689    0.980
```

```text
$ uv run python -m demos.d08_overfit
setting        train loss  train acc  val loss  val acc  best val loss  at step
plain              0.0001      1.000    1.1597    0.882         0.2653       25
weight decay       0.0001      1.000    0.8516    0.882         0.2672       26
dropout            0.0021      1.000    0.5374    0.923         0.2679      132
both               0.0036      1.000    0.3743    0.923         0.2672       41
```

```text
$ uv run pytest -v
============================== 16 passed in 0.96s ==============================
```

Timings and the CPU-versus-GPU lines of `d10_precision_device` depend on your machine. The learning curves are repeatable, because every random choice uses a fixed seed; a different PyTorch version or processor can still change the last digit.

### What `worked_examples.py` reproduces

The video calculates small numbers by hand. `worked_examples.py` prints exactly those numbers using plain Python:

```text
parameters: layer 1 = 48, layer 2 = 34, total = 82
layer, unit A: 2.0 * 0.5 + 1.0 * -1.0 + 0.5 = 0.5  ->  ReLU keeps 0.5
layer, unit B: -2.0 * 0.5 + 1.0 * -1.0 + 0.5 = -1.5  ->  ReLU keeps 0.0
gradient descent, learning rate 0.1: [0.0, 0.4, 0.72, 0.976, 1.1808, 1.3446, 1.4757]
gradient descent, learning rate 1.1: [0.0, 4.4, -0.88, 5.456, -2.1472, 6.9766, -3.972]
forward: hidden = 3.0, prediction = 3.0, loss = 9.0
backward: dL/dprediction = -6.0, dL/db2 = -6.0, dL/dw2 = -18.0, dL/db1 = -6.0, dL/dw1 = -12.0
nudge w2 by 0.01: loss goes from 9.0 to 8.8209 (change -0.1791, predicted -0.18)
exp: [7.389, 2.718, 1.105]  sum: 11.213
probabilities: [0.659, 0.242, 0.099]  total: 1.0
cross-entropy for class 0: -ln(0.659) = 0.417
if the right class were 2: loss = 2.317 (a confident miss costs more)
momentum 0.9, learning rate 0.1: [0.0, 0.4, 1.08, 1.876, 2.617]  (it rolls past 2, then comes back)
dropout: hidden [2.0, 4.0, 6.0, 8.0], mask [1, 0, 1, 0] -> [4.0, 0.0, 12.0, 0.0]; in eval mode the values stay [2.0, 4.0, 6.0, 8.0]
each unit is kept half the time and doubled, so on average it stays the same: [2.0, 4.0, 6.0, 8.0]
weight decay: weight 2.0, learning rate 0.1, decay 0.1 -> factor 0.99 -> new weight 1.98
schedule (base 0.1, warmup 2, total 10): [0.05, 0.1, 0.1, 0.0962, 0.0854, 0.0691, 0.05, 0.0309, 0.0146, 0.0038, 0.0]
pi in bfloat16 = 3.140625, 1.001 in bfloat16 = 1.0, 1000.7 in bfloat16 = 1000.0
```

The same text is saved in `data/expected_output/worked_examples.txt`.

## Lab: check and extend

The exercise at the end of the video: our network has 16 hidden units and reaches about 98 percent on the validation points. Predict what happens with 2, 4, 8 and 64 hidden units, and think about how many parameters each size has. Then run:

```bash
uv run python -m demos.d11_lab_width
```

The recorded run:

```text
 hidden  parameters  train loss  train acc  val acc
      2          12      0.5606      0.618    0.553
      4          22      0.4149      0.800    0.755
      8          42      0.1990      0.897    0.910
     16          82      0.0819      0.970    0.980
     64         322      0.0473      0.993    0.993
```

Two hidden units have too few bends to follow a spiral. Sixteen is enough, and 64 adds only a little. To extend the lab, change one setting at a time (the learning rate, the number of steps, the noise level or the dropout share), predict the result first, then run the demo and compare. Predicting before running turns experiments into understanding.

## Honest notes

- **The data is synthetic.** The two spirals are drawn by `data.py` from a seed, so everyone gets the same points.
- **The network is small on purpose.** 82 parameters, one hidden layer, a couple of seconds per run. The same four moves train every network in the course, from this one to a language model.
- **Timings are examples.** The speed lines in `d06`, `d07` and `d10` come from the recorded run. Yours will differ; the loss and accuracy numbers are the ones to compare.
- **Everything is free.** The whole lesson runs on a laptop CPU.

## Where these ideas return

The training loop returns when we pretrain our own small language model (lesson 10) and when we fine-tune models (lesson 36). Softmax and cross-entropy return in lesson 3, when a network predicts the next word. Optimizers, schedules, number formats and devices return whenever a model grows large.
