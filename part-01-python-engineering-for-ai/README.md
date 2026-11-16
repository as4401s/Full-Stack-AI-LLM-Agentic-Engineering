# Video 1 - Python engineering for AI

Video: [VIDEO LINK, to be added when the video is published]

**Learning goal:** write Python the way AI projects need it: clear about the data it handles, able to do many things at once, and tested, because every later part of the course builds on it. We build one small program, the **CareDesk fan-out client**, that asks several AI services the same question at the same time, checks every reply, tries again when something fails for a moment and comes with tests.

This lesson assumes basic coding skills and a little familiarity with machine learning. Every idea is introduced before it is used.

## What the lesson covers

1. Projects with **uv**: `pyproject.toml` and the lockfile as a recipe card every machine cooks from.
2. **Type hints and Pydantic v2**: validation as a customs desk at the border of your service.
3. **async/await**: the event loop as one waiter serving many tables, `httpx.AsyncClient`, `asyncio.gather` and a semaphore as the door host.
4. **Reliability habits**: timeouts, retries with exponential backoff and jitter, idempotency keys.
5. **Testing with pytest**: fixtures, mocked HTTP, property checks, snapshot tests, testing output that varies.
6. **Logging, settings with pydantic-settings, and a small CLI.**
7. **SQL refresher**: a join, a GROUP BY and a window function on the synthetic CareDesk tables.

## Folder map

| Path | What it holds |
|---|---|
| `code/caredesk_fanout/models.py` | The typed shapes: `ChatRequest`, `ModelReply`, `Usage`, `Answer`, `Failure`, `FanOutResult` |
| `code/caredesk_fanout/fanout.py` | `fetch_one`, `fan_out` (concurrent) and `fan_out_sequential` (the slow way, for comparison) |
| `code/caredesk_fanout/reliability.py` | Retry policy, exponential backoff with jitter, idempotency keys |
| `code/caredesk_fanout/settings.py` | Settings read from `CAREDESK_*` environment variables with `pydantic-settings` |
| `code/caredesk_fanout/cli.py` | A small command line that prints one line per provider and the wall-clock time |
| `code/caredesk_fanout/mock_server.py` | A local stand-in for a model API (standard library only), so everything is free |
| `code/demos/` | Four short demos: sequential vs concurrent, validation, retries, idempotency |
| `code/tests/` | 21 pytest tests, including a snapshot file in `tests/snapshots/` |
| `code/worked_examples.py` | Reproduces every number calculated by hand in the video (plain Python) |
| `code/run_sql.py` | Builds the CareDesk tables and runs the three queries |
| `code/pyproject.toml`, `code/uv.lock` | The project recipe and the exact versions used in the recorded run |
| `code/.env.example` | Names of the settings (copy to `.env`; real keys come later in the course) |
| `data/schema.sql`, `seed.sql`, `queries.sql` | Synthetic CareDesk tables, rows and the three queries |
| `data/expected_output/` | The real output of every command below, captured in the recorded run |

## Run plan

| Mode | What | Cost |
|---|---|---|
| Any laptop or desktop with Python 3.12+ | every script, demo, test and query in this folder | free |
| Free Colab | not needed in this video | - |
| Shown, not run | nothing: the mock server stands in for paid model APIs | - |

The recorded run used uv 0.9.7 and CPython 3.13.9 (see `data/expected_output/environment.txt`). Your timings will differ a little, and the numbers in the lesson come from our recorded run.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the `code/` folder:

```bash
cd code
uv sync                 # creates .venv and installs the exact versions from uv.lock
```

## Run it

Each demo starts its own mock server, so one command is enough.

```bash
uv run python demos/01_sequential_vs_concurrent.py   # about 9 s one after another, about 3 s all at once
uv run python demos/02_validation.py                 # a good parcel, a broken parcel, a "5" that becomes 5
uv run python demos/03_retries.py                    # backoff with jitter (scaled-down waits, base 0.25 s)
uv run python demos/04_idempotency.py                # the double charge, with and without a key
uv run pytest -v                                     # 21 tests
uv run python worked_examples.py                     # the hand calculations from the video
uv run python run_sql.py                             # the join, the GROUP BY and the window function
```

To use the small command line, start the mock server in one terminal and the client in another:

```bash
uv run python -m caredesk_fanout.mock_server                                  # terminal 1
uv run python -m caredesk_fanout.cli "Summarise this referral" alpha beta gamma               # terminal 2
uv run python -m caredesk_fanout.cli "Summarise this referral" alpha beta gamma --sequential  # the slow way
```

## What you should see

The files in `data/expected_output/` hold the complete output. Highlights from the recorded run:

```text
$ uv run python demos/01_sequential_vs_concurrent.py
Three model calls, 3.0 s each (local mock server)
one after another :  9.11 s
all at once       :  3.02 s
```

```text
$ uv run python demos/02_validation.py
--- a broken parcel
rejected with 3 problems:
  text                       String should have at least 1 character
  usage.prompt_tokens        Input should be greater than or equal to 0
  usage.completion_tokens    Input should be a valid integer, unable to parse string as an integer
```

```text
$ uv run python demos/04_idempotency.py
without an idempotency key: work done 2x, charged 2x
with an idempotency key   : work done 1x, charged 1x
```

```text
$ uv run pytest -v
============================== 21 passed in 1.13s ==============================
```

Timings vary slightly from machine to machine; the reply texts of the mock server vary on purpose (that is what the "testing output that varies" tests are about).

### What `worked_examples.py` reproduces

The video calculates small numbers by hand. `worked_examples.py` prints exactly those numbers:

```text
== 1. Three calls of 3 s
one after another: 9 s   (3 + 3 + 3)
all at once      : 3 s   (the slowest call decides)

== 2. Ten calls of 3 s, door lets 5 in at a time
rounds: 2 -> total 6 s   versus one by one: 30 s

== 2b. Twelve calls of 3 s, door lets 4 in at a time
rounds: 3 (12 / 4)
each round takes 3 s -> total 9 s   versus one by one: 36 s

== 3. Backoff: base 1 s, doubling, cap 30 s, jitter up to 50 % of the wait
after failure 1: wait    1 s, plus 0 to  0.5 s random -> between 1.0 and 1.5 s
after failure 2: wait    2 s, plus 0 to  1.0 s random -> between 2.0 and 3.0 s
after failure 3: wait    4 s, plus 0 to  2.0 s random -> between 4.0 and 6.0 s
after failure 4: wait    8 s, plus 0 to  4.0 s random -> between 8.0 and 12.0 s
after failure 5: wait   16 s, plus 0 to  8.0 s random -> between 16.0 and 24.0 s
after failure 6: wait   30 s, plus 0 to 15.0 s random -> between 30.0 and 45.0 s
one possible run with seed 7:
  wait 1: 1 s + 0.16 s extra = 1.16 s
  wait 2: 2 s + 0.15 s extra = 2.15 s
  wait 3: 4 s + 1.30 s extra = 5.30 s
  total waiting before the fourth try: 8.61 s   (without jitter it would be 7 s)

== 4. 1000 clients all fail at the same moment and retry after 4 s
no jitter  : 1000 requests arrive in the same instant
with jitter: spread over 2 s -> about 500 requests per second

== 5. The first reply is lost, so the client retries
without a key: charges = 2
with a key   : charges = 1

== 6. CareDesk waits (minutes)
Emergency  count 4  sum 170  average 42.5  longest 65
Cardiology count 3  sum  70  average 23.3  longest 30
Radiology  count 5  sum 125  average 25.0  longest 40
Radiology running average, in time order:
  after appointment 1: 40.0
  after appointment 2: 25.0
  after appointment 3: 23.3
  after appointment 4: 25.0
  after appointment 5: 25.0
Radiology rank by wait (1 = longest): {40: 1, 10: 5, 20: 4, 30: 2, 25: 3}
```

The same text is saved in `data/expected_output/worked_examples.txt`.

## Lab: check and extend

The exercise at the end of the video: twelve providers, each taking 3 s, with a door that lets 4 in at a time. Twelve divided by four is 3 rounds, and 3 rounds of 3 s make 9 s in total. `worked_examples.py` prints this as section 2b.

Your lab has four steps:

1. **Clone and install.** Clone the repository, open this part's `code/` folder and run `uv sync`.
2. **Run the demos and the tests.** Use the commands in "Run it" above and compare your timings with the files in `data/expected_output/`.
3. **Add a fourth provider and a limit of two, and predict the time before you run it.** Start the mock server in one terminal, then ask four providers with a limit of two in another:

   ```bash
   uv run python -m caredesk_fanout.mock_server          # terminal 1: every normal call takes 3 s
   CAREDESK_MAX_CONCURRENCY=2 uv run python -m caredesk_fanout.cli "Summarise this referral" alpha beta gamma delta   # terminal 2
   ```

   Work out the rounds first (four calls, two places at the door), then check the "wall-clock time" line. Expect about 6 s: 2 rounds of 3 s.
4. **Make the flaky provider fail five times and watch the retry give up.** The provider named `flaky` answers with HTTP 503 a set number of times per request. With the default of 3 retries (4 attempts in total), five failures end in a `FAILED` line instead of an answer:

   ```bash
   uv run python -m caredesk_fanout.mock_server --flaky-failures 5     # terminal 1
   uv run python -m caredesk_fanout.cli "Summarise this referral" flaky  # terminal 2: gives up and returns a failure
   ```

   For a faster second try, allow 5 retries and shorten the first wait: `CAREDESK_MAX_RETRIES=5 CAREDESK_BACKOFF_BASE_S=0.25` in front of the same command. The sixth attempt succeeds.

Settings come from `CAREDESK_*` environment variables (see `code/caredesk_fanout/settings.py` and `code/.env.example`).

## Honest notes

- **The model API is a local mock.** Three "providers" (`alpha`, `beta`, `gamma`) each answer after 3.0 s. This makes the timings repeatable and free. Real paid providers return in later videos; the same client code will talk to them.
- **The retry demo is scaled down.** It uses a base wait of 0.25 s so it finishes quickly; `worked_examples.py` and the video use the real 1 s, 2 s, 4 s schedule.
- **SQL runs in SQLite here.** `run_sql.py` uses SQLite, which ships with Python, so there is nothing to install. The statements are standard SQL (join, GROUP BY, `RANK()` and `AVG() OVER`).
- **All data is synthetic.** Patient codes such as `SYN-0001` are made up. Nothing here is medical advice.

## Where these ideas return

The async client returns in Video 14 (serving) and Video 34 (streaming APIs). Pydantic models return as structured outputs and tool schemas. The SQL refresher returns when we build the app database and text-to-SQL.
