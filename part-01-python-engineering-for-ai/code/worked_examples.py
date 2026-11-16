"""worked_examples.py - the small numbers from lesson 1, calculated by hand in plain Python.

What it does: reproduces, with no libraries at all, every little calculation the video does on screen:
how long calls take one by one versus all at once, how many fit through the semaphore door,
the backoff schedule with jitter, a retry stampede with and without jitter, how an idempotency key
protects a ledger, and the averages and running averages of the SQL refresher.
What it needs: Python 3.10 or newer, nothing else.
What to expect: the printed lines listed in the part README, identical on every run (the random
numbers use a fixed seed).
Run it:  python3 worked_examples.py
"""
import math
import random


def section(title: str) -> None:
    print(f"\n== {title}")


# 1. One after another versus all at once -------------------------------------------------------
section("1. Three calls of 3 s")
call_seconds = [3, 3, 3]
print("one after another:", sum(call_seconds), "s   (3 + 3 + 3)")
print("all at once      :", max(call_seconds), "s   (the slowest call decides)")

# 2. The semaphore door: ten calls, five at a time ---------------------------------------------
section("2. Ten calls of 3 s, door lets 5 in at a time")
calls, limit, seconds_each = 10, 5, 3
waves = math.ceil(calls / limit)       # how many rounds of five are needed
print("rounds:", waves, "-> total", waves * seconds_each, "s   versus one by one:", calls * seconds_each, "s")

# 3. Exponential backoff with jitter -------------------------------------------------------------
section("3. Backoff: base 1 s, doubling, cap 30 s, jitter up to 50 % of the wait")
base, cap, jitter = 1.0, 30.0, 0.5
for attempt in range(6):
    wait = min(cap, base * 2**attempt)
    print(f"after failure {attempt + 1}: wait {wait:>4.0f} s, plus 0 to {wait * jitter:>4.1f} s random -> between {wait:.1f} and {wait * (1 + jitter):.1f} s")
rng = random.Random(7)                  # a fixed seed makes the 'random' numbers repeatable
print("one possible run with seed 7:")
total = 0.0
for attempt in range(3):
    wait = min(cap, base * 2**attempt)
    delay = wait + rng.uniform(0, wait * jitter)
    total += delay
    print(f"  wait {attempt + 1}: {wait:.0f} s + {delay - wait:.2f} s extra = {delay:.2f} s")
print(f"  total waiting before the fourth try: {total:.2f} s   (without jitter it would be {1 + 2 + 4} s)")

# 4. A stampede with and without jitter -----------------------------------------------------------
section("4. 1000 clients all fail at the same moment and retry after 4 s")
clients = 1000
print("no jitter  :", clients, "requests arrive in the same instant")
window = 4 * jitter                      # the retries spread over 0 to 2 s
print(f"with jitter: spread over {window:.0f} s -> about {clients / window:.0f} requests per second")

# 5. Idempotency keys and a ledger ------------------------------------------------------------------
section("5. The first reply is lost, so the client retries")
no_key_charges = 1 + 1                   # original run + the retry both run and are charged
key_charges = 1                          # the retry is recognised and replayed from the saved reply
print("without a key: charges =", no_key_charges)
print("with a key   : charges =", key_charges)

# 6. The SQL refresher by hand ------------------------------------------------------------------------
section("6. CareDesk waits (minutes)")
waits = {"Emergency": [35, 50, 20, 65], "Cardiology": [15, 30, 25], "Radiology": [40, 10, 20, 30, 25]}
for name, w in waits.items():
    print(f"{name:<10} count {len(w)}  sum {sum(w):>3}  average {sum(w) / len(w):.1f}  longest {max(w)}")
print("Radiology running average, in time order:")
running = []
for i, value in enumerate(waits["Radiology"], start=1):
    running.append(sum(waits["Radiology"][:i]) / i)
    print(f"  after appointment {i}: {running[-1]:.1f}")
ranks = sorted(waits["Radiology"], reverse=True)
print("Radiology rank by wait (1 = longest):", {v: ranks.index(v) + 1 for v in waits["Radiology"]})
