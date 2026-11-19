"""make_corpus.py - write the small CareDesk practice text used in lesson 3.

What it does
    Builds a tiny, invented text about the front desk of a fictional hospital, one sentence per line, every
    word lowercase, every sentence ending in a full stop. It writes three files into ../data:

      toy_corpus.txt       the three sentences we count by hand in the video
      caredesk_train.txt   600 sentences the language models learn from
      caredesk_test.txt    40 sentences the models have never seen (half of them use word pairings that are
                           absent from the training sentences, on purpose)

What it needs
    Python 3.9 or newer. No other packages. Everything is invented: no real patient data is involved.

What to expect
    The same files every time you run it (the random seed is fixed). Run:  python make_corpus.py
"""
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# Who does what at the front desk. Each entry is (verb, object, weight): a bigger weight makes the sentence more common.
ROLES = {
    "nurse": [("checks", "chart", 3), ("greets", "patient", 3), ("updates", "chart", 2), ("prepares", "room", 2)],
    "doctor": [("reads", "chart", 3), ("greets", "patient", 3), ("signs", "form", 2), ("reviews", "schedule", 2)],
    "clerk": [("books", "appointment", 3), ("updates", "schedule", 3), ("prints", "form", 2), ("greets", "patient", 2),
              ("signs", "form", 1)],
    "pharmacist": [("fills", "prescription", 3), ("checks", "prescription", 3), ("signs", "form", 2),
                   ("greets", "patient", 1)],
}

# Pairings kept out of the training sentences. Each one makes a word pair (for example "doctor checks")
# that the training text never shows, so we can test how a model copes with a pair it has not seen.
HELD_OUT = [("doctor", "checks", "chart"), ("doctor", "updates", "chart"), ("clerk", "signs", "form"),
            ("pharmacist", "greets", "patient")]

TODAY_CHANCE = 0.3  # some sentences end with the extra word "today", so the next word after "chart" is not always a full stop


def make_sentence(rng, subject, verb, obj):
    """Return one sentence (a string) for a subject, a verb and an object; sometimes add the word 'today'."""
    words = ["the", subject, verb, "the", obj]
    if rng.random() < TODAY_CHANCE:
        words.append("today")
    words.append(".")
    return " ".join(words)


def usual_pairings():
    """Every (subject, verb, object, weight) the training text may use, with the held-out pairings left out."""
    held = set(HELD_OUT)
    return [(s, v, o, w) for s, acts in ROLES.items() for (v, o, w) in acts if (s, v, o) not in held]


def main():
    """Write the three corpus files."""
    rng = random.Random(7)  # fixed seed: the same files every time
    DATA.mkdir(parents=True, exist_ok=True)

    # the three sentences we count by hand in the video
    toy = ["the nurse checks the chart .", "the nurse greets the patient .", "the doctor reads the chart ."]
    (DATA / "toy_corpus.txt").write_text("\n".join(toy) + "\n")

    # training sentences: draw 600 pairings in proportion to their weights
    usual = usual_pairings()
    weights = [w for (_s, _v, _o, w) in usual]
    train = []
    for s, v, o, _w in rng.choices(usual, weights=weights, k=600):
        train.append(make_sentence(rng, s, v, o))
    (DATA / "caredesk_train.txt").write_text("\n".join(train) + "\n")

    # test sentences: 20 fresh sentences from usual pairings, then 20 from the held-out pairings (5 for each)
    test = []
    for s, v, o, _w in rng.choices(usual, weights=weights, k=20):
        test.append(make_sentence(rng, s, v, o))
    for s, v, o in HELD_OUT:
        for _ in range(5):
            test.append(make_sentence(rng, s, v, o))
    (DATA / "caredesk_test.txt").write_text("\n".join(test) + "\n")

    print(f"wrote {len(toy)} toy, {len(train)} training and {len(test)} test sentences to {DATA}")


if __name__ == "__main__":
    main()
