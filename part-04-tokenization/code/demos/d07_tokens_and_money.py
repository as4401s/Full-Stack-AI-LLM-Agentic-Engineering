"""Demo 7 - tokens and money: the same letter in English, German and Python, and a context-window budget.

What it does:
  1. counts the tokens of the invented discharge letter in English, German and Python with six tokenizers;
  2. shows how German compound words are cut by two OpenAI encodings;
  3. turns token counts into a cost with an EXAMPLE price (not any provider's real price) for easy arithmetic;
  4. fills example context windows (8,000 and 128,000 tokens) with a CareDesk request and shows how many notes still fit.
What it needs: tiktoken and tokenizers (small downloads on the first run).
What to expect: German costs more tokens than English on every tokenizer; our English-only BPE shows it most.
Run it: uv run python -m demos.d07_tokens_and_money
"""
from tokenizer_lab.corpus import letters, lines
from tokenizer_lab.realtok import all_tokenizers, show, tiktoken_encoding, try_load
from tokenizer_lab.report import run_and_save, save_results

# EXAMPLE prices, chosen only to make the arithmetic easy to follow (dollars per one million tokens)
PRICE_IN, PRICE_OUT = 2.0, 8.0
WINDOWS = (8_000, 128_000)   # example context-window sizes

SYSTEM_PROMPT = ("You are CareDesk, an assistant for the operations team of a fictional hospital. Answer in short, plain sentences. "
                 "Use only the notes you are given. If the notes do not contain the answer, say so and suggest who to ask. "
                 "Never give medical advice. Keep every answer under one hundred words.")
HISTORY = [
    ("user", "Which room is Mr. Berg in?"), ("assistant", "Mr. Berg is in room 12 on the cardiology ward."),
    ("user", "When is his follow-up visit?"), ("assistant", "The clerk booked it for 2026-03-19 at the cardiology clinic."),
    ("user", "Who prepares the room before he arrives?"), ("assistant", "The nurse prepares the room, and the clerk updates the schedule."),
]
QUESTION = "Please list the medicine orders for room 12 and tell me which ones start this month."


def main():
    texts = letters()
    toks = all_tokenizers()
    print("1. the same letter, three ways (English, German, Python)")
    print(f"  {'tokenizer':26s}{'English':>9s}{'German':>9s}{'Python':>9s}{'German / English':>19s}")
    table = {}
    for tok in toks:
        e, g, p = (tok.count(texts[k]) for k in ("english", "german", "python"))
        table[tok.name] = {"english": e, "german": g, "python": p, "ratio": g / e}
        print(f"  {tok.name:26s}{e:9d}{g:9d}{p:9d}{g / e:19.2f}")
    print(f"  (characters: English {len(texts['english'])}, German {len(texts['german'])}, Python {len(texts['python'])};"
          f" words: English {len(texts['english'].split())}, German {len(texts['german'].split())})")

    print("\n2. German compounds and their English counterparts")
    pairs = [("Kardiologiestation", "cardiology ward"), ("Kontrolltermin", "follow-up visit"), ("Blutdruck", "blood pressure"),
             ("Herzinfarkt", "heart attack"), ("Entlassungsbrief", "discharge letter")]
    compounds = {}
    for name in ("cl100k_base", "o200k_base"):
        enc = try_load(name, tiktoken_encoding, name)
        if enc is None:
            continue
        print(f"  {name}:")
        for de, en in pairs:
            pd, pe = enc.pieces(" " + de), enc.pieces(" " + en)
            print(f"    {de:20s} {len(pd):2d} tokens: {' | '.join(show(p) for p in pd)}")
            print(f"    {en:20s} {len(pe):2d} tokens: {' | '.join(show(p) for p in pe)}")
            compounds[f"{name}:{de}"] = {"de": pd, "en": pe}

    print(f"\n3. money, with an EXAMPLE price of ${PRICE_IN:.2f} per million input tokens and ${PRICE_OUT:.2f} per million output tokens")
    enc = try_load("o200k_base", tiktoken_encoding, "o200k_base")
    en_tokens, de_tokens = enc.count(texts["english"]), enc.count(texts["german"])
    requests_per_day, reply_tokens = 10_000, 300
    money = {}
    for label, prompt_tokens in (("English letter", en_tokens), ("German letter", de_tokens)):
        cost_in = requests_per_day * prompt_tokens * PRICE_IN / 1e6
        cost_out = requests_per_day * reply_tokens * PRICE_OUT / 1e6
        money[label] = {"prompt_tokens": prompt_tokens, "in": cost_in, "out": cost_out}
        print(f"  {label}: {prompt_tokens} prompt tokens x {requests_per_day:,} requests = {requests_per_day * prompt_tokens:,} tokens in -> ${cost_in:.2f};"
              f" {reply_tokens} reply tokens x {requests_per_day:,} = ${cost_out:.2f}; total ${cost_in + cost_out:.2f} per day")

    print("\n4. a context window is a shared budget (counted with o200k_base; the window sizes are EXAMPLES)")
    notes = lines("clinical_train")[:40]
    parts = {
        "system prompt": enc.count(SYSTEM_PROMPT),
        "chat so far": sum(enc.count(f"{r}: {m}\n") for r, m in HISTORY),
        "the question": enc.count(QUESTION),
    }
    per_note = enc.count("\n".join(notes)) / len(notes)
    reserve = 1000
    fixed = sum(parts.values()) + reserve
    budget = {}
    for window in WINDOWS:
        room = window - fixed
        budget[window] = {"room_for_notes": room, "notes_that_fit": int(room // per_note)}
    for name, n in parts.items():
        print(f"  {name:24s}{n:7d} tokens")
    print(f"  {'reserved for the reply':24s}{reserve:7d} tokens")
    print(f"  {'fixed parts together':24s}{fixed:7d} tokens")
    print(f"  one retrieved note costs about {per_note:.1f} tokens (average of 40 notes)")
    for window, b in budget.items():
        print(f"  window of {window:>7,}: {b['room_for_notes']:,} tokens left for notes -> about {b['notes_that_fit']:,} notes fit")
    save_results("d07_tokens_and_money", {"table": table, "compounds": compounds, "money": money, "parts": parts, "reserve": reserve, "per_note": per_note, "budget": budget})


if __name__ == "__main__":
    run_and_save("d07_tokens_and_money", main)
