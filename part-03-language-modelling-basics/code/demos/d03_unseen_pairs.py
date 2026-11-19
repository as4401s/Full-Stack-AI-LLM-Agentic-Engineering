"""Demo 3 - what a counting model does with a word pair it has never seen, and the add-one repair.

What it does: lists word pairs in the test sentences that the training sentences never contain, shows that the plain
bigram model gives them probability 0 (so its perplexity is infinite), then adds a small amount to every count.
What it needs: only the Python standard library.
What to expect: perplexity inf without smoothing; a finite number once every count gets a head start.
Run it: uv run python -m demos.d03_unseen_pairs
"""
from lm_basics.corpus import load, make_examples
from lm_basics.ngram import NGramModel
from lm_basics.report import run_and_save, save_results


def main():
    train, test = load("caredesk_train"), load("caredesk_test")
    seen_pairs = {(c[0], n) for c, n in make_examples(train, 1)}
    test_pairs = [(c[0], n) for c, n in make_examples(test, 1)]
    unseen = sorted({p for p in test_pairs if p not in seen_pairs})
    print(f"word pairs in the test sentences that the training sentences never contain: {unseen}")
    held = test[20:]
    results = {}
    for order in (2, 3):
        for k in (0.0, 0.1, 1.0):
            model = NGramModel(order=order, smoothing=k).fit(train)
            ppl_all, ppl_held = model.perplexity(test), model.perplexity(held)
            name = {2: "bigram ", 3: "trigram"}[order]
            print(f"{name} add {k:<3}: perplexity on all 40 test sentences {ppl_all:7.3f}   on the 20 new pairings {ppl_held:7.3f}")
            results[f"order{order}_k{k}"] = {"all": ppl_all, "held_out": ppl_held}
    plain = NGramModel(order=2).fit(train)
    print("\nP(checks | doctor) with plain counts   =", plain.prob("checks", ("doctor",)))
    smooth = NGramModel(order=2, smoothing=1.0).fit(train)
    print("P(checks | doctor) with add-one counts  =", round(smooth.prob("checks", ("doctor",)), 4))
    save_results("d03_unseen_pairs", results)


if __name__ == "__main__":
    run_and_save("d03_unseen_pairs", main)
