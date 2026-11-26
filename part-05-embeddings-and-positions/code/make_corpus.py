"""make_corpus.py - write the invented clinical practice text used in lesson 5.

What it does
    Writes three small files into ../data from fixed templates and a fixed random seed:
      clinical_words_train.txt   3000 short invented sentences about six body systems (heart, kidney, lung, liver, brain, bone)
      clinical_words_test.txt     300 more sentences of the same kind, for checking the trained model on unseen text
      word_groups.json           which group every word belongs to (used to colour the 2-D map and to check it)
    Each body system has its own organ, specialist, conditions, medicines, tests and clinic, so words that belong together
    appear in similar sentences. That is exactly the signal a model needs to learn that they are related.
    Every sentence is made up; nothing here is medical advice.

What it needs
    Only the Python standard library.

What to expect
    The same files every time you run it (the seed is fixed). Run it from this folder: python make_corpus.py
"""
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# one entry per body system: the words that belong to it, by role
SYSTEMS = {
    "heart": dict(organ=["heart"], staff=["cardiologist"], conditions=["arrhythmia", "angina", "infarction"],
                  drugs=["ramipril", "bisoprolol"], tests=["ecg", "troponin"], clinic=["cardiology"]),
    "kidney": dict(organ=["kidney"], staff=["nephrologist"], conditions=["nephritis", "uraemia", "cystitis"],
                   drugs=["furosemide", "sevelamer"], tests=["creatinine", "urinalysis"], clinic=["nephrology"]),
    "lung": dict(organ=["lung"], staff=["pulmonologist"], conditions=["pneumonia", "asthma", "bronchitis"],
                 drugs=["salbutamol", "budesonide"], tests=["spirometry", "oximetry"], clinic=["pulmonology"]),
    "liver": dict(organ=["liver"], staff=["hepatologist"], conditions=["hepatitis", "cirrhosis", "jaundice"],
                  drugs=["lactulose", "ursodiol"], tests=["bilirubin", "fibroscan"], clinic=["hepatology"]),
    "brain": dict(organ=["brain"], staff=["neurologist"], conditions=["stroke", "migraine", "epilepsy"],
                  drugs=["sumatriptan", "levetiracetam"], tests=["eeg", "mri"], clinic=["neurology"]),
    "bone": dict(organ=["bone"], staff=["orthopaedist"], conditions=["fracture", "osteoporosis", "arthritis"],
                 drugs=["alendronate", "naproxen"], tests=["dexa", "densitometry"], clinic=["orthopaedics"]),
}

# eight sentence templates; {organ}, {staff}, {condition}, {drug}, {test} and {clinic} are always filled from ONE body system
TEMPLATES = [
    "the {staff} reviewed the {organ} of the patient with {condition} .",
    "a patient with {condition} was treated with {drug} on the {clinic} ward .",
    "the {test} showed signs of {condition} in the {organ} .",
    "{drug} is often used for {condition} and for the {organ} .",
    "the {clinic} clinic ordered a {test} for {condition} .",
    "the {staff} in the {clinic} clinic prescribed {drug} after the {test} .",
    "a problem of the {organ} such as {condition} needs a {test} and {drug} .",
    "the {staff} explained the {test} and the {drug} to the patient with {condition} .",
]


def sentence(rng, system):
    """One invented sentence about a single body system, picked from the eight templates."""
    parts = SYSTEMS[system]
    fill = dict(organ=rng.choice(parts["organ"]), staff=rng.choice(parts["staff"]), condition=rng.choice(parts["conditions"]),
                drug=rng.choice(parts["drugs"]), test=rng.choice(parts["tests"]), clinic=rng.choice(parts["clinic"]))
    return rng.choice(TEMPLATES).format(**fill)


def group_labels():
    """Map every domain word to its body system (function words such as 'the' are left out)."""
    labels = {}
    for system, parts in SYSTEMS.items():
        for role, words in parts.items():
            for w in words:
                labels[w] = {"group": system, "role": role}
    return labels


def main():
    """Write the training text, the test text and the word-group file."""
    rng = random.Random(5)
    DATA.mkdir(parents=True, exist_ok=True)
    names = list(SYSTEMS)
    for name, count in (("clinical_words_train", 3000), ("clinical_words_test", 300)):
        lines = [sentence(rng, rng.choice(names)) for _ in range(count)]
        (DATA / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DATA / "word_groups.json").write_text(json.dumps(group_labels(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {DATA}/clinical_words_train.txt, clinical_words_test.txt and word_groups.json")


if __name__ == "__main__":
    main()
