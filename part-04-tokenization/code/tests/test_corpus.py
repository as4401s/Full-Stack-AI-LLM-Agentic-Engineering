"""Tests for the invented text files."""
from tokenizer_lab.corpus import letters, lines, read

NEW_DRUGS = ("rivaroxaban", "dapagliflozin", "carvedilol", "tamsulosin")


def test_sizes():
    assert len(lines("clinical_train")) == 2000 and len(lines("clinical_test")) == 200


def test_new_drug_names_appear_only_in_the_test_text():
    train, test = read("clinical_train"), read("clinical_test")
    assert all(d not in train for d in NEW_DRUGS) and any(d in test for d in NEW_DRUGS)


def test_letters_have_three_versions():
    texts = letters()
    assert set(texts) == {"english", "german", "python"}
    assert texts["english"].startswith("Discharge letter.") and texts["german"].startswith("Entlassungsbrief.")
    assert texts["python"].startswith("def discharge_letter(patient):")
