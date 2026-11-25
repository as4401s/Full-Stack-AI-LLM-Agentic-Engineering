"""make_corpus.py - write the invented CareDesk clinical text used in lesson 4.

What it does
    Writes four small text files into ../data from fixed templates and a fixed random seed:
      clinical_train.txt   2000 short lines of invented hospital-operations text (training text for our tokenizer)
      clinical_test.txt     200 more lines, including drug names the training text never mentions
      letters.txt          the same invented discharge letter in English, in German and as a few lines of Python
    Every name, room, date and medicine order is made up; nothing here is medical advice.

What it needs
    Only the Python standard library.

What to expect
    The same files every time you run it (the seed is fixed). Run it from this folder: python make_corpus.py
"""
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

SURNAMES = ["Berg", "Okafor", "Silva", "Novak", "Haddad", "Lindqvist", "Tanaka", "Moreau", "Kowalski", "Rahman", "Costa", "Weber"]
DRUGS_SEEN = ["amoxicillin", "metformin", "lisinopril", "atorvastatin", "paracetamol", "ibuprofen", "omeprazole", "salbutamol",
              "amlodipine", "levothyroxine", "warfarin", "furosemide", "prednisolone", "ramipril", "bisoprolol", "sertraline"]
DRUGS_NEW = ["rivaroxaban", "dapagliflozin", "carvedilol", "tamsulosin"]   # appear only in the test file
DOSES = [5, 10, 20, 40, 50, 100, 250, 400, 500, 1000]
HOW_OFTEN = ["once daily", "twice daily", "three times daily", "every morning", "at night"]
WARDS = ["cardiology", "radiology", "maternity", "surgery", "oncology", "paediatrics", "geriatrics"]
ROLES = ["nurse", "doctor", "clerk", "pharmacist"]
TASKS = {
    "nurse": ["checks the chart", "greets the patient", "prepares the room", "updates the chart"],
    "doctor": ["reads the chart", "reviews the results", "signs the form"],
    "clerk": ["books the appointment", "prints the form", "updates the schedule"],
    "pharmacist": ["fills the prescription", "checks the prescription", "signs the form"],
}


def date(rng):
    """A made-up date in ISO form, for example 2026-03-14."""
    return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def line(rng, drugs):
    """One invented line, picked from five templates."""
    kind = rng.randint(0, 4)
    if kind == 0:
        return f"Room {rng.randint(1, 40)}: {rng.choice(drugs)} {rng.choice(DOSES)} mg {rng.choice(HOW_OFTEN)} from {date(rng)}."
    if kind == 1:
        role = rng.choice(ROLES)
        return f"The {role} {rng.choice(TASKS[role])} for Mr. {rng.choice(SURNAMES)} in room {rng.randint(1, 40)}."
    if kind == 2:
        return f"The clerk booked a follow-up visit at the {rng.choice(WARDS)} clinic on {date(rng)}."
    if kind == 3:
        return f"The pharmacist filled the prescription for {rng.choice(drugs)} {rng.choice(DOSES)} mg for Ms. {rng.choice(SURNAMES)}."
    return f"Ward {rng.randint(1, 9)} has {rng.randint(0, 6)} free beds today and {rng.randint(1, 12)} patients on the {rng.choice(WARDS)} list."


ENGLISH = (
    "Discharge letter. Mr. Anders Berg, born 12 May 1961, was admitted to the cardiology ward on 3 March 2026 with chest pain. "
    "The tests showed no heart attack. He received aspirin 100 mg once daily and metoprolol 50 mg twice daily. "
    "His blood pressure improved after two days. He may go home today. "
    "Please book a follow-up visit with the family doctor in two weeks, and call the clinic if the pain returns."
)
GERMAN = (
    "Entlassungsbrief. Herr Anders Berg, geboren am 12. Mai 1961, wurde am 3. März 2026 wegen Brustschmerzen auf der Kardiologiestation aufgenommen. "
    "Die Untersuchungen zeigten keinen Herzinfarkt. Er erhielt einmal täglich Aspirin 100 mg und zweimal täglich Metoprolol 50 mg. "
    "Sein Blutdruck verbesserte sich nach zwei Tagen. Er darf heute nach Hause. "
    "Bitte vereinbaren Sie in zwei Wochen einen Kontrolltermin beim Hausarzt und rufen Sie die Klinik an, wenn die Schmerzen wiederkehren."
)
PYTHON = '''def discharge_letter(patient):
    """Write the discharge letter for one patient."""
    medicines = [("aspirin", 100, 1), ("metoprolol", 50, 2)]
    lines = [f"Discharge letter. {patient.name}, born {patient.born}, was admitted on {patient.admitted}."]
    for name, mg, times in medicines:
        lines.append(f"{name} {mg} mg, {times} times daily")
    if patient.blood_pressure_improved:
        lines.append("Blood pressure improved after two days. May go home today.")
    lines.append("Book a follow-up visit with the family doctor in two weeks.")
    return "\\n".join(lines)
'''


def main():
    DATA.mkdir(exist_ok=True)
    rng = random.Random(4)
    train = [line(rng, DRUGS_SEEN) for _ in range(2000)]
    test = [line(rng, DRUGS_SEEN + DRUGS_NEW) for _ in range(200)]
    (DATA / "clinical_train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (DATA / "clinical_test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    (DATA / "letters.txt").write_text(
        "### ENGLISH\n" + ENGLISH + "\n### GERMAN\n" + GERMAN + "\n### PYTHON\n" + PYTHON, encoding="utf-8")
    print(f"wrote {len(train)} training lines, {len(test)} test lines and the three letters to {DATA}")


if __name__ == "__main__":
    main()
