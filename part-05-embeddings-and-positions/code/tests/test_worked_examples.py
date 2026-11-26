"""Run worked_examples.py and check the key numbers it prints."""
import contextlib
import io

import worked_examples


def test_worked_examples_print_the_video_numbers():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        worked_examples.main()
    out = buffer.getvalue()
    for expected in ("table size: 6 x 3 = 18 numbers", "row 3 = [0.2, 0.8, 0.7]", "30,522 x 384 = 11,720,448", "dot = 1*2 + 2*1 + 0*0 = 4",
                     "cosine = 4 / (sqrt(5) * sqrt(5)) = 4 / 5 = 0.80", "a with itself: dot = 5, cosine = 5 / 5 = 1.00",
                     "'dog bites man' sum = [3, 4, 3]   'man bites dog' sum = [3, 4, 3]  (identical)",
                     "dog at position 1 = [2, 0, 2], dog at position 3 = [1, 0, 3]", "score(q at 3, k at 1) = 2.732", "score(q at 5, k at 3) = 2.732",
                     "position 6000 becomes 3000", "20,452"):
        assert expected in out, expected
