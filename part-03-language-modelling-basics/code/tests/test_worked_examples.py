"""Run worked_examples.py and check the key numbers it prints."""
import io
import contextlib

import worked_examples


def test_worked_examples_print_the_video_numbers():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        worked_examples.main()
    out = buffer.getvalue()
    for expected in ("P(nurse | the) = 2/6 = 0.333", "1/18 = 0.0556", "a roll of 0.52 lands in the slice of 'chart'",
                     "(0 + 1) / (1 + 9) = 0.1", "[0.659, 0.242, 0.099]", "perplexity = e^1.213 = 3.364",
                     "e^2.0 = 7.389", "[0.864, 0.117, 0.019]", "[0.502, 0.304, 0.194]", "[0.731, 0.269, 0.0]", "[1.0, 0.0, 0.0]", "[0.368, 0.316, 0.211, 0.105, 0.0]"):
        assert expected in out, expected
