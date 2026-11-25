"""Run worked_examples.py and check the key numbers it prints."""
import contextlib
import io

import worked_examples


def test_worked_examples_print_the_video_numbers():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        worked_examples.main()
    out = buffer.getvalue()
    for expected in ("e+d: 18", "merge 5: war + d -> ward   (count 8)", "pieces: 11 letters + 5 merges = 16", "warded: ward + ed",
                     "two bytes: 11000011 10111100 = c3 bc", "4158 tokens / 2308 words = 1.80 tokens per word", "total $54.00", "total $60.00, which is $6.00 more per day",
                     "= 6,841 tokens left for notes", "so 357 notes fit"):
        assert expected in out, expected
