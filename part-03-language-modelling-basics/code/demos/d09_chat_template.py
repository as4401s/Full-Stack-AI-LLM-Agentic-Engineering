"""Demo 9 - write a conversation as one piece of text, the way a chat model receives it.

What it does: formats a one-question chat and a three-message chat with our template, shows the same messages as the
list of roles and contents that chat libraries use, and shows what happens when a role name is wrong.
What it needs: only the Python standard library.
What to expect: each message starts with a role marker and ends with <|end|>; the text finishes with <|assistant|>,
the cue for the model to write its reply.
Run it: uv run python -m demos.d09_chat_template
"""
from lm_basics.chat_template import format_chat
from lm_basics.report import run_and_save, save_results


def main():
    one = [{"role": "system", "content": "You are CareDesk, a helpful front-desk assistant."},
           {"role": "user", "content": "When does the clinic open?"}]
    print("messages as a list of roles and contents:")
    for m in one:
        print("  ", m)
    print("\nthe same chat as one piece of text:")
    text = format_chat(one)
    print(text)
    three = one + [{"role": "assistant", "content": "The clinic opens at eight in the morning."},
                   {"role": "user", "content": "And on Saturdays?"}]
    print("a longer chat, ready for the model to write the next reply:")
    print(format_chat(three))
    print("the same chat without the final cue (used when training on finished conversations):")
    print(format_chat(three, add_generation_prompt=False))
    try:
        format_chat([{"role": "robot", "content": "hello"}])
    except ValueError as error:
        print("a wrong role name is reported:", error)
    save_results("d09_chat_template", {"one_question": text})


if __name__ == "__main__":
    run_and_save("d09_chat_template", main)
