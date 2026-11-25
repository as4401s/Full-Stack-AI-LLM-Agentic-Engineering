"""Demo 3 - text is bytes: how many bytes does each character take in UTF-8?

What it does: prints the UTF-8 bytes of some letters, symbols and words so we can see why byte-level tokenizers
start from just 256 possible values.
What it needs: only the Python standard library.
What to expect: 'e' is one byte; 'ü' and 'é' are two; the euro sign is three; an emoji is four.
Run it: uv run python -m demos.d03_bytes_utf8
"""
from tokenizer_lab.report import run_and_save, save_results


def main():
    print(f"{'text':14s}{'characters':>11s}{'bytes':>7s}   bytes in hex")
    rows = []
    for text in ["e", "é", "ü", "€", "日", "😀", "ward", "Größe", "Herzinfarkt", "Kardiologiestation", "paracetamol"]:
        data = text.encode("utf-8")
        rows.append((text, len(text), len(data)))
        shown = data.hex(" ")
        print(f"{text:14s}{len(text):11d}{len(data):7d}   {shown if len(shown) < 45 else shown[:42] + '...'}")
    print("\nevery byte is a number from 0 to 255, so a vocabulary of 256 pieces can write any text in any language")
    print("decoding goes back the same way:", bytes.fromhex("c3 bc").decode("utf-8"), "=", "c3 bc")
    print("a broken sequence cannot be read:", bytes.fromhex("c3").decode("utf-8", errors="replace"), "(c3 alone is half of a character)")
    save_results("d03_bytes_utf8", {"rows": rows})


if __name__ == "__main__":
    run_and_save("d03_bytes_utf8", main)
