"""worked_examples.py - reproduces every small number calculated by hand in lesson 5.

What it does
    Recomputes, with NumPy only, each calculation we do on a calculator in the video: a lookup in a tiny table, the dot product and cosine of
    [1, 2, 0] and [2, 1, 0], why a bag of tokens cannot tell "dog bites man" from "man bites dog", the sinusoidal position codes, a rotation
    that keeps a vector's length and makes the score depend only on the distance, and two ways of stretching a context window.

What it needs
    NumPy.

What to expect
    Run:  python worked_examples.py
    Every line printed here is a number you can also get on a calculator. The saved output is in data/expected_output/worked_examples.txt.
"""
import numpy as np

from embedding_lab import positions, rope, vectors

TOKENS = ["the", "patient", "has", "kidney", "failure", "."]
TABLE = np.array([[0.1, 0.0, 0.2], [0.9, 0.3, 0.1], [0.2, 0.1, 0.0], [0.2, 0.8, 0.7], [0.1, 0.6, 0.9], [0.0, 0.0, 0.1]])


def lookup():
    """A table of six tokens with three numbers each: token id 3 picks row 3."""
    print("== 1. The table is a lookup (6 tokens x 3 numbers)")
    print(f"  table size: {TABLE.shape[0]} x {TABLE.shape[1]} = {TABLE.size} numbers")
    print(f"  token id 3 is {TOKENS[3]!r}: row 3 = {TABLE[3].tolist()}")
    one_hot = np.zeros(6)
    one_hot[3] = 1
    print(f"  one-hot [0,0,0,1,0,0] times the table = {(one_hot @ TABLE).tolist()}  (the same row)")
    print(f"  real sizes: 30,522 x 384 = {30522 * 384:,}   50,257 x 768 = {50257 * 768:,}")


def dot_and_cosine():
    """Dot product, lengths and cosine of a = [1, 2, 0] and b = [2, 1, 0]."""
    a, b = np.array([1.0, 2.0, 0.0]), np.array([2.0, 1.0, 0.0])
    print("\n== 2. Dot product and cosine of a = [1, 2, 0] and b = [2, 1, 0]")
    print(f"  dot = 1*2 + 2*1 + 0*0 = {vectors.dot(a, b):.0f}")
    print(f"  length of a = sqrt(1 + 4 + 0) = sqrt(5) = {vectors.length(a):.3f}; length of b = sqrt(5) = {vectors.length(b):.3f}")
    print(f"  cosine = 4 / (sqrt(5) * sqrt(5)) = 4 / 5 = {vectors.cosine(a, b):.2f}")
    print(f"  angle = {np.degrees(np.arccos(vectors.cosine(a, b))):.1f} degrees")
    print(f"  a with itself: dot = {vectors.dot(a, a):.0f}, cosine = 5 / 5 = {vectors.cosine(a, a):.2f}")
    print(f"  a with 2a = [2, 4, 0]: cosine = {vectors.cosine(a, 2 * a):.2f}, but the straight-line distance is {vectors.euclidean(a, 2 * a):.3f} (a to b: {vectors.euclidean(a, b):.3f})")
    print(f"  right angle [1,0,0] and [0,1,0]: cosine = {vectors.cosine([1, 0, 0], [0, 1, 0]):.0f}; opposite a and -a: cosine = {vectors.cosine(a, -a):.0f}")


def bag_of_tokens():
    """A model that only adds up its tokens cannot tell the two sentences apart; adding a position vector to each token can."""
    dog, bites, man = np.array([1, 0, 2]), np.array([0, 3, 1]), np.array([2, 1, 0])
    pos = np.eye(3, dtype=int)                                      # position 1 -> [1,0,0], position 2 -> [0,1,0], position 3 -> [0,0,1]
    print("\n== 3. A bag of tokens: dog = [1, 0, 2], bites = [0, 3, 1], man = [2, 1, 0]")
    print(f"  'dog bites man' sum = {(dog + bites + man).tolist()}   'man bites dog' sum = {(man + bites + dog).tolist()}  (identical)")
    print(f"  with position vectors added: dog at position 1 = {(dog + pos[0]).tolist()}, dog at position 3 = {(dog + pos[2]).tolist()}  (different now)")
    print(f"  man at position 3 = {(man + pos[2]).tolist()}, man at position 1 = {(man + pos[0]).tolist()}")


def sinusoidal():
    """The first four position codes of a table with four numbers per position."""
    table = positions.sinusoidal_table(4, 4)
    print("\n== 4. Sinusoidal codes with 4 numbers per position (two turning pairs: 1 radian and 0.01 radian per position)")
    for p, row in enumerate(table):
        print(f"  position {p}: {np.round(row, 3).tolist()}")
    print(f"  dot(position 0, position 1) = cos(1) + cos(0.01) = {np.cos(1) + np.cos(0.01):.4f}   from the table: {float(table[0] @ table[1]):.4f}")
    print(f"  dot(position 1, position 2) = {float(table[1] @ table[2]):.4f}   dot(position 2, position 3) = {float(table[2] @ table[3]):.4f}  (same distance, same dot product)")


def rotation():
    """One pair rotated by 30 degrees per position: the length stays, and the score depends only on the distance."""
    step = np.radians(30)

    def turn(v, position):
        a = position * step
        return np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)])

    q, k = np.array([2.0, 0.0]), np.array([1.0, 1.0])
    print("\n== 5. Rotation by 30 degrees per position: q = [2, 0], k = [1, 1]")
    print(f"  q at position 3 turns 90 degrees: {np.round(turn(q, 3), 3).tolist()}; its length stays {np.linalg.norm(turn(q, 3)):.3f} (before: {np.linalg.norm(q):.3f})")
    print(f"  k at position 1 turns 30 degrees: {np.round(turn(k, 1), 3).tolist()}")
    print(f"  score(q at 3, k at 1) = {float(turn(q, 3) @ turn(k, 1)):.3f}   distance 2")
    print(f"  score(q at 5, k at 3) = {float(turn(q, 5) @ turn(k, 3)):.3f}   distance 2 again, same score")
    print(f"  score(q at 4, k at 1) = {float(turn(q, 4) @ turn(k, 1)):.3f}   distance 3, a different score")
    print(f"  the library function rope.rotate keeps lengths too: {np.linalg.norm(rope.rotate(np.array([3.0, 4.0, 1.0, 2.0]), 9)):.6f} vs {np.linalg.norm([3.0, 4.0, 1.0, 2.0]):.6f}")


def stretching():
    """Two simple ways of fitting a longer text into a window the model was trained for."""
    print("\n== 6. Stretching a window trained for 4,096 tokens to 8,192")
    print(f"  position interpolation: scale = 4096 / 8192 = {4096 / 8192}; position 6000 becomes {float(rope.scale_positions(6000, 4096, 8192)):.0f}; position 8191 becomes {float(rope.scale_positions(8191, 4096, 8192)):.1f}")
    print(f"  NTK-style base for 64 numbers per head and a stretch of 2: 10000 x 2^(64/62) = {rope.ntk_scaled_base(10000.0, 64, 2.0):,.0f}")


def main():
    lookup()
    dot_and_cosine()
    bag_of_tokens()
    sinusoidal()
    rotation()
    stretching()


if __name__ == "__main__":
    main()
