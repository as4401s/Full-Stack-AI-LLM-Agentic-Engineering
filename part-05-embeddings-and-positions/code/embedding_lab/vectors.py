"""vectors.py - dot product, length, cosine and nearest neighbours with plain NumPy.

What it does
    The few vector operations the lesson uses: dot product, length (norm), cosine similarity, Euclidean distance, and a helper
    that lists the words closest to a given word.
What it needs
    NumPy.
What to expect
    cosine([1, 2, 0], [2, 1, 0]) is 0.8; the cosine of any vector with itself is 1.
"""
import numpy as np


def dot(a, b):
    """Dot product: multiply the numbers pair by pair and add them up."""
    return float(np.dot(a, b))


def length(a):
    """Length of a vector (its norm): the square root of the dot product of the vector with itself."""
    return float(np.sqrt(np.dot(a, a)))


def cosine(a, b):
    """Cosine similarity: the dot product divided by the two lengths. It reads 1 for the same direction, 0 for right angles, -1 for opposite."""
    return dot(a, b) / (length(a) * length(b))


def euclidean(a, b):
    """Straight-line distance between the tips of two vectors."""
    return length(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def normalise_rows(matrix):
    """Scale every row to length 1, so that a plain dot product between two rows equals their cosine."""
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def nearest(word, words, index, matrix, k=5, exclude=()):
    """Return the k words whose rows have the highest cosine with the row of `word`, as [(word, cosine), ...].

    `exclude` lists words to skip (for analogies the input words are skipped so the answer cannot be one of them).
    """
    unit = normalise_rows(matrix)
    scores = unit @ unit[index[word]]
    skip = {word, *exclude}
    order = [i for i in np.argsort(-scores) if words[i] not in skip]
    return [(words[i], float(scores[i])) for i in order[:k]]
