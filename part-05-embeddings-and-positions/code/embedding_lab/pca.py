"""pca.py - squash many numbers per word into two, and check that the squashed map still groups related words.

What it does
    pca_2d() finds the two directions along which the words spread out most (principal components, through a singular value
    decomposition) and returns each word's position along them. neighbour_purity() asks, for every word, whether its closest
    neighbour belongs to the same group.
What it needs
    NumPy.
What to expect
    The signs of a principal component are arbitrary, so we fix them: the largest number in each component is made positive.
    That keeps the saved map identical from run to run.
"""
import numpy as np


def pca_2d(matrix):
    """Return (coords, explained): coords has shape (rows, 2); explained is the share of the spread each component keeps."""
    centred = matrix - matrix.mean(axis=0)
    _, singular, components = np.linalg.svd(centred, full_matrices=False)
    for row in components:                                       # fix the arbitrary sign of every component
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    explained = singular ** 2 / np.sum(singular ** 2)
    return centred @ components[:2].T, explained


def neighbour_purity(matrix, labels, metric="cosine"):
    """Share of words whose nearest neighbour has the same label (1.0 means every nearest neighbour is a group mate).

    metric="cosine" compares directions (used for the full table); metric="euclid" compares straight-line distances (used for the 2-D map).
    """
    if metric == "cosine":
        unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        closeness = unit @ unit.T                                   # larger = closer
    else:
        closeness = -np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    np.fill_diagonal(closeness, -np.inf)                            # a word is not its own neighbour
    nearest = closeness.argmax(axis=1)
    return float(np.mean([labels[i] == labels[j] for i, j in enumerate(nearest)]))
