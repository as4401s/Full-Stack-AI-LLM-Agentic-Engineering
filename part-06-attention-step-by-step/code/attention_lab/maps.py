"""maps.py - draw an attention map as a picture.

What it does
    Draws a table of weights as a heat map: rows are the tokens that listen, columns are the tokens listened to, colour is the weight.
    The numbers are written inside the cells, so the picture can be read without a colour key.

What it needs
    matplotlib (optional: when it is missing the function prints a note and the demos carry on).

What to expect
    plot_map(weights, tokens, path, title) writes a PNG and returns True; it returns False when matplotlib is missing.
"""


def plot_map(weights, tokens, path, title="", show_numbers=True):
    """Save an attention map. weights: (n, n) array or tensor, tokens: list of n strings, path: PNG file to write."""
    try:
        import matplotlib
        matplotlib.use("Agg")                                     # draw to a file, no window needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib is not installed: skipping the picture)")
        return False
    if hasattr(weights, "detach"):
        weights = weights.detach().cpu().numpy()
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * n, 1.0 + 0.7 * n))
    ax.imshow(weights, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n), tokens, rotation=45, ha="right")
    ax.set_yticks(range(n), tokens)
    ax.set_xlabel("listened to (columns)")
    ax.set_ylabel("listening (rows)")
    ax.set_title(title)
    if show_numbers:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if weights[i, j] > 0.55 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
