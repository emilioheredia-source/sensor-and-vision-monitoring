"""Matplotlib setup shared by the scripts.

Interactive by default: figures pop up in Qt windows. Set HEADLESS=1 to fall
back to file-only output (useful when running over a remote shell or in CI).

Figures are always saved to <project>/figures/ as well, so a run leaves a
record behind whether or not anyone was watching.
"""

import os
from pathlib import Path

import matplotlib

INTERACTIVE = os.environ.get("HEADLESS", "") not in ("1", "true", "True")

if INTERACTIVE:
    try:
        matplotlib.use("QtAgg")
    except Exception:            # Qt missing or no display available
        matplotlib.use("Agg")
        INTERACTIVE = False
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def save(fig, name: str) -> Path:
    """Save a figure into figures/ and report where it went."""
    path = FIG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"saved {path}")
    return path


def show():
    """Block on open figure windows, if we have any."""
    if INTERACTIVE:
        print(f"backend: {matplotlib.get_backend()} - close the window(s) to continue")
        plt.show()
    else:
        print("headless: figures written to files only")
