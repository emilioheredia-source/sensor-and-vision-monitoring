"""Every figure goes through here, so they all look the same.

The notebooks and the workbench take their colours from this file, and the notebooks call
style() before they draw. There's one ready-made figure here too, profile_figure(), which
draws the concentrations stage by stage down the circuit.

Rules followed (dataviz skill, reference palette):
- one axis per panel, never two y-scales;
- each component keeps its own colour everywhere (colour follows the entity, not its rank);
- thin lines;
- direct labels instead of a legend where there are four series or fewer;
- recessive grid and axes;
- text in ink colours, never in a series colour.

Usage:

    from src.plotting import COLOR, FIGURES, style
    style()
    ax.plot(t, purity, color=COLOR["Nd"])
    fig.savefig(FIGURES / "my_figure.png")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Where the report's figures are written.
FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---- colours -------------------------------------------------------------------------------
# Categorical slots 1-4 of the reference palette, fixed to the components, so La+Ce is the same
# blue in every figure.
COLOR = {"LaCe": "#2a78d6", "Pr": "#eb6834", "Nd": "#1baf7a", "SmPlus": "#eda100"}
# The palette's slots in order, for series that aren't components (two scrub modes, say).
# The first four are the component colours; the last two are extra.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
# Good and bad, for pass/fail marks.
STATUS_GOOD, STATUS_CRITICAL = "#0ca30c", "#d03b3b"

# The background (SURFACE), text from strongest to weakest (INK, INK2, MUTED), and the grid
# and axis lines (GRID, AXIS).
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


# ---- the shared look -------------------------------------------------------------------------

def style():
    """Set matplotlib's defaults to the project's look. Call it before making a figure.

    It changes matplotlib's global settings (rcParams) for the rest of the Python session.
    """
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def section_bands(ax, cmap, y_text=None):
    """Mark the three sections (extraction, scrub, strip) along a stage axis.

    Each section's name goes just above the plot, centred on it, and the scrub section gets a
    light grey band so the boundaries show. A section with no stages is skipped.

    Args:
        ax: the matplotlib axes, with stage numbers (from 0) on the x axis.
        cmap: the circuit's connection map (circuit.cmap): n_ext, n_scrub and N.
        y_text: not used; kept so existing calls still work.
    """
    bounds = [(0, cmap.n_ext, "extraction"), (cmap.n_ext, cmap.n_ext + cmap.n_scrub, "scrub"),
              (cmap.n_ext + cmap.n_scrub, cmap.N, "strip")]
    for k, (first, end, name) in enumerate(bounds):
        if end <= first:
            continue                                   # no stages in this section
        if k % 2 == 1:                                 # every other section shaded: the scrub
            ax.axvspan(first - 0.5, end - 0.5, color=GRID, alpha=0.5, lw=0)
        ax.text((first + end - 1) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=MUTED, fontsize=8)


# ---- a ready-made figure: the concentration profile ------------------------------------------

def profile_figure(circuit, x, title: str, path: str | Path | None = None,
                   components=("Pr", "Nd")):
    """Concentration down the train: organic, aqueous, acid. One axis each.

    Args:
        circuit: the circuit.Circuit the state belongs to.
        x: the state vector to draw.
        title: the figure title.
        path: where to save it; None doesn't save.
        components: which components to draw on the organic and aqueous panels.

    Returns:
        The matplotlib figure: three panels sharing the stage axis (0 = raffinate end).

    Why five parameters: what to draw (circuit, state), its title, where to save it, and which elements.
    """
    style()
    p, cmap = circuit.p, circuit.cmap
    c_aq, c_org, h = _stage_arrays(circuit, x)
    stages = np.arange(circuit.N)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.7], "hspace": 0.35})
    # the two concentration panels: one line per component, labelled at its peak
    for ax, data, ylabel in ((axes[0], c_org, "organic, mol/L"), (axes[1], c_aq, "aqueous, mol/L")):
        for name in components:
            _plot_labelled(ax, stages, data[:, p.index(name)], name)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.15 * float(data[:, [p.index(c) for c in components]].max()))
        section_bands(ax, cmap)
    # the acid panel, on a log scale
    axes[2].plot(stages, h, color=INK2)
    axes[2].set_ylabel("acid, mol/L")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("stage (0 = raffinate end)")
    section_bands(axes[2], cmap)
    axes[0].set_title(title, loc="left", color=INK, pad=16)
    fig.align_ylabels(axes)
    if path:
        fig.savefig(path)
    return fig


def _stage_arrays(circuit, x):
    """The state as plain arrays: aqueous and organic (stage x component, mol/L) and acid
    (one per stage, mol/L)."""
    c_aq, c_org, h, _ = circuit.unpack(x)
    c_aq = np.array([np.asarray(v, float).ravel() for v in c_aq])
    c_org = np.array([np.asarray(v, float).ravel() for v in c_org])
    h = np.array([float(v) for v in h])
    return c_aq, c_org, h


def _plot_labelled(ax, stages, values, name: str):
    """One component's line in its own colour, with its name written at the line's peak.

    Labelling at each line's own peak keeps the labels apart; at the ends they'd collide.
    """
    ax.plot(stages, values, color=COLOR[name])
    k = int(np.argmax(values))
    ax.annotate(name, (stages[k], values[k]), xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", color=INK2, fontsize=8)
