"""Shared matplotlib style for publication-quality financial figures.

Professional financial color palette — apply uniformly across ALL figures.
"""
import matplotlib as mpl

# Force the non-interactive Agg backend before pyplot is imported. Without this, if
# this module loads before reporting.charts, matplotlib binds the default (Tk) backend
# and the batch pipeline crashes at teardown with "Tcl_AsyncDelete: async handler
# deleted by the wrong thread" on Windows.
mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# ── Professional Financial Palette ────────────────────────────────────────────
PALETTE = {
    "navy":        "#1B3A6B",   # Primary Blue  — titles, main lines
    "blue":        "#2E6DB4",   # Secondary Blue — primary bars, fills
    "gold":        "#C9A84C",   # Accent Gold   — secondary bars, reference lines
    "gray":        "#6B7280",   # Neutral Gray  — random/reference lines, low SHAP
    "light_gray":  "#F3F4F6",   # Background fills
    "grid":        "#E5E7EB",   # Horizontal gridlines
    "red":         "#B91C1C",   # Alert Red     — stressed/bad/60-month
    "green":       "#15803D",   # Success Green — good/performing/36-month
}

# Convenience aliases
C_NAVY  = PALETTE["navy"]
C_BLUE  = PALETTE["blue"]
C_GOLD  = PALETTE["gold"]
C_GRAY  = PALETTE["gray"]
C_RED   = PALETTE["red"]
C_GREEN = PALETTE["green"]
C_GRID  = PALETTE["grid"]

PUBLICATION_RC = {
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "font.family":          "DejaVu Sans",
    "font.size":            11,
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.labelsize":       12,
    "axes.labelcolor":      C_NAVY,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":      10,
    "legend.framealpha":    0.9,
    "legend.loc":           "best",
    "lines.linewidth":      2.0,
    "lines.markersize":     5,
    "lines.markerfacecolor": "white",
    "lines.markeredgewidth": 1.5,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.spines.left":     True,
    "axes.spines.bottom":   True,
    "axes.grid":            True,
    "axes.grid.axis":       "y",
    "grid.color":           C_GRID,
    "grid.linewidth":       0.6,
    "axes.prop_cycle":      mpl.cycler(color=[C_BLUE, C_GOLD, C_NAVY, C_GREEN,
                                               C_RED, C_GRAY]),
}


def apply_publication_style() -> None:
    """Apply global matplotlib rcParams for publication-ready financial figures."""
    plt.rcParams.update(PUBLICATION_RC)


def despine(ax: "plt.Axes") -> None:
    """Remove top/right spines and set grid style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
