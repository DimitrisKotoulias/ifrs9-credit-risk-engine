"""Exploratory Data Analysis for Lending Club data.

Produces and saves diagnostic figures to reports/figures/eda/.
All figures are saved as PNG for inclusion in the PDF report.

Color palette: professional financial palette (see reporting.style).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from credit_risk.data.split import DataSplit, parse_issue_date
from credit_risk.data.target import TARGET_COL
from credit_risk.reporting.style import (
    apply_publication_style,
    despine,
    C_NAVY, C_BLUE, C_GOLD, C_GRAY, C_RED, C_GREEN, C_GRID,
)

logger = logging.getLogger(__name__)

_FIG_DIR = Path("reports/figures/eda")


def _savefig(fig: plt.Figure, name: str, fig_dir: Path = _FIG_DIR) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved EDA figure: %s", path)
    return path


def plot_default_rate_by_grade(df: pd.DataFrame, fig_dir: Path = _FIG_DIR) -> Path:
    """Bar chart: default rate (%) by loan grade — financial palette."""
    apply_publication_style()
    rates = df.groupby("grade")[TARGET_COL].agg(["mean", "count"]).reset_index()
    rates.columns = ["grade", "default_rate", "count"]
    rates = rates.sort_values("grade")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Gradient colors from green (A) to red (G)
    import matplotlib.cm as cm
    n_grades = len(rates)
    norm_vals = np.linspace(0.85, 0.15, n_grades)
    colors = [cm.RdYlGn(v) for v in norm_vals]
    
    bars = ax.bar(rates["grade"], rates["default_rate"] * 100,
                  color=colors, alpha=0.88, width=0.6)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("Loan Grade", fontsize=12, labelpad=8)
    ax.set_ylabel("Default Rate (%)", fontsize=12, labelpad=8)
    ax.set_title("Observed Default Rate by Underwriting Grade", fontsize=13, pad=12)
    despine(ax)

    # Portfolio average line
    portfolio_mean = df[TARGET_COL].mean()
    ax.axhline(portfolio_mean * 100, color=C_GOLD, linewidth=2.0, linestyle="--",
               label=f"Portfolio Mean ({portfolio_mean:.2%})")

    for bar, (dr, cnt) in zip(bars, zip(rates["default_rate"], rates["count"])):
        # Value label on top
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{dr:.2%}",
            ha="center", va="bottom", fontsize=9, color=C_NAVY, fontweight="bold",
        )
        # Sample size below x-axis using offset points to avoid grade overlaps
        ax.annotate(
            f"n={cnt:,}",
            xy=(bar.get_x() + bar.get_width() / 2, 0),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center", va="top", fontsize=8, color=C_GRAY,
        )

    ax.margins(y=0.15)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    return _savefig(fig, "default_rate_by_grade", fig_dir)


def plot_vintage_default_curves(df: pd.DataFrame, fig_dir: Path = _FIG_DIR) -> Path:
    """Line chart: default rate by vintage year with GFC highlight."""
    apply_publication_style()
    issue_dt = parse_issue_date(df)
    df2 = df.copy()
    df2["vintage_year"] = issue_dt.dt.year
    rates = df2.groupby("vintage_year")[TARGET_COL].agg(["mean", "count"]).reset_index()
    rates.columns = ["vintage_year", "default_rate", "count"]
    rates = rates.sort_values("vintage_year")

    mean_dr = rates["default_rate"].mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # GFC shaded region (2007-2009)
    ax.axvspan(2007, 2009, color="#E5E7EB", alpha=0.5, label="GFC Stressed Period (2007-2009)")

    # Plot the curve
    ax.plot(
        rates["vintage_year"], rates["default_rate"] * 100,
        marker="o", color=C_NAVY, linewidth=2.5,
        markerfacecolor="white", markeredgewidth=2.0, markersize=6,
        label="Observed Default Rate"
    )
    ax.fill_between(rates["vintage_year"], rates["default_rate"] * 100,
                    alpha=0.15, color=C_BLUE)

    # Mean reference line
    ax.axhline(mean_dr * 100, color=C_GOLD, linewidth=2.0,
               linestyle="--", label=f"Portfolio Mean ({mean_dr:.2%})")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("Vintage Year (Loan Origination)", fontsize=12, labelpad=8)
    ax.set_ylabel("Observed Default Rate (%)", fontsize=12, labelpad=8)
    ax.set_title("Portfolio Default Rate by Origination Vintage Year", fontsize=13, pad=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    despine(ax)
    ax.set_xticks(rates["vintage_year"])
    ax.grid(True, axis="both", color=C_GRID, linewidth=0.6)
    fig.tight_layout()
    return _savefig(fig, "vintage_default_curves", fig_dir)


def plot_default_rate_by_term(df: pd.DataFrame, fig_dir: Path = _FIG_DIR) -> Path:
    """Horizontal bar chart: default rate by loan term (36=green, 60=red)."""
    if "term" not in df.columns:
        logger.warning("'term' column not found; skipping term default chart.")
        return fig_dir / "default_rate_by_term.png"

    apply_publication_style()
    rates_df = df.groupby("term")[TARGET_COL].agg(["mean", "count"]).reset_index()
    rates_df.columns = ["term", "dr", "count"]
    
    labels = [f"{int(t)} Months" for t in rates_df["term"]]
    values = (rates_df["dr"] * 100).tolist()
    colors = [C_GREEN if t == 36 else C_RED for t in rates_df["term"]]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(labels, values, color=colors, alpha=0.88, height=0.45)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("Default Rate (%)", fontsize=12, labelpad=8)
    ax.set_title("Default Rate by Loan Amortisation Term", fontsize=13, pad=12)
    despine(ax)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.6)
    ax.grid(False, axis="y")

    # Add counts and percentages inside the bars
    for bar, dr, cnt in zip(bars, rates_df["dr"], rates_df["count"]):
        x_pos = bar.get_width() * 0.95
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{dr:.2%} (n={cnt:,})",
            va="center", ha="right", fontsize=10, color="white", fontweight="bold"
        )
    ax.margins(x=0.15)
    fig.tight_layout()
    return _savefig(fig, "default_rate_by_term", fig_dir)


def plot_default_rate_by_purpose(df: pd.DataFrame, fig_dir: Path = _FIG_DIR) -> Path:
    """Horizontal bar chart: DR by purpose, RdYlGn_r colormap + mean vline."""
    apply_publication_style()
    rates = (
        df.groupby("purpose")[TARGET_COL]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "dr", "count": "n"})
        .sort_values("dr", ascending=True)
    )
    mean_dr = df[TARGET_COL].mean() * 100

    # Map DR to color via reversed RdYlGn (low DR=green, high DR=red)
    norm_vals = (rates["dr"] - rates["dr"].min()) / (rates["dr"].max() - rates["dr"].min() + 1e-9)
    colors = [cm.RdYlGn_r(v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(rates) * 0.4)))
    bars = ax.barh(rates["purpose"], rates["dr"] * 100, color=colors, alpha=0.88, height=0.6)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("Default Rate (%)", fontsize=12, labelpad=8)
    ax.set_title("Observed Default Rate by Loan Purpose", fontsize=13, pad=12)
    despine(ax)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.6)
    ax.grid(False, axis="y")

    # Mean vertical reference line
    ax.axvline(mean_dr, color=C_GOLD, linewidth=2.0, linestyle="--",
               label=f"Portfolio Mean ({mean_dr/100:.2%})")
    
    # Add value labels next to bars
    for bar, dr in zip(bars, rates["dr"]):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{dr:.2%}",
            va="center", ha="left", fontsize=9, color=C_NAVY, fontweight="bold"
        )
        
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.margins(x=0.15)
    fig.tight_layout()
    return _savefig(fig, "default_rate_by_purpose", fig_dir)


def plot_missingness(df: pd.DataFrame, top_n: int = 20, fig_dir: Path = _FIG_DIR) -> Path:
    """Bar chart: top-N columns by missingness rate colored by threshold tiers."""
    apply_publication_style()
    miss_pct = df.isnull().mean() * 100
    miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=False).head(top_n)

    if miss_pct.empty:
        logger.info("No missing values found; skipping missingness chart.")
        return fig_dir / "missingness.png"

    # Color mapping by threshold
    colors = []
    for val in miss_pct.values:
        if val > 50.0:
            colors.append(C_RED)
        elif val > 20.0:
            colors.append(C_GOLD)
        else:
            colors.append(C_BLUE)

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(miss_pct) * 0.35)))
    bars = ax.barh(miss_pct.index, miss_pct.values, color=colors, alpha=0.88, height=0.6)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("Missing Rate (%)", fontsize=12, labelpad=8)
    ax.set_title("Missing Data Density Analysis (Top Missing Fields)", fontsize=13, pad=12)
    
    # Add threshold lines
    ax.axvline(20.0, color=C_GOLD, linestyle="--", linewidth=1.5, label="20% Threshold")
    ax.axvline(50.0, color=C_RED, linestyle="--", linewidth=1.5, label="50% Threshold")
    ax.legend(loc="lower right", fontsize=10)

    # Add value labels
    for bar, val in zip(bars, miss_pct.values):
        ax.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", ha="left", fontsize=9, color=C_NAVY, fontweight="bold"
        )

    despine(ax)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    ax.margins(x=0.12)
    fig.tight_layout()
    return _savefig(fig, "missingness", fig_dir)


def plot_target_distribution(df: pd.DataFrame, fig_dir: Path = _FIG_DIR) -> Path:
    """Horizontal bar chart of good/bad ratio with labels inside."""
    apply_publication_style()
    counts = df[TARGET_COL].value_counts()
    n_good = int(counts.get(0, 0))
    n_bad  = int(counts.get(1, 0))
    total  = n_good + n_bad

    labels  = ["Good (0)", "Bad (1)"]
    values  = [n_good, n_bad]
    colors  = [C_GREEN, C_RED]
    pcts    = [n_good / total, n_bad / total]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    bars = ax.barh(labels, values, color=colors, alpha=0.88, height=0.5)
    ax.set_xlabel("Count", fontsize=12, labelpad=8)
    despine(ax)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.6)
    ax.grid(False, axis="y")

    for bar, count, pct in zip(bars, values, pcts):
        # If the bar width is less than 30% of total, place the label outside the bar to prevent overlap
        if bar.get_width() < total * 0.30:
            x_pos = bar.get_width() + total * 0.02
            ha_align = "left"
            text_color = C_NAVY
        else:
            x_pos = bar.get_width() - total * 0.02
            ha_align = "right"
            text_color = "white"
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.2%})",
            va="center", ha=ha_align, fontsize=10, color=text_color, fontweight="bold",
        )
    ax.margins(x=0.2)
    ax.set_title(f"Target Distribution (Total n = {total:,})", fontsize=13, pad=12)
    fig.tight_layout()
    return _savefig(fig, "target_distribution", fig_dir)


def plot_numeric_distributions(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    fig_dir: Path = _FIG_DIR,
) -> Path:
    """Grid of histograms for key numeric features."""
    apply_publication_style()
    default_cols = ["int_rate", "annual_inc", "dti", "loan_amnt", "fico_range_low", "revol_util"]
    cols = [c for c in (cols or default_cols) if c in df.columns]

    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes_flat = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for ax, col in zip(axes_flat, cols):
        data = df[col].dropna()
        if col == "annual_inc":
            data = data.clip(upper=data.quantile(0.995))
        good = data[df.loc[data.index, TARGET_COL] == 0]
        bad  = data[df.loc[data.index, TARGET_COL] == 1]
        ax.hist(good, bins=40, alpha=0.6, color=C_GREEN, label="Good", density=True)
        ax.hist(bad,  bins=40, alpha=0.6, color=C_RED,   label="Bad",  density=True)
        ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_yticks([])
        ax.legend(fontsize=8, framealpha=0.9)
        despine(ax)

    for ax in axes_flat[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Feature Distributions: Good vs Bad", fontsize=14, y=0.98, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "numeric_distributions", fig_dir)


def run_eda(split: DataSplit, fig_dir: Path = _FIG_DIR) -> dict[str, Path]:
    """Run full EDA on the combined train+test DataFrame."""
    df = pd.concat([split.train, split.test], ignore_index=True)
    paths: dict[str, Path] = {}

    logger.info("Running EDA on %d loans...", len(df))
    paths["target_dist"]  = plot_target_distribution(df, fig_dir)
    paths["grade_dr"]     = plot_default_rate_by_grade(df, fig_dir)
    paths["vintage"]      = plot_vintage_default_curves(df, fig_dir)
    paths["term_dr"]      = plot_default_rate_by_term(df, fig_dir)
    paths["purpose_dr"]   = plot_default_rate_by_purpose(df, fig_dir)
    paths["missingness"]  = plot_missingness(df, fig_dir=fig_dir)
    paths["distributions"]= plot_numeric_distributions(df, fig_dir=fig_dir)

    logger.info("EDA complete. %d figures saved to %s.", len(paths), fig_dir)
    return paths
