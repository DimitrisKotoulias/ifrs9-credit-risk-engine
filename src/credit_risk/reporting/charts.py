"""Portfolio reporting charts — professional financial style."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from credit_risk.reporting.style import (  # noqa: E402
    apply_publication_style, despine,
    C_NAVY, C_BLUE, C_GOLD, C_GRAY, C_RED, C_GREEN, C_GRID,
)

logger = logging.getLogger(__name__)


def plot_ecl_tornado(
    sensitivity_df: pd.DataFrame,
    fig_dir: Path = Path("reports/figures"),
    scenario_shocks: dict[str, float] | None = None,
) -> plt.Figure:
    """ECL macro sensitivity tornado chart — financial style.

    Z convention (Vasicek Eq. 15): Z < 0 = adverse shock (recession, ECL up);
    Z > 0 = favourable shock (expansion, ECL down).
    """
    apply_publication_style()
    df = sensitivity_df.copy().sort_values("macro_shock", ascending=False)
    base_row = df[df["macro_shock"] == 0.0]
    if base_row.empty:
        base_ecl = float(df["total_ecl"].median())
    else:
        base_ecl = float(base_row["total_ecl"].iloc[0])

    df["ecl_change_pct"] = (df["total_ecl"] - base_ecl) / (base_ecl + 1e-9) * 100
    
    # We want labels like "Z = +2.0" or "Z = -2.0"
    labels = []
    for z in df["macro_shock"]:
        if z == 0.0:
            labels.append("Z = 0.0 (Baseline)")
        else:
            labels.append(f"Z = {z:+.1f}")
            
    values = df["ecl_change_pct"].values
    colors = [C_RED if v >= 0 else C_GREEN for v in values]

    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.5)))
    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.6)
    ax.axvline(0, color=C_NAVY, linewidth=1.5, zorder=3)
    ax.set_xlabel("ECL Change vs Baseline (%)", fontsize=12, labelpad=8)
    ax.set_ylabel("Macro Shock (Vasicek Z-factor)", fontsize=12, labelpad=8)
    ax.set_title("ECL Portfolio Sensitivity to Macroeconomic Shocks", fontsize=13, fontweight="bold", pad=12)
    despine(ax)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.6)
    ax.grid(False, axis="y")

    # Value labels with absolute differences
    x_range = max(abs(values)) if len(values) else 1.0
    for bar, val, (_, row) in zip(bars, values, df.iterrows()):
        offset = x_range * 0.02
        ha = "left" if val >= 0 else "right"
        x_pos = bar.get_width() + (offset if val >= 0 else -offset)
        
        # Calculate absolute change in millions
        chg_abs = row["total_ecl"] - base_ecl
        chg_abs_str = f"${chg_abs/1e6:+.1f}M"
        if val == 0.0:
            label_text = f"Baseline (${row['total_ecl']/1e6:.1f}M)"
            x_pos = 0.5
            ha = "left"
        else:
            label_text = f"{val:+.1f}% ({chg_abs_str})"
            
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label_text, va="center", ha=ha,
                fontsize=8.5, color=C_NAVY, fontweight="bold")

    ax.margins(x=0.25)
    fig.tight_layout()

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "ecl_tornado.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    logger.info("ECL tornado chart saved to %s/ecl_tornado.png", fig_dir)
    return fig


def plot_cutoff_profit(
    strategy_df: pd.DataFrame,
    fig_dir: Path = Path("reports/figures"),
    opt_cutoff: int | None = None,
) -> plt.Figure:
    """Expected Profit and RAROC vs score cutoff, optimum marked.

    Two stacked panels sharing the x-axis (no dual axis): top = Expected
    Profit ($M), bottom = RAROC (%). ``opt_cutoff`` marks the reconciled
    marginal-RAROC-hurdle optimum (interior); if None, falls back to the
    total-profit argmax for backward compatibility.
    """
    apply_publication_style()
    df = strategy_df.copy().sort_values("cutoff")
    df = df[df["approval_rate"] > 0.0]

    if opt_cutoff is not None and (df["cutoff"] == opt_cutoff).any():
        opt_idx = df.index[df["cutoff"] == opt_cutoff][0]
    else:
        opt_idx = df["expected_profit"].idxmax()
    opt_cut = int(df.loc[opt_idx, "cutoff"])
    opt_profit = float(df.loc[opt_idx, "expected_profit"])
    opt_raroc = float(df.loc[opt_idx, "raroc"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
    )

    ax1.plot(df["cutoff"], df["expected_profit"] / 1e6, color=C_NAVY, linewidth=2.0)
    ax1.axvline(opt_cut, color=C_GOLD, linewidth=1.5, linestyle="--", zorder=2)
    ax1.plot([opt_cut], [opt_profit / 1e6], marker="o", markersize=8,
             markerfacecolor=C_GOLD, markeredgecolor=C_NAVY, zorder=3)
    ax1.annotate(
        f"Optimal cutoff = {opt_cut}\nProfit = ${opt_profit/1e6:,.1f}M",
        xy=(opt_cut, opt_profit / 1e6), xytext=(12, -8),
        textcoords="offset points", fontsize=9, color=C_NAVY, fontweight="bold",
        va="top",
    )
    ax1.set_ylabel("Expected Profit ($M)", fontsize=11)
    ax1.set_title("Expected Profit and RAROC vs. Score Cutoff",
                  fontsize=13, fontweight="bold", pad=10)
    despine(ax1)

    ax2.plot(df["cutoff"], df["raroc"] * 100, color=C_BLUE, linewidth=2.0)
    ax2.axvline(opt_cut, color=C_GOLD, linewidth=1.5, linestyle="--", zorder=2)
    ax2.plot([opt_cut], [opt_raroc * 100], marker="o", markersize=8,
             markerfacecolor=C_GOLD, markeredgecolor=C_NAVY, zorder=3)
    ax2.set_ylabel("RAROC (%)", fontsize=11)
    ax2.set_xlabel("Score Cutoff", fontsize=11)
    despine(ax2)

    fig.align_ylabels((ax1, ax2))

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "cutoff_profit_curve.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Cutoff profit curve saved to %s/cutoff_profit_curve.png", fig_dir)
    return fig


def plot_loss_distribution(
    losses: np.ndarray,
    measures: dict[str, float],
    fig_dir: Path = Path("reports/figures"),
) -> plt.Figure:
    """Monte Carlo portfolio loss distribution with EL / VaR / ES markers.

    ``losses`` is the simulated loss array and ``measures`` the dict returned by
    ``risk.economic_capital.risk_measures`` (keys ``expected_loss, var, es, alpha``).
    Losses are drawn in $M.
    """
    apply_publication_style()
    losses_m = np.asarray(losses, dtype=float) / 1e6
    el = measures["expected_loss"] / 1e6
    var = measures["var"] / 1e6
    es = measures["es"] / 1e6
    alpha_pct = measures.get("alpha", 0.999) * 100.0

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(losses_m, bins=150, density=True, color=C_BLUE, alpha=0.75,
            edgecolor="none")
    ax.axvline(el, color=C_GREEN, linestyle="--", linewidth=2.0,
               label=f"Expected Loss = ${el:,.1f}M")
    ax.axvline(var, color=C_RED, linestyle="--", linewidth=2.0,
               label=f"VaR {alpha_pct:.1f}% = ${var:,.1f}M")
    ax.axvline(es, color=C_NAVY, linestyle="-", linewidth=2.0,
               label=f"ES {alpha_pct:.1f}% = ${es:,.1f}M")
    ax.set_xlabel("Portfolio Loss ($M)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Portfolio Loss Distribution — Monte Carlo ASRF (Vasicek)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(frameon=True, framealpha=0.9)
    despine(ax)

    fig.tight_layout()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "loss_distribution.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Loss distribution saved to %s/loss_distribution.png", fig_dir)
    return fig


def plot_km_survival(
    km_curves: dict[str, pd.DataFrame],
    fig_dir: Path = Path("reports/figures"),
) -> plt.Figure:
    """Kaplan-Meier survival curves per grade.

    ``km_curves`` maps a grade label to its lifelines ``survival_function_`` DataFrame
    (index = months on book, single column = survival probability).
    """
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    cycle = [C_BLUE, C_GOLD, C_NAVY, C_GREEN, C_RED, C_GRAY, "#8B5CF6"]
    for i, grade in enumerate(sorted(km_curves)):
        sf = km_curves[grade]
        ax.step(sf.index.to_numpy(), sf.iloc[:, 0].to_numpy(), where="post",
                color=cycle[i % len(cycle)], linewidth=2.0, label=f"Grade {grade}")
    ax.set_xlabel("Months on Book", fontsize=11)
    ax.set_ylabel("Survival Probability (Non-Default)", fontsize=11)
    ax.set_title("Kaplan-Meier Survival Curves by Grade", fontsize=13,
                 fontweight="bold", pad=10)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True, framealpha=0.9, ncol=2)
    despine(ax)

    fig.tight_layout()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "km_survival_curves.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("KM survival curves saved to %s/km_survival_curves.png", fig_dir)
    return fig


def plot_lgd_calibration(
    actual: np.ndarray,
    predicted: np.ndarray,
    decile_df: pd.DataFrame,
    fig_dir: Path = Path("reports/figures/validation"),
) -> plt.Figure:
    """LGD validation: distribution overlay + decile calibration scatter."""
    apply_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    bins = np.linspace(0, 1, 41)
    ax1.hist(np.asarray(actual, dtype=float), bins=bins, alpha=0.55,
             color=C_RED, label="Actual LGD")
    ax1.hist(np.asarray(predicted, dtype=float), bins=bins, alpha=0.55,
             color=C_BLUE, label="Predicted LGD")
    ax1.set_title("LGD Distribution: Predicted vs Actual", fontsize=13,
                  fontweight="bold", pad=10)
    ax1.set_xlabel("LGD", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.legend(frameon=True, framealpha=0.9)
    despine(ax1)

    if not decile_df.empty:
        ax2.scatter(decile_df["mean_predicted"], decile_df["mean_actual"],
                    s=60, color=C_NAVY, zorder=3, edgecolor="white")
    ax2.plot([0, 1], [0, 1], linestyle="--", color=C_GOLD, linewidth=1.8,
             label="Perfect Calibration")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("LGD Calibration by Decile", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Mean Predicted LGD", fontsize=11)
    ax2.set_ylabel("Mean Actual LGD", fontsize=11)
    ax2.legend(frameon=True, framealpha=0.9)
    despine(ax2)

    fig.tight_layout()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "lgd_calibration.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("LGD calibration saved to %s/lgd_calibration.png", fig_dir)
    return fig


def plot_shock_tornado(
    whatif_df: pd.DataFrame,
    fig_dir: Path = Path("reports/figures"),
) -> plt.Figure:
    """Tornado chart of ECL change ($M) under PD/LGD/EAD what-if stress scenarios.

    ``whatif_df`` must carry ``scenario`` and ``delta_ecl`` columns (as produced by
    ``risk.ifrs9_ecl.ecl_shock_sensitivity``).
    """
    apply_publication_style()
    df = whatif_df.copy()
    df = df.sort_values("delta_ecl")
    delta_m = df["delta_ecl"].to_numpy(dtype=float) / 1e6
    colors = [C_GREEN if v < 0 else C_RED for v in delta_m]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(df["scenario"].astype(str), delta_m, color=colors, alpha=0.85)
    ax.axvline(0, color=C_NAVY, linewidth=1.0)
    ax.set_xlabel(r"Change in ECL vs Baseline (\$M)", fontsize=11)
    ax.set_title("ECL Sensitivity — PD / LGD / EAD Stress Scenarios",
                 fontsize=13, fontweight="bold", pad=10)
    span = max(abs(delta_m.min()), abs(delta_m.max()), 1e-9)
    for bar, val in zip(bars, delta_m):
        off = span * 0.02
        ax.text(bar.get_width() + (off if val >= 0 else -off),
                bar.get_y() + bar.get_height() / 2,
                f"${val:+,.1f}M", va="center",
                ha="left" if val >= 0 else "right", fontsize=9, color=C_NAVY)
    ax.set_xlim(-span * 1.25, span * 1.25)
    despine(ax)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.6)

    fig.tight_layout()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "ecl_shock_tornado.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("ECL shock tornado saved to %s/ecl_shock_tornado.png", fig_dir)
    return fig


def plot_concentration(
    grouped: dict[str, pd.Series],
    fig_dir: Path = Path("reports/figures"),
    top_n: int = 15,
) -> plt.Figure:
    """Exposure concentration by dimension — one horizontal-bar panel per dimension.

    ``grouped`` maps a dimension name to a Series of exposure totals per category
    (as produced by ``risk.concentration.grouped_exposures``).
    """
    apply_publication_style()
    dims = list(grouped.keys())
    n = max(1, len(dims))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        s = grouped[dim]
        s = s.nlargest(top_n) if len(s) > top_n else s
        pct = (s / s.sum() * 100.0).sort_values(ascending=True)
        ax.barh([str(i) for i in pct.index], pct.to_numpy(), color=C_BLUE, alpha=0.85)
        title = dim.replace("addr_state", "State").replace("_", " ").title()
        ax.set_title(f"Exposure by {title}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("% of Portfolio Exposure", fontsize=10)
        despine(ax)
        ax.grid(True, axis="x", color=C_GRID, linewidth=0.6)

    fig.tight_layout()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "concentration_risk.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Concentration chart saved to %s/concentration_risk.png", fig_dir)
    return fig


def plot_pit_vs_ttc(
    pit_ttc: dict,
    fig_dir: Path = Path("reports/figures"),
) -> plt.Figure:
    """Two-panel PiT vs TTC view: default rate + TTC line (top), Z-factor (bottom).

    ``pit_ttc`` is the dict from ``risk.pit_ttc.run_pit_ttc`` (keys ``quarters,
    default_rates, ttc_pd, z_factors``).
    """
    apply_publication_style()
    quarters = list(pit_ttc.get("quarters", []))
    dr = np.asarray(pit_ttc.get("default_rates", []), dtype=float)
    z = np.asarray(pit_ttc.get("z_factors", []), dtype=float)
    ttc = float(pit_ttc.get("ttc_pd", 0.0))
    x = np.arange(len(quarters))
    step = max(1, len(x) // 12)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                   gridspec_kw={"hspace": 0.12})

    ax1.plot(x, dr * 100.0, marker="o", color=C_BLUE, linewidth=2.0,
             label="PiT Default Rate")
    ax1.axhline(ttc * 100.0, color=C_RED, linestyle="--", linewidth=2.0,
                label=f"TTC PD = {ttc:.2%}")
    ax1.set_ylabel("Default Rate (%)", fontsize=11)
    ax1.set_title("Point-in-Time vs Through-the-Cycle Probability of Default",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.legend(frameon=True, framealpha=0.9)
    despine(ax1)

    ax2.plot(x, z, marker="o", color=C_NAVY, linewidth=2.0, label="Systematic Factor Z")
    ax2.axhline(0.0, color=C_GRAY, linewidth=0.9)
    ax2.fill_between(x, z, 0, where=(z < 0), color=C_RED, alpha=0.30,
                     label="Adverse (Z<0)", interpolate=True)
    ax2.fill_between(x, z, 0, where=(z > 0), color=C_GREEN, alpha=0.30,
                     label="Benign (Z>0)", interpolate=True)
    ax2.set_ylabel("Vasicek Z-Factor", fontsize=11)
    ax2.set_xlabel("Quarter", fontsize=11)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([quarters[i] for i in x[::step]], rotation=45, ha="right")
    ax2.legend(frameon=True, framealpha=0.9, ncol=3)
    despine(ax2)
    ax2.grid(True, axis="y", color=C_GRID, linewidth=0.6)

    fig.tight_layout()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "pit_vs_ttc.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("PiT vs TTC chart saved to %s/pit_vs_ttc.png", fig_dir)
    return fig


def plot_calibration_by_vintage(
    vintage_df: pd.DataFrame,
    fig_dir: Path = Path("reports/figures/validation"),
) -> plt.Figure:
    """PD calibration ratio (predicted / actual) per vintage group, raw vs recalibrated.

    ``vintage_df`` comes from ``validation.calibration.calibration_by_vintage_group``
    (columns ``group, pd_ratio_raw, pd_ratio_isotonic``). A ratio of 1.0 is perfect;
    below 1.0 is under-prediction.
    """
    apply_publication_style()
    df = vintage_df.copy()
    groups = df["group"].astype(str).tolist()
    x = np.arange(len(groups))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, df["pd_ratio_raw"], width, color=C_RED, alpha=0.85,
           label="Raw PD / Actual")
    ax.bar(x + width / 2, df["pd_ratio_isotonic"], width, color=C_BLUE, alpha=0.85,
           label="Isotonic-recalibrated / Actual")
    ax.axhline(1.0, color=C_NAVY, linestyle="--", linewidth=1.5,
               label="Perfect (ratio = 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Predicted PD / Actual Default Rate", fontsize=11)
    ax.set_title("PD Calibration Ratio by Vintage Group (raw vs era-recalibrated)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(frameon=True, framealpha=0.9)
    despine(ax)

    fig.tight_layout()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "calibration_by_vintage.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Vintage calibration chart saved to %s/calibration_by_vintage.png", fig_dir)
    return fig


def plot_shap_comparison(
    shap_full: pd.DataFrame,
    shap_bureau: pd.DataFrame,
    fig_dir: Path = Path("reports/figures/validation"),
    top_n: int = 12,
) -> plt.Figure:
    """Side-by-side SHAP importance: full model vs bureau-only model.

    Each input is a mean-abs-SHAP summary (columns ``feature, mean_abs_shap``). The full
    model includes price features (``int_rate``, ``grade``); the bureau-only model
    excludes them, explaining why those dominate the full view but are absent on the right.
    """
    apply_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    for ax, sdf, title, color in (
        (ax1, shap_full, "Full Model (incl. int_rate, grade)", C_NAVY),
        (ax2, shap_bureau, "Bureau-Only Model", C_BLUE),
    ):
        top = sdf.head(top_n)
        ax.barh(top["feature"][::-1].astype(str), top["mean_abs_shap"][::-1],
                color=color, alpha=0.88, height=0.62)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        despine(ax)

    fig.suptitle("SHAP Global Feature Importance: Full vs Bureau-Only Challenger",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "shap_comparison.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("SHAP comparison saved to %s/shap_comparison.png", fig_dir)
    return fig
