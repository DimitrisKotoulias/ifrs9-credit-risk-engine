"""Discrimination metrics: AUC, Gini, KS, gains/lift, decile rank-ordering."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from credit_risk.reporting.style import (
    apply_publication_style, despine,
    C_NAVY, C_BLUE, C_GOLD, C_GRAY, C_RED, C_GREEN, C_GRID,
)

GINI_AMBER_DEGRADATION = 0.05
GINI_RED_DEGRADATION   = 0.10
PSI_GREEN = 0.10
PSI_AMBER = 0.25


@dataclass
class RAGStatus:
    """RAG (Red/Amber/Green) traffic light for model stability.

    Thresholds per SR 11-7 / ECB validation guidelines.
    """
    gini_train: float
    gini_oot: float
    psi: float

    @property
    def gini_rag(self) -> str:
        d = self.gini_train - self.gini_oot
        return "GREEN" if d < GINI_AMBER_DEGRADATION else ("AMBER" if d < GINI_RED_DEGRADATION else "RED")

    @property
    def psi_rag(self) -> str:
        return "GREEN" if self.psi < PSI_GREEN else ("AMBER" if self.psi < PSI_AMBER else "RED")

    @property
    def overall(self) -> str:
        statuses = [self.gini_rag, self.psi_rag]
        return "RED" if "RED" in statuses else ("AMBER" if "AMBER" in statuses else "GREEN")


logger = logging.getLogger(__name__)


def compute_discrimination(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    label: str = "model",
) -> dict[str, float]:
    """Compute AUC, Gini, KS."""
    y_t = np.asarray(y_true, dtype=float)
    y_s = np.asarray(y_score, dtype=float)

    auc = float(roc_auc_score(y_t, y_s))
    gini = 2.0 * auc - 1.0

    order = np.argsort(y_s)
    y_s_sorted = y_s[order]
    y_t_sorted = y_t[order]

    n_bad = y_t.sum()
    n_good = len(y_t) - n_bad
    cum_bad = np.cumsum(y_t_sorted) / (n_bad + 1e-12)
    cum_good = np.cumsum(1 - y_t_sorted) / (n_good + 1e-12)
    ks = float(np.max(np.abs(cum_bad - cum_good)))

    logger.info("[%s] AUC=%.4f | Gini=%.4f | KS=%.4f", label, auc, gini, ks)
    return {"auc": auc, "gini": gini, "ks": ks}


def compute_decile_table(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    score_is_pd: bool = True,
) -> pd.DataFrame:
    """Rank-ordering table by score decile."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "score": np.asarray(y_score)})
    ascending = score_is_pd
    df["decile"] = pd.qcut(
        df["score"].rank(method="first", ascending=ascending),
        q=10, labels=False, duplicates="drop",
    ) + 1

    tbl = (
        df.groupby("decile")
        .agg(n=("y", "count"), n_bad=("y", "sum"))
        .reset_index()
    )
    tbl["bad_rate"] = tbl["n_bad"] / tbl["n"]
    tbl["cum_n_bad"] = tbl["n_bad"].cumsum()
    tbl["cum_bad_rate"] = tbl["cum_n_bad"] / tbl["n_bad"].sum()
    overall_bad_rate = tbl["n_bad"].sum() / tbl["n"].sum()
    tbl["lift"] = tbl["bad_rate"] / overall_bad_rate
    return tbl


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str = "Scorecard",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """ROC curve: navy line, shaded AUC, annotated value (Fig 5A spec)."""
    apply_publication_style()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    ax.plot(fpr, tpr, color=C_NAVY, linewidth=2.5,
            label=f"{label}")
    ax.plot([0, 1], [0, 1], color=C_GRAY, linestyle="--", linewidth=1.5,
            label="Random")
    # Shade AUC area
    ax.fill_between(fpr, tpr, alpha=0.08, color=C_BLUE)
    
    # Stats box
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=C_NAVY, linewidth=0.8)
    ax.text(0.55, 0.15, f"AUC: {auc:.4f}\nGini: {gini:.4f}", transform=ax.transAxes,
            fontsize=10, fontweight='bold', color=C_NAVY, bbox=props)

    # Annotate AUC with arrow
    mid = len(fpr) // 2
    ax.annotate(
        f"AUC = {auc:.4f}",
        xy=(fpr[mid], tpr[mid]),
        xytext=(fpr[mid] + 0.15, tpr[mid] - 0.12),
        fontsize=9, color=C_NAVY,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=1.2),
    )
    ax.set_xlabel("False Positive Rate", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate", fontsize=11, labelpad=8)
    ax.set_title(f"ROC Curve: {label}", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    despine(ax)
    ax.grid(True, axis="both", color=C_GRID, linewidth=0.6)
    fig.tight_layout()
    return fig


def plot_ks_chart(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """KS chart: Bad=#B91C1C, Good=#15803D, fill between, KS annotation box (Fig 5B spec)."""
    apply_publication_style()
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score")
    n_bad = y_true.sum()
    n_good = len(y_true) - n_bad
    cum_bad  = np.cumsum(df["y"].values) / (n_bad + 1e-12)
    cum_good = np.cumsum(1 - df["y"].values) / (n_good + 1e-12)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    else:
        fig = ax.figure

    # Scale x to go from 0 to 100 (Population Percentile)
    x = np.linspace(0, 100, len(df))
    ax.plot(x, cum_bad,  color=C_RED,   linewidth=2.5, label="Bad (Default)")
    ax.plot(x, cum_good, color=C_GREEN, linewidth=2.5, label="Good")

    ks_pos = int(np.argmax(np.abs(cum_bad - cum_good)))
    ks = float(np.max(np.abs(cum_bad - cum_good)))
    ks_pos_pct = x[ks_pos]

    # Fill between
    ax.fill_between(x, cum_bad, cum_good, alpha=0.08, color=C_GOLD)

    # KS vertical line + annotation box
    ax.axvline(ks_pos_pct, color=C_GRAY, linestyle="--", linewidth=1.5)
    
    # Position box nicely
    box_x = ks_pos_pct + 4.0 if ks_pos_pct < 70 else ks_pos_pct - 22.0
    ax.annotate(
        f"KS = {ks:.4f}\n(at {ks_pos_pct:.1f}%)",
        xy=(ks_pos_pct, (cum_bad[ks_pos] + cum_good[ks_pos]) / 2),
        xytext=(box_x, 0.45),
        fontsize=10, color=C_NAVY, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9,
                  edgecolor=C_GRAY, linewidth=0.8),
        arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.2),
    )

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Population Percentile (Lowest to Highest Risk)", fontsize=11, labelpad=8)
    ax.set_ylabel("Cumulative Proportion", fontsize=11, labelpad=8)
    ax.set_title("Kolmogorov-Smirnov (KS) Separation Chart", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    despine(ax)
    ax.grid(True, axis="both", color=C_GRID, linewidth=0.6)
    fig.tight_layout()
    return fig


def plot_gains_chart(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Cumulative gains chart: navy model, gray random, shaded fill, capture annotations (Fig 10A spec)."""
    apply_publication_style()
    decile_tbl = compute_decile_table(y_true, y_score, score_is_pd=True)
    deciles = decile_tbl["decile"].values
    cum_bad_rate = decile_tbl["cum_bad_rate"].values

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    else:
        fig = ax.figure

    x_vals = deciles / 10
    ax.plot(x_vals, cum_bad_rate, marker="o", color=C_NAVY, linewidth=2.5,
            markerfacecolor="white", markeredgewidth=1.5, markersize=6, label="Model")
    ax.plot([0, 1], [0, 1], color=C_GRAY, linestyle="--", linewidth=1.5, label="Random")
    ax.fill_between(x_vals, cum_bad_rate, x_vals, alpha=0.08, color=C_BLUE)

    # Capture annotations at 20%, 40%, 60%
    for target_x in [0.2, 0.4, 0.6]:
        idx = np.argmin(np.abs(x_vals - target_x))
        cap = cum_bad_rate[idx]
        ax.annotate(
            f"{cap:.1%}",
            xy=(x_vals[idx], cap),
            xytext=(x_vals[idx] + 0.04, cap - 0.08),
            fontsize=9, color=C_NAVY, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.8),
        )

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Proportion of Population (Lowest PD First)", fontsize=11, labelpad=8)
    ax.set_ylabel("Cumulative % of Bads Captured", fontsize=11, labelpad=8)
    ax.set_title("Cumulative Gains (Lift) Chart", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    despine(ax)
    ax.grid(True, axis="both", color=C_GRID, linewidth=0.6)
    fig.tight_layout()
    return fig


def plot_roc_oot_overlay(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    y_oot: np.ndarray,
    y_pred_oot: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """ROC curve comparison between In-time Test and OOT showing Gini degradation."""
    apply_publication_style()
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)
    fpr_oot, tpr_oot, _ = roc_curve(y_oot, y_pred_oot)
    
    auc_test = roc_auc_score(y_test, y_pred_test)
    auc_oot = roc_auc_score(y_oot, y_pred_oot)
    gini_test = 2 * auc_test - 1
    gini_oot = 2 * auc_oot - 1
    delta_gini = gini_test - gini_oot

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    # Plot curves
    ax.plot(fpr_test, tpr_test, color=C_GOLD, linewidth=2.5,
            label=f"In-Time Test (Gini = {gini_test:.4f})")
    ax.plot(fpr_oot, tpr_oot, color=C_NAVY, linewidth=2.5,
            label=f"Out-of-Time Validation (Gini = {gini_oot:.4f})")
    ax.plot([0, 1], [0, 1], color=C_GRAY, linestyle="--", linewidth=1.5)

    # Interpolate to shade the difference
    from scipy.interpolate import interp1d  # noqa: PLC0415
    all_fpr = np.unique(np.concatenate([fpr_test, fpr_oot]))
    tpr_test_interp = interp1d(fpr_test, tpr_test, kind='linear')(all_fpr)
    tpr_oot_interp = interp1d(fpr_oot, tpr_oot, kind='linear')(all_fpr)

    ax.fill_between(all_fpr, tpr_test_interp, tpr_oot_interp,
                    where=(tpr_test_interp >= tpr_oot_interp),
                    alpha=0.15, color=C_RED, label=f"Gini Degradation ({delta_gini:+.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate", fontsize=11, labelpad=8)
    ax.set_title("OOT Generalisation: ROC Curve Overlay", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    despine(ax)
    ax.grid(True, axis="both", color=C_GRID, linewidth=0.6)
    fig.tight_layout()
    return fig


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (auc_mean, lower_ci, upper_ci) via stratified bootstrap."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if y_true[idx].sum() < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    if not aucs:
        mean_auc = float(roc_auc_score(y_true, y_pred))
        return mean_auc, mean_auc, mean_auc
    a = (1 - ci) / 2
    return float(np.mean(aucs)), float(np.quantile(aucs, a)), float(np.quantile(aucs, 1 - a))


def delong_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, float]:
    """Hanley-McNeil variance approximation DeLong (1988) AUC comparison test."""
    from scipy import stats  # noqa: PLC0415

    y_true = np.asarray(y_true, dtype=float)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    auc_a = float(roc_auc_score(y_true, y_pred_a))
    auc_b = float(roc_auc_score(y_true, y_pred_b))

    def _variance(auc: float) -> float:
        q1 = auc / (2 - auc)
        q2 = 2 * auc**2 / (1 + auc)
        return (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0)

    se = float(np.sqrt(_variance(auc_a) + _variance(auc_b)))
    z = (auc_a - auc_b) / (se + 1e-15)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"z_stat": float(z), "p_value": p, "auc_a": auc_a, "auc_b": auc_b}


def pd_backtest_by_vintage(
    df: pd.DataFrame,
    pd_col: str = "pd_pred",
    target_col: str = "target",
    vintage_col: str = "issue_d",
    freq: str = "Q",
) -> pd.DataFrame:
    """Compare predicted mean 12m PD vs observed default rate by origination cohort."""
    df_work = df[[pd_col, target_col, vintage_col]].copy()
    df_work["vintage"] = pd.to_datetime(df_work[vintage_col], errors="coerce").dt.to_period(freq)
    df_work = df_work.dropna(subset=["vintage"])
    result = (
        df_work.groupby("vintage")
        .agg(
            n_loans=(target_col, "count"),
            predicted_pd=(pd_col, "mean"),
            actual_default_rate=(target_col, "mean"),
        )
        .reset_index()
    )
    result["pd_ratio"] = result["predicted_pd"] / result["actual_default_rate"].clip(lower=1e-9)
    result["vintage"] = result["vintage"].astype(str)
    return result
