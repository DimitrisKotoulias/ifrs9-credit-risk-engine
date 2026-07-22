"""Stability metrics: PSI and CSI.

PSI (Population Stability Index):
    PSI = Σ_bins (A% − E%) · ln(A% / E%)
    Bands: <0.10 stable | 0.10–0.25 moderate | >0.25 significant shift.

CSI (Characteristic Stability Index): per-feature PSI.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.reporting.style import (
    apply_publication_style, despine,
    C_NAVY, C_BLUE, C_GOLD, C_GRAY, C_RED, C_GREEN, C_GRID,
)

logger = logging.getLogger(__name__)

PSI_STABLE   = 0.10
PSI_MODERATE = 0.25


def psi_band(value: float) -> str:
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_MODERATE:
        return "moderate_shift"
    return "significant_shift"


def compute_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute PSI between two score distributions."""
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr   = np.asarray(actual, dtype=float)

    quantiles = np.nanpercentile(expected_arr, np.linspace(0, 100, n_bins + 1))
    quantiles = np.unique(quantiles)
    quantiles[0]  = -np.inf
    quantiles[-1] = np.inf

    exp_counts = np.histogram(expected_arr, bins=quantiles)[0].astype(float)
    act_counts = np.histogram(actual_arr,   bins=quantiles)[0].astype(float)

    exp_pct = np.clip(exp_counts / exp_counts.sum(), epsilon, None)
    act_pct = np.clip(act_counts / act_counts.sum(), epsilon, None)

    psi_val = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return psi_val


def compute_psi_table(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    label: str = "train → test",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute per-bin PSI table with traffic-light band."""
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr   = np.asarray(actual, dtype=float)

    quantiles = np.nanpercentile(expected_arr, np.linspace(0, 100, n_bins + 1))
    quantiles = np.unique(quantiles)
    quantiles[0]  = -np.inf
    quantiles[-1] = np.inf

    exp_counts = np.histogram(expected_arr, bins=quantiles)[0].astype(float)
    act_counts = np.histogram(actual_arr,   bins=quantiles)[0].astype(float)

    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()
    epsilon = 1e-4
    psi_bins = (
        np.clip(act_pct, epsilon, None) - np.clip(exp_pct, epsilon, None)
    ) * np.log(
        np.clip(act_pct, epsilon, None) / np.clip(exp_pct, epsilon, None)
    )

    bin_labels = [
        f"({quantiles[i]:.2f}, {quantiles[i + 1]:.2f}]"
        for i in range(len(quantiles) - 1)
    ]

    tbl = pd.DataFrame({
        "bin":              bin_labels[:len(psi_bins)],
        "exp_pct":          exp_pct[:len(psi_bins)] * 100,
        "act_pct":          act_pct[:len(psi_bins)] * 100,
        "psi_contribution": psi_bins,
    })

    total_psi = float(psi_bins.sum())
    band = psi_band(total_psi)
    logger.info("PSI [%s]: %.4f → %s", label, total_psi, band.replace("_", " "))
    tbl.attrs["psi"]  = total_psi
    tbl.attrs["band"] = band
    return tbl


def compute_csi(
    X_expected: pd.DataFrame,
    X_actual: pd.DataFrame,
    features: list[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute CSI (PSI per feature)."""
    features = features or [c for c in X_expected.columns if c in X_actual.columns]
    records = []
    for feat in features:
        if feat not in X_expected.columns or feat not in X_actual.columns:
            continue
        try:
            csi_val = compute_psi(X_expected[feat].dropna(), X_actual[feat].dropna(), n_bins=n_bins)
        except Exception:
            csi_val = np.nan
        records.append({"feature": feat, "csi": csi_val, "band": psi_band(csi_val)})

    return pd.DataFrame(records).sort_values("csi", ascending=False).reset_index(drop=True)


def plot_psi_distribution(
    expected: np.ndarray,
    actual: np.ndarray,
    label_exp: str = "Train",
    label_act: str = "OOT",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Overlay histograms of expected vs actual score distributions with RAG status box."""
    apply_publication_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
    else:
        fig = ax.figure

    psi_val = compute_psi(expected, actual)
    band    = psi_band(psi_val)

    ax.hist(expected, bins=40, alpha=0.6, density=True,
            color=C_BLUE, label=label_exp)
    ax.hist(actual,   bins=40, alpha=0.6, density=True,
            color=C_GOLD, label=label_act)

    # Color status box by band
    if psi_val < 0.10:
        box_edge = C_GREEN
        status_text = "STABLE"
        box_face = "#E8F5E9"
    elif psi_val < 0.25:
        box_edge = C_GOLD
        status_text = "WARNING"
        box_face = "#FFFDE7"
    else:
        box_edge = C_RED
        status_text = "UNSTABLE"
        box_face = "#FFEBEE"

    status_box = (
        f"PSI: {psi_val:.4f}\n"
        f"Status: {status_text}\n\n"
        f"PSI Thresholds:\n"
        f"  < 0.10: Stable\n"
        f"  0.10 - 0.25: Warning\n"
        f"  > 0.25: Unstable"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor=box_face, edgecolor=box_edge, linewidth=1.5)
    ax.text(0.05, 0.52, status_box, transform=ax.transAxes,
            fontsize=9, fontweight='bold', color=C_NAVY, bbox=props)

    ax.set_title(
        f"Population Stability Index (PSI): {label_exp} vs {label_act}",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("Predicted Probability of Default (PD)", fontsize=11, labelpad=8)
    ax.set_ylabel("Probability Density", fontsize=11, labelpad=8)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    despine(ax)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.6)
    fig.tight_layout()
    return fig
