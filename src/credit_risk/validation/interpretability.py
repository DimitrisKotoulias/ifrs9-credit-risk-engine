"""Partial Dependence (PDP) and Individual Conditional Expectation (ICE) plots.

SHAP reports *which* features matter; PDP/ICE show *how* a feature moves the prediction --
essential for regulatory model documentation. Partial dependence is computed directly from
the challenger's ``predict_proba`` (a plain probability callable), which sidesteps the
scikit-learn estimator protocol that a raw LightGBM ``Booster`` does not satisfy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from credit_risk.reporting.style import (  # noqa: E402
    C_BLUE,
    C_GRAY,
    C_NAVY,
    apply_publication_style,
    despine,
)

logger = logging.getLogger(__name__)

PredictFn = Callable[[pd.DataFrame], np.ndarray]


def _feature_grid(values: np.ndarray, grid_size: int) -> np.ndarray:
    """Percentile grid between the 1st and 99th percentiles (robust to outliers)."""
    # Filter out NaNs and Inf values
    clean_vals = values[np.isfinite(values)]
    if len(clean_vals) == 0:
        # Fallback if all values are missing or invalid
        return np.linspace(-1.0, 1.0, grid_size)

    lo, hi = np.percentile(clean_vals, [1, 99])
    if hi <= lo:
        lo, hi = float(clean_vals.min()), float(clean_vals.max())
    if hi <= lo:
        # Constant feature — create a tiny symmetric range around the value
        centre = lo if np.isfinite(lo) else 0.0
        eps = max(abs(centre) * 0.1, 1e-6)
        lo, hi = centre - eps, centre + eps
    return np.linspace(lo, hi, grid_size)


def partial_dependence_1d(
    predict_fn: PredictFn,
    X: pd.DataFrame,
    feature: str,
    grid_size: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 1-D PDP and per-row ICE curves for ``feature``.

    Returns ``(grid, pdp, ice)`` where ``pdp`` has shape ``(grid_size,)`` and ``ice`` has
    shape ``(n_rows, grid_size)``.
    """
    grid = _feature_grid(X[feature].to_numpy(dtype=float), grid_size)
    ice = np.empty((len(X), len(grid)), dtype=float)
    base = X.copy()
    for j, val in enumerate(grid):
        base[feature] = val
        ice[:, j] = np.asarray(predict_fn(base), dtype=float)
    pdp = ice.mean(axis=0)
    return grid, pdp, ice


def plot_pdp_grid(
    predict_fn: PredictFn,
    X: pd.DataFrame,
    features: list[str],
    fig_dir: Path = Path("reports/figures/validation"),
    grid_size: int = 25,
) -> plt.Figure:
    """Grid of partial-dependence plots for the top ``features``."""
    apply_publication_style()
    feats = [f for f in features if f in X.columns][:4]
    n = max(1, len(feats))
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), squeeze=False)
    flat = axes.ravel()

    for ax, feat in zip(flat, feats, strict=False):
        grid, pdp, _ = partial_dependence_1d(predict_fn, X, feat, grid_size)
        ax.plot(grid, pdp, color=C_BLUE, linewidth=2.2)
        ax.set_title(feat.replace("_", " "), fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(feat.replace("_", " "), fontsize=10)
        ax.set_ylabel("Partial dependence (PD)", fontsize=10)
        despine(ax)
    for ax in flat[len(feats):]:
        ax.set_visible(False)

    fig.suptitle("Partial Dependence Plots — LightGBM Challenger",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "pdp_grid.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    logger.info("PDP grid saved to %s/pdp_grid.png", fig_dir)
    return fig


def plot_ice(
    predict_fn: PredictFn,
    X: pd.DataFrame,
    feature: str,
    fig_dir: Path = Path("reports/figures/validation"),
    grid_size: int = 25,
    n_ice: int = 200,
    seed: int = 42,
) -> plt.Figure:
    """ICE plot for a single feature: individual curves + bold average PDP."""
    apply_publication_style()
    sample = X.sample(min(n_ice, len(X)), random_state=seed) if len(X) > n_ice else X
    grid, _, ice = partial_dependence_1d(predict_fn, sample, feature, grid_size)
    pdp = ice.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color curves dynamically using a professional sequential 'Blues' colormap
    # (ranging from a visible light blue to deep indigo) to keep the look clean and corporate
    mean_pds = ice.mean(axis=1)
    # Sort curves so darker ones are plotted on top of lighter ones for depth
    sort_idx = np.argsort(mean_pds)
    cmap = plt.cm.Blues
    
    for idx in sort_idx:
        # Scale between 0.35 and 0.85 to avoid lines being too pale or too dark
        val_norm = 0.35 + 0.50 * (mean_pds[idx] - mean_pds.min()) / (mean_pds.max() - mean_pds.min() + 1e-9)
        color = cmap(val_norm)
        ax.plot(grid, ice[idx], color=color, alpha=0.28, linewidth=0.9)
    ax.plot(grid, pdp, color=C_NAVY, linewidth=3.6, label="Average (PDP)")
    ax.set_xlabel(feature.replace("_", " "), fontsize=11)
    ax.set_ylabel("Predicted PD", fontsize=11)
    ax.set_title(f"ICE Plot: {feature.replace('_', ' ')}\n"
                 "(thin = individual borrowers, bold = average)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, framealpha=0.9)
    despine(ax)

    fig.tight_layout()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(fig_dir) / "ice_plot.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    logger.info("ICE plot saved to %s/ice_plot.png", fig_dir)
    return fig
