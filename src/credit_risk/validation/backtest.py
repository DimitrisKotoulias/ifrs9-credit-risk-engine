"""Backtesting: compare model predictions against realised outcomes.

Implements:
  - Vintage-level PD back-test (predicted vs realised default frequency)
  - Score-band stability heat-map generation

References:
  - Cantor & Mann (2003): Measuring the Performance of Corporate Bond Ratings
  - BCBS (2005): Studies on the Validation of Internal Rating Systems
  - EBA (2017): Guidelines on PD estimation, LGD estimation and treatment of defaulted assets
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

# ── Vintage backtest tolerance band: the single source of truth ─────────────────
#
# There were four. The report's equation and text described a "50% tolerance band"
# ([0.50, 1.50]); the very next paragraph counted vintages against [0.80, 1.25]; this
# production flag used [0.80, 1.20] with an amber zone of [0.60, 1.50]; and
# render_latex.py carried a second, independent implementation of the same flag on
# [0.80, 1.25]. Two implementations of one rule with different thresholds, and three
# different descriptions of it in prose. Everything now derives from these constants,
# including the wording, so the numbers in the text cannot drift from the flag again.
VINTAGE_PASS_BAND = (0.80, 1.25)
VINTAGE_AMBER_BAND = (0.60, 1.50)


def vintage_calibration_flag(ratio: float) -> str:
    """pass / amber / fail for a predicted-to-actual PD ratio."""
    lo, hi = VINTAGE_PASS_BAND
    if lo <= ratio <= hi:
        return "pass"
    lo_a, hi_a = VINTAGE_AMBER_BAND
    if lo_a <= ratio <= hi_a:
        return "amber"
    return "fail"


def vintage_band_text() -> str:
    """The pass band as it should be written in the report, e.g. ``[0.80, 1.25]``."""
    lo, hi = VINTAGE_PASS_BAND
    return f"[{lo:.2f}, {hi:.2f}]"


def vintage_pd_accuracy(
    df: pd.DataFrame,
    pd_col: str = "pd_pred",
    target_col: str = "target",
    vintage_col: str = "issue_d",
    freq: str = "Q",
) -> pd.DataFrame:
    """Compare predicted average PD vs realised default rate by vintage cohort.

    For a well-calibrated model, the predicted mean PD should track the
    actual default rate within 50% tolerance (BCBS validation studies).

    Returns DataFrame with columns:
        vintage, n_loans, predicted_pd, actual_dr, pd_ratio, calibration_flag
    """
    df_work = df[[pd_col, target_col, vintage_col]].copy()
    df_work["vintage"] = pd.to_datetime(df_work[vintage_col], errors="coerce").dt.to_period(freq)
    df_work = df_work.dropna(subset=["vintage"])

    agg = (
        df_work.groupby("vintage")
        .agg(
            n_loans=(target_col, "count"),
            predicted_pd=(pd_col, "mean"),
            actual_dr=(target_col, "mean"),
        )
        .reset_index()
    )
    result = agg
    result["pd_ratio"] = result["predicted_pd"] / result["actual_dr"].clip(lower=1e-6)
    def _flag(row) -> str:
        # Perfect zero prediction with zero actual defaults: treat as pass
        if row["predicted_pd"] == 0.0 and row["actual_dr"] == 0.0:
            return "pass"
        return vintage_calibration_flag(row["pd_ratio"])
    result["calibration_flag"] = result.apply(_flag, axis=1)
    # Also add actual_default_rate alias to prevent 0.0000 bug in Latex row rendering:
    result["actual_default_rate"] = result["actual_dr"]
    result["vintage"] = result["vintage"].astype(str)
    return result


def score_band_stability_heatmap(
    df_train: pd.DataFrame,
    df_oot: pd.DataFrame,
    score_col: str = "score",
    n_bands: int = 10,
    fig_dir: Path = Path("reports/figures"),
) -> pd.DataFrame:
    """Generate score-band population heatmap (train vs OOT).

    Retained for ad-hoc use; not called by the pipeline and the figure is not referenced
    by the report.

    Useful for identifying score-band-level distribution shifts beyond
    aggregate PSI, per BCBS Working Paper 14 guidance.
    """
    try:
        bounds = np.percentile(df_train[score_col].dropna(), np.linspace(0, 100, n_bands + 1))
        bounds = np.unique(bounds)
        labels = [f"B{i+1}" for i in range(len(bounds) - 1)]

        train_bands = pd.cut(df_train[score_col], bins=bounds, labels=labels, include_lowest=True)
        oot_bands = pd.cut(df_oot[score_col], bins=bounds, labels=labels, include_lowest=True)

        train_pct = train_bands.value_counts(normalize=True).sort_index()
        oot_pct = oot_bands.value_counts(normalize=True).sort_index()

        diff = (oot_pct - train_pct).fillna(0)
        matrix = pd.DataFrame({"train_pct": train_pct, "oot_pct": oot_pct, "diff": diff})

        fig, ax = plt.subplots(figsize=(10, 3))
        im = ax.imshow(
            diff.values.reshape(1, -1),
            aspect="auto",
            cmap="RdYlGn_r",
            vmin=-0.05,
            vmax=0.05,
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label="OOT − Train population share")
        ax.set_title("Score Band Population Shift (OOT vs Train)")
        fig.tight_layout()
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(fig_dir) / "score_band_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return matrix
    except Exception as exc:
        logger.warning("Score band heatmap failed: %s", exc)
        return pd.DataFrame()
