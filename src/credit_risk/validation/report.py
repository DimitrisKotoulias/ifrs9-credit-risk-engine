"""Validation report generator.

Runs full validation suite and writes metrics.json + all figures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.validation.calibration import (
    compute_calibration,
    fit_isotonic_calibrator,
    plot_calibration_curve,
)
from credit_risk.validation.discrimination import (
    compute_decile_table,
    compute_discrimination,
    plot_gains_chart,
    plot_ks_chart,
    plot_roc_curve,
    plot_roc_oot_overlay,
)
from credit_risk.validation.stability import (
    compute_csi,
    compute_psi,
    compute_psi_table,
    plot_psi_distribution,
)
from credit_risk.reporting.style import apply_publication_style

logger = logging.getLogger(__name__)
_FIG_DIR = Path("reports/figures/validation")


def _savefig(fig: "plt.Figure", name: str, fig_dir: Path = _FIG_DIR) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved validation figure: %s/%s.png", fig_dir, name)


def run_validation(
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    y_oot: np.ndarray,
    y_pred_oot: np.ndarray,
    X_train: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    X_oot: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    output_dir: Path = Path("outputs"),
    fig_dir: Path = _FIG_DIR,
) -> tuple[dict, object | None]:
    """Run full validation: discrimination, calibration, PSI, OOT.

    Returns
    -------
    dict
        Nested metrics dict suitable for JSON serialisation.
    """
    apply_publication_style()
    metrics: dict = {}

    # ── Discrimination ─────────────────────────────────────────────────────────
    train_disc = compute_discrimination(y_train, y_pred_train, label="train")
    test_disc = compute_discrimination(y_test, y_pred_test, label="test")
    oot_disc = compute_discrimination(y_oot, y_pred_oot, label="OOT")
    metrics["discrimination"] = {
        "train": train_disc,
        "test": test_disc,
        "oot": oot_disc,
    }

    # ROC + KS + gains figures — in-time test set (clearly labeled)
    _savefig(plot_roc_curve(y_test, y_pred_test, label="Scorecard (test)"), "roc_curve_test", fig_dir)
    _savefig(plot_ks_chart(y_test, y_pred_test), "ks_chart_test", fig_dir)
    _savefig(plot_gains_chart(y_test, y_pred_test), "gains_chart", fig_dir)

    # ROC + KS figures — OOT set (Fix 1.5: Figure 5 must show OOT metrics)
    _savefig(plot_roc_curve(y_oot, y_pred_oot, label="Scorecard (OOT)"), "roc_curve_oot", fig_dir)
    _savefig(plot_ks_chart(y_oot, y_pred_oot), "ks_chart_oot", fig_dir)

    # OOT ROC overlay
    fig = plot_roc_oot_overlay(y_test, y_pred_test, y_oot, y_pred_oot)
    _savefig(fig, "roc_oot_overlay", fig_dir)

    # Decile rank-ordering (OOT)
    decile_tbl = compute_decile_table(y_oot, y_pred_oot, score_is_pd=True)
    metrics["oot_decile_table"] = decile_tbl.to_dict(orient="records")

    # ── Calibration ────────────────────────────────────────────────────────────
    test_cal = compute_calibration(y_test, y_pred_test, label="test")
    oot_cal = compute_calibration(y_oot, y_pred_oot, label="OOT")
    metrics["calibration"] = {"test": test_cal, "oot": oot_cal}

    _savefig(plot_calibration_curve(y_test, y_pred_test, label="Test"), "calibration_test", fig_dir)
    _savefig(plot_calibration_curve(y_oot, y_pred_oot, label="OOT"), "calibration_oot", fig_dir)

    # Fit calibrator if HL p-value < 0.05
    calibrator = None
    if test_cal["hl_pvalue"] < 0.05:
        calibrator = fit_isotonic_calibrator(y_test, y_pred_test)
        metrics["calibration"]["isotonic_applied"] = True
        logger.info("Isotonic calibrator fitted and will be attached to scorecard.")
    else:
        metrics["calibration"]["isotonic_applied"] = False

    # ── Stability (PSI) ────────────────────────────────────────────────────────
    psi_test = compute_psi(y_pred_train, y_pred_test)
    psi_oot = compute_psi(y_pred_train, y_pred_oot)

    metrics["stability"] = {
        "psi_train_test": psi_test,
        "psi_train_oot": psi_oot,
        "band_test": "stable" if psi_test < 0.10 else ("moderate" if psi_test < 0.25 else "significant"),
        "band_oot": "stable" if psi_oot < 0.10 else ("moderate" if psi_oot < 0.25 else "significant"),
    }

    _savefig(
        plot_psi_distribution(y_pred_train, y_pred_oot, "Train", "OOT"),
        "psi_distribution", fig_dir
    )

    if X_train is not None and X_oot is not None and feature_cols:
        csi_tbl = compute_csi(X_train, X_oot, features=feature_cols)
        metrics["csi_table"] = csi_tbl.to_dict(orient="records")

    # ── OOT summary ───────────────────────────────────────────────────────────
    oot_degradation = {
        "auc_degradation": test_disc["auc"] - oot_disc["auc"],
        "gini_degradation": test_disc["gini"] - oot_disc["gini"],
        "ks_degradation": test_disc["ks"] - oot_disc["ks"],
    }
    metrics["oot_degradation"] = oot_degradation

    logger.info(
        "OOT degradation: AUC -%.4f | Gini -%.4f | KS -%.4f",
        oot_degradation["auc_degradation"],
        oot_degradation["gini_degradation"],
        oot_degradation["ks_degradation"],
    )

    # ── Save metrics ───────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    logger.info("Metrics written to %s", metrics_path)

    return metrics, calibrator
