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
    chronological_oot_split,
    compute_calibration,
    plot_calibration_curve,
    select_oot_recalibrator,
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
    oot_order_key: np.ndarray | None = None,
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

    # Gains chart — OOT, so it matches the OOT material it is presented alongside in the
    # report. It used to be built on the in-time test partition and shown next to the OOT
    # ROC overlay with no partition label (Flaws.md finding N42).
    _savefig(plot_gains_chart(y_oot, y_pred_oot), "gains_chart", fig_dir)

    # The test-partition ROC and KS figures are no longer produced: neither was referenced
    # by the report, and the OOT overlay below already shows the test curve alongside the
    # OOT one (Flaws.md finding N37).

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

    # ── Recalibration gate ────────────────────────────────────────────────────
    # Triggered by OUT-OF-TIME evidence, fitted on the earlier half of the OOT window,
    # and accepted only if it demonstrably improves the later half (which is never used
    # for fitting). The previous gate tested the in-time partition, which cannot see the
    # 2016-2018 era drift at all -- see docs/AUDIT.md finding A1.
    if oot_order_key is None:
        logger.warning(
            "No OOT ordering key supplied: the recalibration gate will use a POSITIONAL "
            "split, which is not an out-of-time test. Pass issue dates instead."
        )
        oot_order_key = np.arange(len(y_oot))
        split_basis = "positional"
    else:
        split_basis = "issue_date"
    fit_mask = chronological_oot_split(oot_order_key, fit_fraction=0.5)

    # Record the actual date span of each slice. A chronological split must satisfy
    # max(fit) <= min(eval); publishing the bounds makes that checkable rather than
    # assumed, which is exactly what let a positional split masquerade as out-of-time
    # (Flaws.md finding N3).
    slice_bounds: dict = {}
    if split_basis == "issue_date":
        _keys = pd.Series(oot_order_key)
        _fit_keys, _eval_keys = _keys[fit_mask].dropna(), _keys[~fit_mask].dropna()
        if len(_fit_keys) and len(_eval_keys):
            slice_bounds = {
                "fit_slice_min_date": str(pd.Timestamp(_fit_keys.min()).date()),
                "fit_slice_max_date": str(pd.Timestamp(_fit_keys.max()).date()),
                "eval_slice_min_date": str(pd.Timestamp(_eval_keys.min()).date()),
                "eval_slice_max_date": str(pd.Timestamp(_eval_keys.max()).date()),
            }
            if pd.Timestamp(_fit_keys.max()) > pd.Timestamp(_eval_keys.min()):
                logger.warning(
                    "Recalibration gate slices overlap in time (fit max %s > eval min %s); "
                    "this is not a clean out-of-time split.",
                    slice_bounds["fit_slice_max_date"], slice_bounds["eval_slice_min_date"],
                )

    gate = select_oot_recalibrator(
        y_fit=y_oot[fit_mask],
        p_fit=y_pred_oot[fit_mask],
        y_eval=y_oot[~fit_mask],
        p_eval=y_pred_oot[~fit_mask],
        split_basis=split_basis,
        slice_bounds=slice_bounds,
    )
    calibrator = gate.pop("calibrator", None)
    metrics["calibration"]["recalibration_gate"] = gate
    # Back-compatible flag: True only when a transform is actually attached.
    metrics["calibration"]["isotonic_applied"] = bool(
        calibrator is not None and gate.get("chosen_method") == "isotonic"
    )
    metrics["calibration"]["recalibration_applied"] = calibrator is not None
    if calibrator is None:
        logger.info(
            "No recalibrator attached (triggered=%s, accepted=%s): %s",
            gate.get("triggered"), gate.get("accepted"),
            gate.get("accept_reason") or gate.get("trigger_reason") or gate.get("skip_reason"),
        )
    else:
        logger.info(
            "Recalibrator attached: %s fitted on the earlier OOT slice (%s)",
            gate.get("chosen_method"), gate.get("accept_reason"),
        )

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
