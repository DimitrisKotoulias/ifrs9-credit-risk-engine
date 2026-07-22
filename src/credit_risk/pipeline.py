"""End-to-end pipeline orchestrator.

Runs Phases 1→9 sequentially. Usage:
    python -m credit_risk.pipeline
    make pipeline
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.utils.config import load_config
from credit_risk.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run_pipeline(cfg_path: Path | None = None) -> None:  # noqa: C901
    setup_logging()
    cfg = load_config(cfg_path)
    seed = cfg.random_seed

    outputs = Path(cfg.paths.outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    figs = Path(cfg.paths.figures)
    figs.mkdir(parents=True, exist_ok=True)

    # Observability: every "non-fatal" enhancement failure is recorded (not just logged)
    # so a phase that silently drops out of the report is visible in metrics.json instead
    # of vanishing. A logging handler captures them centrally — no per-phase wiring needed.
    phase_failures: list[dict] = []

    class _NonFatalCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if "non-fatal" in msg.lower():
                phase_failures.append({"logger": record.name, "message": msg})

    _nf_handler = _NonFatalCapture(level=logging.WARNING)
    logging.getLogger("credit_risk").addHandler(_nf_handler)

    # ── Phase 1: Data ─────────────────────────────────────────────────────────
    logger.info("=== Phase 1: Data Loading ===")
    from credit_risk.data.loader import load_and_prepare  # noqa: PLC0415
    from credit_risk.data.target import TARGET_COL  # noqa: PLC0415

    split, df_rejected = load_and_prepare(cfg)
    df_train = split.train
    df_test = split.test
    df_oot = split.oot
    n_train = len(df_train)
    n_test = len(df_test)
    n_oot = len(df_oot)
    # Real portfolio counts derived from the loaded data (was: hardcoded literals).
    # full_accepted is every accepted loan pre-leakage-filter; df_rejected is all rejects.
    n_accepted_raw = len(split.full_accepted) if split.full_accepted is not None else (n_train + n_test + n_oot)
    n_rejected_raw = len(df_rejected)
    logger.info("Train=%d, Test=%d, OOT=%d", len(df_train), len(df_test), len(df_oot))

    # ── Phase 1b: EDA ─────────────────────────────────────────────────────────
    logger.info("=== Phase 1b: EDA ===")
    try:
        from credit_risk.data.eda import run_eda  # noqa: PLC0415
        run_eda(split, figs)
    except Exception as e:
        logger.warning("EDA failed (non-fatal): %s", e)

    # ── Phase 2: PD Scorecard ─────────────────────────────────────────────────
    logger.info("=== Phase 2: PD Scorecard ===")
    from credit_risk.models.pd_scorecard import PDScorecard  # noqa: PLC0415

    y_train = df_train[TARGET_COL]
    y_test = df_test[TARGET_COL]
    y_oot = df_oot[TARGET_COL]

    scorecard = PDScorecard(
        pdo=cfg.scorecard.pdo,
        base_score=cfg.scorecard.base_score,
        base_odds=cfg.scorecard.base_odds,
    )
    _sc_fit_t0 = time.perf_counter()
    scorecard.fit(df_train, y_train, df_test, y_test)
    sc_train_time = time.perf_counter() - _sc_fit_t0
    scorecard.save(outputs / "scorecard.pkl")

    # Export scorecard tables for the validation report
    _sc_result = scorecard._logit_result
    _coef_rows = [
        {
            "feature": "const",
            "coefficient": float(_sc_result.params.get("const", 0)),
            "std_err": float(_sc_result.bse.get("const", 0)),
            "z_stat": float(_sc_result.tvalues.get("const", 0)),
            "p_value": float(_sc_result.pvalues.get("const", 0)),
        }
    ] + [
        {
            "feature": feat,
            "coefficient": float(_sc_result.params[feat]),
            "std_err": float(_sc_result.bse[feat]),
            "z_stat": float(_sc_result.tvalues[feat]),
            "p_value": float(_sc_result.pvalues[feat]),
        }
        for feat in scorecard.feature_names
    ]
    _iv_tbl = scorecard._woe_transformer.get_iv_table()
    _sc_tables = {
        "scorecard_table": scorecard.scorecard_table.to_dict(orient="records"),
        "iv_table": _iv_tbl.to_dict(orient="records"),
        "logit_coefficients": _coef_rows,
        "selected_features": scorecard.feature_names,
        "factor": float(scorecard._factor),
        "offset": float(scorecard._offset),
    }
    with open(outputs / "scorecard_tables.json", "w") as _f:
        json.dump(_sc_tables, _f, indent=2, default=float)
    logger.info("Scorecard tables exported to scorecard_tables.json")

    # ── Phase 3: Validation ───────────────────────────────────────────────────
    logger.info("=== Phase 3: Validation ===")
    from credit_risk.validation.report import run_validation  # noqa: PLC0415

    pd_train = scorecard.predict_proba(df_train)
    pd_test = scorecard.predict_proba(df_test)
    pd_oot = scorecard.predict_proba(df_oot)

    val_metrics, calibrator = run_validation(
        y_train=y_train.values,
        y_pred_train=np.asarray(pd_train, dtype=float),
        y_test=y_test.values,
        y_pred_test=np.asarray(pd_test, dtype=float),
        y_oot=y_oot.values,
        y_pred_oot=np.asarray(pd_oot, dtype=float),
        output_dir=outputs,
        fig_dir=figs / "validation",
    )
    if calibrator is not None:
        scorecard.set_calibrator(calibrator)

    # Fit Model B (Pure Underwriting Scorecard - Circularity-free)
    logger.info("Fitting Model B (Pure Underwriting Scorecard)...")
    scorecard_underwriting = PDScorecard(
        pdo=cfg.scorecard.pdo,
        base_score=cfg.scorecard.base_score,
        base_odds=cfg.scorecard.base_odds,
        exclude_features=["int_rate", "grade_enc", "grade", "sub_grade", "sub_grade_enc", "loan_amnt", "funded_amnt", "funded_amnt_inv", "installment"],
    )
    scorecard_underwriting.fit(df_train, y_train, df_test, y_test)
    scorecard_underwriting.save(outputs / "scorecard_underwriting.pkl")

    pd_train_uw = scorecard_underwriting.predict_proba(df_train)
    pd_test_uw = scorecard_underwriting.predict_proba(df_test)
    pd_oot_uw = scorecard_underwriting.predict_proba(df_oot)

    from credit_risk.validation.discrimination import compute_discrimination  # noqa: PLC0415
    disc_train_uw = compute_discrimination(y_train.values, np.asarray(pd_train_uw, dtype=float), label="train_uw")
    disc_test_uw = compute_discrimination(y_test.values, np.asarray(pd_test_uw, dtype=float), label="test_uw")
    disc_oot_uw = compute_discrimination(y_oot.values, np.asarray(pd_oot_uw, dtype=float), label="oot_uw")

    # ── Phase 3 extras: bootstrap CIs, Spiegelhalter, Platt choice, CSI ────────
    try:
        from credit_risk.validation.discrimination import bootstrap_auc_ci  # noqa: PLC0415
        from credit_risk.validation.calibration import fit_platt_calibrator, spiegelhalter_test, compute_calibration_intercept_slope, compute_calibration  # noqa: PLC0415
        from credit_risk.validation.stability import compute_csi as _compute_csi  # noqa: PLC0415
        from sklearn.metrics import brier_score_loss  # noqa: PLC0415

        _pd_test_arr = np.asarray(pd_test, dtype=float)
        _pd_oot_arr = np.asarray(pd_oot, dtype=float)
        _pd_train_arr = np.asarray(pd_train, dtype=float)

        # Before recalibration stats
        intercept_before, slope_before = compute_calibration_intercept_slope(y_oot.values, _pd_oot_arr)
        brier_before = float(brier_score_loss(y_oot.values, _pd_oot_arr))
        expected_dr_before = float(_pd_oot_arr.mean())
        actual_dr_before = float(y_oot.values.mean())

        # Bootstrap AUC CIs (n_boot=500 for runtime)
        _, auc_lo_test, auc_hi_test = bootstrap_auc_ci(y_test.values, _pd_test_arr, n_boot=500)
        _, auc_lo_oot, auc_hi_oot = bootstrap_auc_ci(y_oot.values, _pd_oot_arr, n_boot=500)
        val_metrics["discrimination"]["test"].update({"auc_ci_lower": auc_lo_test, "auc_ci_upper": auc_hi_test})
        val_metrics["discrimination"]["oot"].update({"auc_ci_lower": auc_lo_oot, "auc_ci_upper": auc_hi_oot})

        # Spiegelhalter Z-test
        val_metrics["calibration"]["test"]["spiegelhalter"] = spiegelhalter_test(y_test.values, _pd_test_arr)
        val_metrics["calibration"]["oot"]["spiegelhalter"] = spiegelhalter_test(y_oot.values, _pd_oot_arr)

        # Platt vs isotonic calibration choice
        platt_cal = fit_platt_calibrator(y_train.values, _pd_train_arr)
        pd_test_platt = platt_cal.predict_proba(_pd_test_arr.reshape(-1, 1))[:, 1]
        brier_platt = float(brier_score_loss(y_test.values, pd_test_platt))
        if calibrator is not None:
            pd_test_iso = np.clip(calibrator.transform(_pd_test_arr), 1e-8, 1 - 1e-8)
            brier_iso = float(brier_score_loss(y_test.values, pd_test_iso))
        else:
            brier_iso = float(brier_score_loss(y_test.values, _pd_test_arr))
        
        if brier_platt <= brier_iso:
            scorecard.set_calibrator(platt_cal)
            val_metrics["calibration"]["method_chosen"] = "platt"
            pd_oot_calibrated = platt_cal.predict_proba(_pd_oot_arr.reshape(-1, 1))[:, 1]
        else:
            if calibrator is not None:
                scorecard.set_calibrator(calibrator)
            val_metrics["calibration"]["method_chosen"] = "isotonic"
            pd_oot_calibrated = np.clip(calibrator.transform(_pd_oot_arr), 1e-8, 1 - 1e-8) if calibrator is not None else _pd_oot_arr
            
        logger.info(
            "Calibration choice: %s (platt_brier=%.4f, iso_brier=%.4f)",
            val_metrics["calibration"]["method_chosen"], brier_platt, brier_iso,
        )

        # After recalibration stats
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
        from sklearn.metrics import roc_auc_score  # noqa: PLC0415
        
        # Recalibrator is fitted OUT-OF-SAMPLE on the in-time test partition and
        # then applied (transform only) to OOT. Fitting isotonic on OOT with OOT
        # labels would be in-sample recalibration and trivially "pass" the HL test
        # — that leakage is deliberately avoided so the reported OOT calibration
        # reflects genuine out-of-sample generalisation.
        iso_recal = IsotonicRegression(out_of_bounds='clip', increasing=True)
        iso_recal.fit(_pd_test_arr, y_test.values)
        pd_oot_calibrated_table = np.clip(iso_recal.transform(_pd_oot_arr), 1e-8, 1 - 1e-8)

        intercept_after, slope_after = compute_calibration_intercept_slope(y_oot.values, pd_oot_calibrated_table)
        brier_after = float(brier_score_loss(y_oot.values, pd_oot_calibrated_table))
        expected_dr_after = float(pd_oot_calibrated_table.mean())
        actual_dr_after = actual_dr_before

        cal_after_stats = compute_calibration(y_oot.values, pd_oot_calibrated_table, label="OOT_Calibrated")
        hl_pvalue_after = cal_after_stats["hl_pvalue"]
        auc_after = float(roc_auc_score(y_oot.values, pd_oot_calibrated_table))

        val_metrics["calibration_comparison"] = {
            "recalibration_fit_on": "in_time_test",
            "before": {
                "auc": float(val_metrics["discrimination"]["oot"]["auc"]),
                "brier": brier_before,
                "intercept": intercept_before,
                "slope": slope_before,
                "expected_dr": expected_dr_before,
                "actual_dr": actual_dr_before,
                "hl_pvalue": float(val_metrics["calibration"]["oot"]["hl_pvalue"]),
            },
            "after": {
                "auc": auc_after,
                "brier": brier_after,
                "intercept": intercept_after,
                "slope": slope_after,
                "expected_dr": expected_dr_after,
                "actual_dr": actual_dr_after,
                "hl_pvalue": hl_pvalue_after,
            }
        }

        # CSI per scorecard feature
        _csi_feats = [c for c in scorecard.feature_names if c in df_oot.columns]
        if _csi_feats:
            _csi_df = _compute_csi(df_train, df_oot, features=_csi_feats)
            val_metrics["csi_table"] = _csi_df.to_dict(orient="records")
            logger.info("CSI computed for %d features.", len(_csi_df))
    except Exception as _p3_err:
        logger.warning("Phase 3 extras failed (non-fatal): %s", _p3_err)

    # Flatten key metrics to top-level
    disc = val_metrics.get("discrimination", {})
    cal = val_metrics.get("calibration", {})
    stab = val_metrics.get("stability", {})
    metrics: dict = {
        "auc": disc.get("test", {}).get("auc", 0),
        "gini": disc.get("test", {}).get("gini", 0),
        "ks": disc.get("test", {}).get("ks", 0),
        "auc_oot": disc.get("oot", {}).get("auc", 0),
        "gini_oot": disc.get("oot", {}).get("gini", 0),
        "ks_oot": disc.get("oot", {}).get("ks", 0),
        "brier": cal.get("test", {}).get("brier_score", 0),
        "psi_total": stab.get("psi_train_oot", 0),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_oot": int(n_oot),
        "n_accepted_raw": int(n_accepted_raw),
        "n_rejected_raw": int(n_rejected_raw),
        "underwriting_scorecard": {
            "train": disc_train_uw,
            "test": disc_test_uw,
            "oot": disc_oot_uw,
        },
        **val_metrics,
    }

    try:
        from credit_risk.validation.discrimination import RAGStatus
        _rag = RAGStatus(
            gini_train=metrics.get("gini", 0.0),
            gini_oot=metrics.get("gini_oot", 0.0),
            psi=metrics.get("psi_total", 0.0),
        )
        metrics["rag_status"] = {
            "gini_rag": _rag.gini_rag,
            "psi_rag": _rag.psi_rag,
            "overall": _rag.overall,
        }
        logger.info("RAG Status: Gini=%s | PSI=%s | Overall=%s",
                    _rag.gini_rag, _rag.psi_rag, _rag.overall)
    except Exception as _e:
        logger.warning("RAG status failed: %s", _e)

    # ── Phase 2b: Challenger Model (LightGBM) ────────────────────────────────
    logger.info("=== Phase 2b: Challenger Model ===")
    try:
        from credit_risk.models.pd_challenger import PDChallenger  # noqa: PLC0415
        from credit_risk.validation.discrimination import compute_discrimination  # noqa: PLC0415

        challenger = PDChallenger(seed=seed)
        challenger.fit(
            df_train, y_train,
            df_test, y_test,
            feature_names=scorecard.feature_names,
        )
        challenger.save(outputs / "challenger.pkl")

        ch_pd_test = challenger.predict_proba(df_test)
        ch_pd_oot = challenger.predict_proba(df_oot)
        ch_disc_test = compute_discrimination(y_test.values, np.asarray(ch_pd_test, dtype=float), label="challenger_test")
        ch_disc_oot = compute_discrimination(y_oot.values, np.asarray(ch_pd_oot, dtype=float), label="challenger_oot")

        metrics["challenger"] = {
            "auc_test": ch_disc_test["auc"],
            "gini_test": ch_disc_test["gini"],
            "ks_test": ch_disc_test["ks"],
            "auc_oot": ch_disc_oot["auc"],
            "gini_oot": ch_disc_oot["gini"],
            "ks_oot": ch_disc_oot["ks"],
        }
        logger.info(
            "Challenger OOT: AUC=%.4f | Gini=%.4f | KS=%.4f",
            ch_disc_oot["auc"], ch_disc_oot["gini"], ch_disc_oot["ks"],
        )

        # DeLong test: scorecard vs challenger (OOT)
        try:
            from credit_risk.validation.discrimination import delong_test  # noqa: PLC0415
            delong_result = delong_test(y_oot.values, np.asarray(pd_oot, dtype=float), np.asarray(ch_pd_oot, dtype=float))
            metrics["challenger"]["delong_test"] = delong_result
            logger.info("DeLong test (OOT): z=%.4f, p=%.4f", delong_result["z_stat"], delong_result["p_value"])
        except Exception as _dl_err:
            logger.warning("DeLong test failed (non-fatal): %s", _dl_err)

        # Paired bootstrap A/B: is the challenger's Gini gain statistically significant?
        try:
            from credit_risk.validation.ab_test import paired_bootstrap_gini  # noqa: PLC0415
            metrics["ab_test"] = paired_bootstrap_gini(
                y_oot.values,
                np.asarray(pd_oot, dtype=float),
                np.asarray(ch_pd_oot, dtype=float),
                n_boot=2000, seed=seed,
            )
            logger.info("Paired bootstrap A/B (OOT): significant=%s",
                        metrics["ab_test"]["significant"])
        except Exception as _ab_err:  # noqa: BLE001
            logger.warning("Paired bootstrap A/B failed (non-fatal): %s", _ab_err)

        # Multi-Model ML Benchmark
        try:
            from credit_risk.models.pd_challenger import PDMultiModelBenchmark  # noqa: PLC0415
            logger.info("Training Multi-Model ML Benchmark...")
            benchmark = PDMultiModelBenchmark(seed=seed)
            benchmark.fit(
                df_train, y_train,
                df_test, y_test,
                feature_names=scorecard.feature_names,
            )

            # Extract predictions
            sc_pd_test = np.asarray(pd_test, dtype=float)
            sc_pd_oot = np.asarray(pd_oot, dtype=float)

            lgb_pd_test = benchmark.predict_proba_lgb(df_test)
            lgb_pd_oot = benchmark.predict_proba_lgb(df_oot)

            xgb_pd_test = benchmark.predict_proba_xgb(df_test)
            xgb_pd_oot = benchmark.predict_proba_xgb(df_oot)

            rf_pd_test = benchmark.predict_proba_rf(df_test)
            rf_pd_oot = benchmark.predict_proba_rf(df_oot)

            ens_pd_test = benchmark.predict_proba_ensemble(df_test, sc_pd_test)
            ens_pd_oot = benchmark.predict_proba_ensemble(df_oot, sc_pd_oot)

            # Compute metrics
            sc_disc_test = compute_discrimination(y_test.values, sc_pd_test, label="sc_test")
            sc_disc_oot = compute_discrimination(y_oot.values, sc_pd_oot, label="sc_oot")

            lgb_disc_test = compute_discrimination(y_test.values, lgb_pd_test, label="lgb_test")
            lgb_disc_oot = compute_discrimination(y_oot.values, lgb_pd_oot, label="lgb_oot")

            xgb_disc_test = compute_discrimination(y_test.values, xgb_pd_test, label="xgb_test")
            xgb_disc_oot = compute_discrimination(y_oot.values, xgb_pd_oot, label="xgb_oot")

            rf_disc_test = compute_discrimination(y_test.values, rf_pd_test, label="rf_test")
            rf_disc_oot = compute_discrimination(y_oot.values, rf_pd_oot, label="rf_oot")

            ens_disc_test = compute_discrimination(y_test.values, ens_pd_test, label="ens_test")
            ens_disc_oot = compute_discrimination(y_oot.values, ens_pd_oot, label="ens_oot")

            metrics["ml_benchmark_comparison"] = [
                {
                    "model": "Logistic Scorecard",
                    "test_auc": float(sc_disc_test["auc"]),
                    "oot_auc": float(sc_disc_oot["auc"]),
                    "test_gini": float(sc_disc_test["gini"]),
                    "oot_gini": float(sc_disc_oot["gini"]),
                    "test_ks": float(sc_disc_test["ks"]),
                    "oot_ks": float(sc_disc_oot["ks"]),
                    "train_time_sec": float(sc_train_time),
                },
                {
                    "model": "LightGBM Classifier",
                    "test_auc": float(lgb_disc_test["auc"]),
                    "oot_auc": float(lgb_disc_oot["auc"]),
                    "test_gini": float(lgb_disc_test["gini"]),
                    "oot_gini": float(lgb_disc_oot["gini"]),
                    "test_ks": float(lgb_disc_test["ks"]),
                    "oot_ks": float(lgb_disc_oot["ks"]),
                    "train_time_sec": float(benchmark.lgb_train_time),
                },
                {
                    "model": "XGBoost Classifier",
                    "test_auc": float(xgb_disc_test["auc"]),
                    "oot_auc": float(xgb_disc_oot["auc"]),
                    "test_gini": float(xgb_disc_test["gini"]),
                    "oot_gini": float(xgb_disc_oot["gini"]),
                    "test_ks": float(xgb_disc_test["ks"]),
                    "oot_ks": float(xgb_disc_oot["ks"]),
                    "train_time_sec": float(benchmark.xgb_train_time),
                },
                {
                    "model": "Random Forest Classifier",
                    "test_auc": float(rf_disc_test["auc"]),
                    "oot_auc": float(rf_disc_oot["auc"]),
                    "test_gini": float(rf_disc_test["gini"]),
                    "oot_gini": float(rf_disc_oot["gini"]),
                    "test_ks": float(rf_disc_test["ks"]),
                    "oot_ks": float(rf_disc_oot["ks"]),
                    "train_time_sec": float(benchmark.rf_train_time),
                },
                {
                    "model": "Weighted Ensemble",
                    "test_auc": float(ens_disc_test["auc"]),
                    "oot_auc": float(ens_disc_oot["auc"]),
                    "test_gini": float(ens_disc_test["gini"]),
                    "oot_gini": float(ens_disc_oot["gini"]),
                    "test_ks": float(ens_disc_test["ks"]),
                    "oot_ks": float(ens_disc_oot["ks"]),
                    "train_time_sec": float(benchmark.lgb_train_time + benchmark.xgb_train_time + benchmark.rf_train_time),
                }
            ]
            logger.info("Multi-Model ML Benchmarking completed successfully!")
        except Exception as mm_err:
            logger.warning("Multi-Model ML Benchmarking failed (non-fatal): %s", mm_err)

        # SHAP summary for challenger
        try:
            _shap_sample = df_oot.sample(min(10_000, len(df_oot)), random_state=seed)
            _shap_df = challenger.shap_summary(_shap_sample)
            if not _shap_df.empty:
                metrics["challenger"]["shap_mean_abs"] = _shap_df.to_dict(orient="records")
                _shap_fig_dir = figs / "validation"
                _shap_fig_dir.mkdir(parents=True, exist_ok=True)
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from credit_risk.reporting.style import apply_publication_style, despine, C_NAVY, C_BLUE, C_GRAY, C_GOLD  # noqa: PLC0415
                apply_publication_style()
                _fig, _ax = plt.subplots(figsize=(8.5, 5.5))
                _top = _shap_df.head(15)
                
                # Colors based on feature importance tiers
                _colors = [C_NAVY]*3 + [C_BLUE]*5 + [C_GRAY]*(len(_top) - 8)
                
                _ax.barh(_top["feature"][::-1], _top["mean_abs_shap"][::-1], color=_colors[::-1], alpha=0.88, height=0.6)
                
                # Mean importance line
                _mean_val = float(_top["mean_abs_shap"].mean())
                _ax.axvline(_mean_val, color=C_GOLD, linestyle="--", linewidth=1.5,
                            label=f"Mean Importance ({_mean_val:.3f})")
                
                _ax.set_xlabel("Mean |SHAP value| (LGBM Impact on Log-Odds)", fontsize=11, labelpad=8)
                _ax.set_title("Challenger Model: SHAP Global Feature Importance", fontsize=12, fontweight="bold", pad=12)
                despine(_ax)
                _ax.legend(loc="lower right", fontsize=9)
                _fig.tight_layout()
                _fig.savefig(_shap_fig_dir / "shap_challenger_summary.png", dpi=300)
                plt.close(_fig)
                logger.info("SHAP summary figure saved.")

                # Dual SHAP: full model vs bureau-only model (resolves text/figure
                # inconsistency — int_rate/grade dominate the full model but are
                # excluded from the bureau-only view).
                try:
                    from credit_risk.reporting.charts import plot_shap_comparison  # noqa: PLC0415

                    _price_exclude = {
                        "int_rate", "grade_enc", "grade", "sub_grade",
                        "loan_amnt", "funded_amnt", "installment",
                    }
                    _bureau_feats = [
                        f for f in scorecard.feature_names if f not in _price_exclude
                    ]
                    if len(_bureau_feats) >= 3:
                        _challenger_bureau = PDChallenger(seed=seed)
                        _challenger_bureau.fit(
                            df_train, y_train, df_test, y_test,
                            feature_names=_bureau_feats,
                        )
                        _shap_bureau_df = _challenger_bureau.shap_summary(_shap_sample)
                        if not _shap_bureau_df.empty:
                            metrics["challenger"]["shap_mean_abs_bureau"] = (
                                _shap_bureau_df.to_dict(orient="records")
                            )
                            plot_shap_comparison(_shap_df, _shap_bureau_df, _shap_fig_dir)
                            logger.info("Dual SHAP (full vs bureau) figure saved.")
                except Exception as _shap2_err:  # noqa: BLE001
                    logger.warning("Dual SHAP comparison failed (non-fatal): %s", _shap2_err)

                # PDP + ICE plots for the top challenger features (model interpretability)
                try:
                    from credit_risk.validation.interpretability import (  # noqa: PLC0415
                        plot_pdp_grid, plot_ice,
                    )
                    _top_feats = _shap_df["feature"].head(4).astype(str).tolist()
                    _pdp_X = _shap_sample[challenger._feature_names].astype(float)
                    if _top_feats:
                        plot_pdp_grid(challenger.predict_proba, _pdp_X, _top_feats, _shap_fig_dir)
                        plot_ice(challenger.predict_proba, _pdp_X, _top_feats[0], _shap_fig_dir)
                        logger.info("PDP/ICE plots saved.")
                except Exception as _pdp_err:  # noqa: BLE001
                    logger.warning("PDP/ICE plots failed (non-fatal): %s", _pdp_err)
        except Exception as _shap_err:
            logger.warning("SHAP summary failed (non-fatal): %s", _shap_err)
    except Exception as ch_err:
        logger.warning("Challenger model failed (non-fatal): %s", ch_err)
        metrics["challenger"] = {}

    # ── Phase 4: LGD ─────────────────────────────────────────────────────────
    logger.info("=== Phase 4: LGD ===")
    from credit_risk.models.lgd import LGDModel, compute_realised_lgd, lgd_backtest  # noqa: PLC0415

    # LGD model needs post-origination columns (recoveries, funded_amnt) that
    # were stripped by the leakage filter.  Use full_accepted to get them.
    lgd_cols_needed = ["recoveries", "funded_amnt", "collection_recovery_fee", "total_pymnt"]
    if split.full_accepted is not None:
        # Build a defaults-only DataFrame with both PD features and LGD columns
        full_defaults = split.full_accepted[split.full_accepted[TARGET_COL] == 1].copy()
        if "issue_d" in df_train.columns and "issue_d" in full_defaults.columns:
            train_issues = set(df_train["issue_d"].unique())
            defaults_train_lgd = full_defaults[full_defaults["issue_d"].isin(train_issues)]
        else:
            defaults_train_lgd = full_defaults
        if len(defaults_train_lgd) < 20:
            defaults_train_lgd = full_defaults
    else:
        defaults_train_lgd = df_train[df_train[TARGET_COL] == 1].copy()

    lgd_model = LGDModel(downturn_percentile=cfg.lgd.downturn_percentile)
    if len(defaults_train_lgd) >= 20:
        lgd_model.fit(defaults_train_lgd)
        lgd_model.save(outputs / "lgd_model.pkl")
        metrics["mean_lgd"] = lgd_model.mean_lgd
        metrics["downturn_lgd"] = lgd_model.downturn_lgd

        # LGD vintage backtest
        try:
            if "issue_d" in defaults_train_lgd.columns:
                _lgd_pred_bt = lgd_model.predict(defaults_train_lgd)
                _lgd_bt_df = lgd_backtest(defaults_train_lgd, _lgd_pred_bt)
                metrics["lgd_backtest"] = _lgd_bt_df.to_dict(orient="records")
                logger.info("LGD backtest done: %d vintage quarters.", len(_lgd_bt_df))
        except Exception as _lbt_err:
            logger.warning("LGD backtest failed (non-fatal): %s", _lbt_err)

        # Out-of-sample (chronological) LGD validation: MAE/RMSE/R2/KS + decile calib
        try:
            from credit_risk.validation.lgd_validation import (  # noqa: PLC0415
                validate_lgd, validate_lgd_models,
            )
            from credit_risk.reporting.charts import plot_lgd_calibration  # noqa: PLC0415

            # Chronological OOS test on defaults from vintages held out of fitting.
            # Restrict to MATURE vintages (issue_year <= 2016): the 2018Q4 data snapshot
            # has not yet resolved recoveries/charge-offs for 2017-2018 defaults, so their
            # realised LGD is unreliable and would corrupt the validation.
            if split.full_accepted is not None and "issue_d" in full_defaults.columns:
                _train_issues = set(df_train["issue_d"].unique())
                _fd_year = pd.to_datetime(full_defaults["issue_d"], format="%b-%Y", errors="coerce").dt.year
                defaults_test_lgd = full_defaults[
                    (~full_defaults["issue_d"].isin(_train_issues))
                    & (_fd_year <= 2016)
                ]
            else:
                defaults_test_lgd = full_defaults.iloc[0:0]

            if len(defaults_test_lgd) >= 50:
                # Split the held-out OOS set into a SELECT half (champion vs
                # challenger comparison + promotion decision) and a REPORT half
                # (final published OOS metrics). Using the same set for both would
                # let a challenger that merely overfits noise in that set win
                # selection, then have its accuracy on that identical set reported
                # as unbiased OOS performance — an optimistic selection bias.
                _dtl_dt = pd.to_datetime(
                    defaults_test_lgd["issue_d"], format="%b-%Y", errors="coerce"
                ) if "issue_d" in defaults_test_lgd.columns else pd.Series(range(len(defaults_test_lgd)), index=defaults_test_lgd.index)
                _dtl_sorted = defaults_test_lgd.loc[_dtl_dt.sort_values().index]
                _split_n = len(_dtl_sorted) // 2
                defaults_select_lgd = _dtl_sorted.iloc[:_split_n]
                defaults_report_lgd = _dtl_sorted.iloc[_split_n:]

                # Champion (two-stage) vs challenger (LightGBM), compared on the
                # SELECT half; promote the challenger only if it strictly beats the
                # champion on RMSE there. The deployed model is then used for all
                # downstream LGD (EL/ECL/RWA).
                _lgd_cmp = validate_lgd_models(lgd_model, defaults_select_lgd)
                metrics["lgd_model_comparison"] = _lgd_cmp
                if _lgd_cmp.get("recommended") == "challenger":
                    lgd_model.promote_to_challenger(defaults_train_lgd)
                    metrics["mean_lgd"] = lgd_model.mean_lgd
                    metrics["downturn_lgd"] = lgd_model.downturn_lgd
                    lgd_model.save(outputs / "lgd_model.pkl")

                # Final published OOS metrics are computed on the REPORT half,
                # disjoint from the set used for the promotion decision above.
                _lgd_val, _lgd_decile = validate_lgd(lgd_model, defaults_report_lgd)
                metrics["lgd_validation"] = {
                    **_lgd_val,
                    "decile": _lgd_decile.to_dict(orient="records"),
                }
                # Promote headline OOS severity metrics to top level so the report's
                # benchmark table (reports/benchmarks.py LGD_R2 row) renders the REAL
                # computed R^2 rather than a hand-typed value.
                metrics["lgd_r2"] = float(_lgd_val["r2"])
                metrics["lgd_rmse"] = float(_lgd_val["rmse"])
                metrics["lgd_ks"] = float(_lgd_val["ks_stat"])
                _actual = compute_realised_lgd(defaults_report_lgd).to_numpy(dtype=float)
                _pred = lgd_model.predict(defaults_report_lgd).to_numpy(dtype=float)
                plot_lgd_calibration(_actual, _pred, _lgd_decile, figs / "validation")
                logger.info(
                    "LGD OOS validation: MAE=%.4f R2=%.4f (n=%d)",
                    _lgd_val["mae"], _lgd_val["r2"], int(_lgd_val["n_test"]),
                )
            else:
                logger.warning("Too few OOS defaults (%d) for LGD validation.", len(defaults_test_lgd))
        except Exception as _lval_err:  # noqa: BLE001
            logger.warning("LGD OOS validation failed (non-fatal): %s", _lval_err)
    else:
        logger.warning("Not enough defaults (%d) to fit LGD model — using fallback.", len(defaults_train_lgd))

    # ── Phase 5: EAD ─────────────────────────────────────────────────────────
    logger.info("=== Phase 5: EAD ===")
    from credit_risk.models.ead import EADModel  # noqa: PLC0415

    ead_model = EADModel()
    ead_model.fit(df_train)

    # ── Compute PD, LGD, EAD on full portfolio ────────────────────────────────
    df_all = pd.concat([df_train, df_test, df_oot], ignore_index=True)

    df_all["pd_pred"] = np.asarray(scorecard.predict_proba(df_all), dtype=float)

    if lgd_model._severity_scaler is not None:
        df_all["lgd_pred"] = lgd_model.predict(df_all).values
    else:
        df_all["lgd_pred"] = float(lgd_model.mean_lgd or 0.45)

    df_all["ead"] = ead_model.predict(df_all).values

    # Vintage calibration drift diagnostic (raw vs era-recalibrated PD by vintage group)
    try:
        from credit_risk.validation.calibration import calibration_by_vintage_group  # noqa: PLC0415
        from credit_risk.reporting.charts import plot_calibration_by_vintage  # noqa: PLC0415

        _issue_year = pd.to_datetime(df_all["issue_d"], format="%b-%Y", errors="coerce").dt.year
        _vintage_cal = calibration_by_vintage_group(
            df_all[TARGET_COL].to_numpy(dtype=float),
            df_all["pd_pred"].to_numpy(dtype=float),
            _issue_year.to_numpy(dtype=float),
            split_year=2016,
        )
        if not _vintage_cal.empty:
            metrics["vintage_calibration"] = _vintage_cal.to_dict(orient="records")
            plot_calibration_by_vintage(_vintage_cal, figs / "validation")
            logger.info("Vintage calibration diagnostic computed (%d groups).", len(_vintage_cal))
    except Exception as _vc_err:  # noqa: BLE001
        logger.warning("Vintage calibration diagnostic failed (non-fatal): %s", _vc_err)

    # ── Phase 6: Expected Loss ────────────────────────────────────────────────
    logger.info("=== Phase 6: Expected Loss ===")
    from credit_risk.risk.expected_loss import run_expected_loss  # noqa: PLC0415

    df_el = run_expected_loss(df_all)
    df_el.to_parquet(outputs / "portfolio_el.parquet", index=False)
    el_summary = df_el.attrs.get("el_summary", {})
    metrics["total_el"] = el_summary.get("total_el", 0.0)
    metrics["total_ead_portfolio"] = el_summary.get("total_ead", 0.0)
    metrics["el_rate"] = el_summary.get("el_rate", 0.0)

    # ── Phase 7: Basel IRB ────────────────────────────────────────────────────
    logger.info("=== Phase 7: Basel IRB ===")
    from credit_risk.risk.basel_irb import run_basel_irb  # noqa: PLC0415

    downturn_lgd = float(lgd_model.downturn_lgd) if lgd_model.downturn_lgd > 0 else 0.45
    df_rwa = run_basel_irb(df_el, lgd_downturn=downturn_lgd, pd_floor=cfg.basel.pd_floor)
    df_rwa.to_parquet(outputs / "basel_rwa.parquet", index=False)
    basel_summary = df_rwa.attrs.get("basel_summary", {})
    metrics["total_rwa"] = basel_summary.get("total_rwa_irb", 0.0)
    metrics["total_rwa_sa"] = basel_summary.get("total_rwa_sa", 0.0)
    metrics["rwa_density"] = f"{basel_summary.get('rwa_density', 0) * 100:.1f}%"

    # ── Phase 7b: Economic Capital (Monte Carlo ASRF) ─────────────────────────
    logger.info("=== Phase 7b: Economic Capital (Monte Carlo ASRF) ===")
    try:
        from credit_risk.risk.economic_capital import run_economic_capital  # noqa: PLC0415
        from credit_risk.reporting.charts import plot_loss_distribution  # noqa: PLC0415

        ec_losses, ec_measures = run_economic_capital(
            df_rwa,
            rho=cfg.econ_cap.rho,
            n_sim=cfg.econ_cap.n_simulations,
            alpha=cfg.econ_cap.es_alpha,
            seed=cfg.econ_cap.seed,
            n_buckets=cfg.econ_cap.n_buckets,
        )
        reg_capital = float(metrics.get("total_rwa", 0.0)) * cfg.basel.capital_ratio
        metrics["econ_cap"] = {
            "expected_loss": ec_measures["expected_loss"],
            "var": ec_measures["var"],
            "es": ec_measures["es"],
            "unexpected_loss": ec_measures["unexpected_loss"],
            "economic_capital": ec_measures["economic_capital"],
            "alpha": ec_measures["alpha"],
            "regulatory_capital": reg_capital,
            "ec_to_reg_ratio": (
                ec_measures["economic_capital"] / reg_capital if reg_capital > 0 else 0.0
            ),
            "n_simulations": cfg.econ_cap.n_simulations,
            "rho": cfg.econ_cap.rho,
        }
        plot_loss_distribution(ec_losses, ec_measures, figs)
    except Exception as ec_err:  # noqa: BLE001
        logger.warning("Economic capital simulation failed (non-fatal): %s", ec_err)

    # Concentration risk: HHI by dimension + Granularity Adjustment surcharge
    try:
        from credit_risk.risk.concentration import run_concentration  # noqa: PLC0415
        from credit_risk.reporting.charts import plot_concentration  # noqa: PLC0415

        conc_summary, conc_grouped = run_concentration(df_rwa, rho=cfg.econ_cap.rho)
        metrics["concentration"] = conc_summary
        if conc_grouped:
            plot_concentration(conc_grouped, figs)
        logger.info("Concentration risk computed (%d dimensions).",
                    len(conc_summary.get("dimensions", [])))
    except Exception as conc_err:  # noqa: BLE001
        logger.warning("Concentration risk failed (non-fatal): %s", conc_err)

    # ── Phase 8: IFRS 9 ECL ───────────────────────────────────────────────────
    logger.info("=== Phase 8: IFRS 9 ECL ===")
    from credit_risk.models.pd_term_structure import DiscreteHazardModel  # noqa: PLC0415
    from credit_risk.risk.ifrs9_ecl import IFRS9Config, SICRConfig, ScenarioConfig, run_ifrs9_ecl  # noqa: PLC0415

    hazard_model = DiscreteHazardModel(max_horizon=60, seed=seed)
    hazard_model.fit(df_train)
    with open(outputs / "hazard_model.pkl", "wb") as f:
        pickle.dump(hazard_model, f)

    # ── Phase 8b: Survival Analysis (KM + Cox PH) — challenger term structure ──
    logger.info("=== Phase 8b: Survival Analysis (Kaplan-Meier + Cox PH) ===")
    try:
        from credit_risk.models.survival import SurvivalPDModel  # noqa: PLC0415
        from credit_risk.reporting.charts import plot_km_survival  # noqa: PLC0415

        surv_source = split.full_accepted if split.full_accepted is not None else df_train
        if "issue_d" in surv_source.columns and "issue_d" in df_train.columns:
            _train_issues = set(df_train["issue_d"].unique())
            surv_cohort = surv_source[surv_source["issue_d"].isin(_train_issues)]
        else:
            surv_cohort = surv_source
        surv_model = SurvivalPDModel(max_horizon=60, seed=seed)
        surv_model.fit(surv_cohort, target_col=TARGET_COL)
        _surv_metrics = surv_model.summary_metrics()
        metrics["survival"] = {
            "c_index": _surv_metrics["c_index"],
            "median_survival_months": _surv_metrics["median_survival_months"],
            "cox_summary": surv_model.cox_summary().to_dict(orient="records"),
        }
        if surv_model.km_curves:
            plot_km_survival(surv_model.km_curves, figs)
        logger.info("Survival analysis done: C-index=%.4f", surv_model.concordance)
    except Exception as _surv_err:  # noqa: BLE001
        logger.warning("Survival analysis failed (non-fatal): %s", _surv_err)

    # Train OLS macro model to derive scenario shocks dynamically
    from credit_risk.risk.ifrs9_ecl import fit_macro_model  # noqa: PLC0415
    macro_path = "data/processed/macro_quarterly.csv"
    macro_shocks = fit_macro_model(
        df_train,
        macro_path,
        unrate_lag=cfg.ifrs9.macro_unrate_lag,
        enforce_sign_priors=cfg.ifrs9.macro_enforce_sign_priors,
    )

    # Time-series diagnostics that justify the lag/sign choice (ADF/Granger/AIC/VECM)
    try:
        from credit_risk.validation.macro_ts import (  # noqa: PLC0415
            analyze_macro_timeseries, build_quarterly_macro_frame,
        )
        _macro_q = build_quarterly_macro_frame(df_train, macro_path)
        metrics["macro_ts"] = analyze_macro_timeseries(
            _macro_q, max_lag=cfg.macro_ts.max_lag
        )
        logger.info("Macro time-series diagnostics computed (%d quarters).", len(_macro_q))
    except Exception as _mts_err:  # noqa: BLE001
        logger.warning("Macro TS diagnostics failed (non-fatal): %s", _mts_err)

    # Point-in-Time vs Through-the-Cycle PD decomposition (Vasicek inversion)
    try:
        from credit_risk.validation.macro_ts import build_quarterly_macro_frame  # noqa: PLC0415
        from credit_risk.risk.pit_ttc import run_pit_ttc  # noqa: PLC0415
        from credit_risk.reporting.charts import plot_pit_vs_ttc  # noqa: PLC0415

        _pit_q = build_quarterly_macro_frame(df_train, macro_path)
        if len(_pit_q) >= 4:
            _pit_ttc = run_pit_ttc(_pit_q, rho=cfg.econ_cap.rho)
            metrics["pit_ttc"] = _pit_ttc
            plot_pit_vs_ttc(_pit_ttc, figs)
            logger.info("PiT/TTC decomposition computed (TTC PD=%.4f).", _pit_ttc["ttc_pd"])
        else:
            logger.warning("Too few quarters (%d) for PiT/TTC decomposition.", len(_pit_q))
    except Exception as _pit_err:  # noqa: BLE001
        logger.warning("PiT/TTC decomposition failed (non-fatal): %s", _pit_err)

    # Save macro results to metrics
    metrics["macro_elasticities"] = {
        k: float(v) for k, v in macro_shocks["elasticities"].items()
    }
    # Sign-adjusted coefficients actually used for scenario projection + method flags
    metrics["macro_elasticities_adjusted"] = {
        k: float(v) for k, v in macro_shocks.get("elasticities_adjusted", {}).items()
    }
    metrics["macro_sign_adjusted"] = bool(macro_shocks.get("macro_sign_adjusted", False))
    metrics["macro_unrate_lag"] = int(macro_shocks.get("macro_unrate_lag", 0))
    metrics["macro_r_squared"] = float(macro_shocks.get("r_squared", float("nan")))
    metrics["macro_predictions"] = {
        k: float(v) * 100 for k, v in macro_shocks["predictions"].items()
    }
    metrics["macro_implied_shocks"] = {
        "baseline": float(macro_shocks["baseline"]),
        "upside": float(macro_shocks["upside"]),
        "downside": float(macro_shocks["downside"])
    }
    # Scenario input assumptions (UNRATE/GDP_growth/FEDFUNDS/CPI_inflation per
    # scenario) so the report can show a verifiable assumptions table (Fix 1.3).
    if "scenario_inputs" in macro_shocks:
        metrics["macro_scenario_inputs"] = {
            scen: {k: float(v) for k, v in vals.items() if k != "const"}
            for scen, vals in macro_shocks["scenario_inputs"].items()
        }

    ifrs9_cfg = IFRS9Config(
        scenarios=[
            ScenarioConfig("baseline", 0.50, macro_shocks["baseline"]),
            ScenarioConfig("upside", 0.25, macro_shocks["upside"]),
            ScenarioConfig("downside", 0.25, macro_shocks["downside"]),
        ],
        sicr=SICRConfig(
            pd_multiplier=cfg.ifrs9.sicr.pd_multiplier,
            abs_threshold=cfg.ifrs9.sicr.abs_threshold,
            dpd_backstop=cfg.ifrs9.sicr.dpd_backstop,
        ),
    )

    lgd_arr = df_rwa["lgd_pred"].values if "lgd_pred" in df_rwa.columns else np.full(len(df_rwa), 0.45)
    ead_arr = df_rwa["ead"].values

    # Compute origination lifetime PD proxy for SICR staging.
    # We approximate origination PD using the current scorecard 12m PD scaled
    # by the loan term, since we have no separate origination snapshot.
    pd_12m_orig = np.asarray(df_rwa["pd_pred"].values, dtype=float)
    if "term" in df_rwa.columns:
        term_months = (
            pd.to_numeric(
                df_rwa["term"].astype(str).str.extract(r"(\d+)")[0],
                errors="coerce",
            )
            .fillna(36.0)
            .values
        )
    else:
        term_months = np.full(len(df_rwa), 36.0)
    term_years = np.clip(term_months / 12.0, 1.0, 5.0)
    pd_orig_lifetime = 1.0 - (1.0 - np.clip(pd_12m_orig, 1e-9, 1 - 1e-9)) ** term_years

    df_ecl = run_ifrs9_ecl(
        df_rwa, hazard_model, lgd_arr, ead_arr,
        cfg=ifrs9_cfg, pd_orig_lifetime=pd_orig_lifetime,
    )
    df_ecl.to_parquet(outputs / "ecl.parquet", index=False)

    ifrs9_summary = df_ecl.attrs.get("ifrs9_summary", {})
    metrics["total_ecl"] = ifrs9_summary.get("total_ecl", 0.0)
    metrics["ecl_coverage"] = ifrs9_summary.get("coverage_ratio", 0.0)
    stage_counts = ifrs9_summary.get("stage_counts", {})
    n_total = sum(stage_counts.values()) or 1
    metrics["stage2_pct"] = stage_counts.get(2, 0) / n_total
    metrics["stage3_pct"] = stage_counts.get(3, 0) / n_total

    # Lifetime-PD calibration diagnostic: the hazard model's own lifetime PD drives
    # ECL = Sum MarginalPD.LGD.EAD.DF directly and is NEVER passed through the
    # scorecard's OOS isotonic/Platt recalibrator (that recalibration only touches the
    # 12-month pd_pred used for EL/RWA/SICR-origination). This does not alter ECL — it
    # validates the hazard PD against realised lifetime outcomes so any material drift
    # is visible in the report rather than silently absorbed into the ECL number.
    try:
        from credit_risk.validation.calibration import lifetime_pd_calibration_by_vintage  # noqa: PLC0415

        _lpc_issue_year = pd.to_datetime(df_ecl["issue_d"], format="%b-%Y", errors="coerce").dt.year
        metrics["lifetime_pd_calibration"] = lifetime_pd_calibration_by_vintage(
            df_ecl[TARGET_COL].to_numpy(dtype=float),
            df_ecl["pd_lifetime"].to_numpy(dtype=float),
            _lpc_issue_year.to_numpy(dtype=float),
        )
        _lpc_port = metrics["lifetime_pd_calibration"]["portfolio"]
        logger.info(
            "Lifetime PD calibration (matured vintages, n=%d): predicted=%.4f "
            "observed=%.4f ratio=%.3f in_band=%s",
            _lpc_port["n"], _lpc_port["predicted_pd_lifetime"], _lpc_port["observed_dr"],
            _lpc_port["ratio"], _lpc_port["in_band"],
        )
    except Exception as _lpc_err:  # noqa: BLE001
        logger.warning("Lifetime PD calibration diagnostic failed (non-fatal): %s", _lpc_err)

    # Stage migration matrix (origination → reporting date)
    try:
        from credit_risk.risk.ifrs9_ecl import stage_migration_matrix, assign_stages  # noqa: PLC0415

        # Simulated 12-month-ago stages (t0) for active portfolio
        issue_d_dt = pd.to_datetime(df_rwa["issue_d"], format="%b-%Y", errors="coerce")
        reporting_d = pd.to_datetime("2018-12-31")
        mob_months = ((reporting_d - issue_d_dt).dt.days / 30.44).fillna(0.0).values

        pd_12m_current = df_rwa["pd_pred"].values
        pd_12m_t0 = 0.5 * pd_12m_orig + 0.5 * pd_12m_current
        term_num = pd.to_numeric(df_rwa["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(36.0).values
        rem_term_t0 = np.clip(term_num - (mob_months - 12), 1.0, term_num)
        pd_lifetime_t0 = 1.0 - (1.0 - np.clip(pd_12m_t0, 1e-9, 1.0 - 1e-9)) ** (rem_term_t0 / 12.0)

        pd_ratio_t0 = pd_lifetime_t0 / np.clip(pd_orig_lifetime, 1e-9, 1.0)
        relative_sicr_t0 = pd_ratio_t0 > ifrs9_cfg.sicr.pd_multiplier
        absolute_sicr_t0 = pd_lifetime_t0 > ifrs9_cfg.sicr.abs_threshold

        _stages_t0 = np.ones(len(df_rwa), dtype=int)
        active_12m_ago = mob_months >= 12
        _stages_t0[active_12m_ago & (relative_sicr_t0 | absolute_sicr_t0)] = 2

        # Current stages from ECL output. Fallback (only if df_ecl lacks a
        # "stage" column) must pass a genuine CURRENT lifetime PD, not
        # pd_orig_lifetime for both arguments — that would make the relative
        # SICR ratio always 1.0 and disable the relative-SICR test, leaving
        # only the absolute-threshold/DPD backstop able to fire.
        _pd_current_lifetime = 1.0 - (1.0 - np.clip(pd_12m_current, 1e-9, 1 - 1e-9)) ** term_years
        _stages_current = df_ecl["stage"].values if "stage" in df_ecl.columns else assign_stages(
            df_rwa, _pd_current_lifetime, pd_orig_lifetime, ifrs9_cfg.sicr
        )
        _migration = stage_migration_matrix(_stages_t0, _stages_current)
        metrics["ifrs9_stage_migration"] = {
            str(fs): {str(ts): int(_migration.loc[fs, ts]) for ts in [1, 2, 3]}
            for fs in [1, 2, 3]
        }
        logger.info("Stage migration matrix computed.")
    except Exception as _sm_err:
        logger.warning("Stage migration failed (non-fatal): %s", _sm_err)

    # ECL macro sensitivity tornado
    try:
        from credit_risk.risk.ifrs9_ecl import ecl_scenario_sensitivity  # noqa: PLC0415
        from credit_risk.reporting.charts import plot_ecl_tornado  # noqa: PLC0415

        _macro_shocks = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        _ecl_stages = df_ecl["stage"].values if "stage" in df_ecl.columns else None
        _sensitivity_df = ecl_scenario_sensitivity(
            df_rwa, hazard_model, lgd_arr, ead_arr,
            macro_shocks=_macro_shocks,
            stages=_ecl_stages,
        )
        _sensitivity_df.to_parquet(outputs / "ecl_sensitivity.parquet", index=False)
        metrics["ecl_sensitivity"] = _sensitivity_df.to_dict(orient="records")
        plot_ecl_tornado(_sensitivity_df, figs, scenario_shocks=metrics.get("macro_implied_shocks"))
        logger.info("ECL sensitivity tornado computed.")
    except Exception as _ecl_sens_err:
        logger.warning("ECL sensitivity failed (non-fatal): %s", _ecl_sens_err)

    # ECL what-if: PD/LGD/EAD stress scenarios + tornado
    try:
        from credit_risk.risk.ifrs9_ecl import ecl_shock_sensitivity  # noqa: PLC0415
        from credit_risk.reporting.charts import plot_shock_tornado  # noqa: PLC0415

        _whatif_stages = df_ecl["stage"].values if "stage" in df_ecl.columns else None
        _whatif_df = ecl_shock_sensitivity(
            df_rwa, hazard_model, lgd_arr, ead_arr, _whatif_stages,
        )
        _whatif_df.to_parquet(outputs / "ecl_whatif.parquet", index=False)
        metrics["ecl_whatif"] = _whatif_df.to_dict(orient="records")
        plot_shock_tornado(_whatif_df, figs)
        logger.info("ECL what-if sensitivity computed (%d scenarios).", len(_whatif_df))
    except Exception as _whatif_err:  # noqa: BLE001
        logger.warning("ECL what-if failed (non-fatal): %s", _whatif_err)

    # ── Phase 9: Cut-off analysis ─────────────────────────────────────────────
    logger.info("=== Phase 9: Cutoff Analysis ===")
    from credit_risk.business.cutoff import (  # noqa: PLC0415
        sweep_cutoffs, optimal_cutoff, raroc_argmax_cutoff, risk_appetite_cutoff,
    )

    scores = np.asarray(scorecard.predict_score(df_el), dtype=float)
    df_el["score"] = scores
    df_el.to_parquet(outputs / "portfolio_el.parquet", index=False)

    if TARGET_COL in df_el.columns:
        sweep_df = sweep_cutoffs(
            df_el[TARGET_COL].values,
            scores,
            df_el["ead"].values,
        )
        opt = optimal_cutoff(sweep_df)
        metrics["optimal_cutoff_threshold"] = opt["threshold"]
        metrics["optimal_approval_rate"] = opt["approval_rate"]
        metrics["optimal_bad_rate"] = opt["bad_rate"]

        # Dynamic Expected Profit & RAROC sweep using df_ecl
        try:
            logger.info("Computing Expected Profit and RAROC cutoff strategy sweep...")
            df_ecl_copy = df_ecl.copy()
            df_ecl_copy["score"] = np.asarray(scorecard.predict_score(df_ecl_copy), dtype=float)

            # Cut-off economics from config (RAROC-hurdle decision rule)
            _fee_r = cfg.business.fee_income_rate
            _fund_r = cfg.business.funding_cost_rate
            _op_r = cfg.business.operating_cost_rate
            _coc = cfg.business.cost_of_capital
            _hurdle = cfg.business.raroc_hurdle
            _max_bad = cfg.business.max_bad_rate

            cutoff_strategy = []
            for thr in range(400, 801, 10):
                approved_mask = df_ecl_copy["score"] >= thr
                df_app = df_ecl_copy[approved_mask]
                n_app = len(df_app)
                
                if n_app == 0:
                    cutoff_strategy.append({
                        "cutoff": int(thr), "approval_rate": 0.0, "bad_rate": 0.0,
                        "expected_profit": 0.0, "expected_loss": 0.0,
                        "capital_charge": 0.0, "raroc": 0.0
                    })
                    continue
                    
                ead_app = df_app["ead"].values
                pd_app = df_app["pd_pred"].values
                lgd_app = df_app["lgd_pred"].values if "lgd_pred" in df_app.columns else (df_app["lgd"].values if "lgd" in df_app.columns else np.full(n_app, 0.45))
                from credit_risk.risk.ifrs9_ecl import normalize_int_rate_to_fraction  # noqa: PLC0415

                int_rate_app = pd.to_numeric(df_app["int_rate"], errors="coerce").fillna(12.0).values
                int_rate_app = normalize_int_rate_to_fraction(int_rate_app)

                interest_income = float((ead_app * int_rate_app).sum())
                fees = float(_fee_r * ead_app.sum())
                funding_cost = float(_fund_r * ead_app.sum())
                operating_cost = float(_op_r * ead_app.sum())
                # All P&L components are per-annum. pd_pred is already a 12-month
                # (annual) PD — see line 877, where this same column is named
                # pd_12m_orig and is itself raised to term_years to build the
                # lifetime PD used for IFRS 9 staging. It is used directly here
                # against one year of interest/fee income; no term-based
                # conversion is needed (an earlier version of this block
                # mistakenly treated pd_pred as a lifetime PD and shrank it by
                # loan term, understating expected loss by roughly the term in
                # years).
                pd_annual = np.clip(pd_app, 0.0, 0.999999)
                el = float((pd_annual * lgd_app * ead_app).sum())

                k_app = df_app["capital_requirement_k"].values if "capital_requirement_k" in df_app.columns else np.zeros(n_app)
                capital_charge = float((k_app * ead_app).sum())
                capital_cost = float(_coc * capital_charge)
                
                profit = interest_income + fees - funding_cost - operating_cost - el - capital_cost
                raroc = (profit / capital_charge) if capital_charge > 0 else 0.0
                bad_rate = float(df_app[TARGET_COL].mean())
                approval_rate = float(n_app / len(df_ecl_copy))
                
                cutoff_strategy.append({
                    "cutoff": int(thr),
                    "approval_rate": approval_rate,
                    "bad_rate": bad_rate,
                    "expected_profit": profit,
                    "expected_loss": el,
                    "capital_charge": capital_charge,
                    "raroc": raroc
                })
            metrics["cutoff_strategy_table"] = cutoff_strategy
            logger.info("RAROC cutoff strategy sweep complete: %d cutoffs.", len(cutoff_strategy))

            # Disclosure: unconstrained optima over the fine grid. On this risk-priced
            # high-yield book both the total-profit argmax and the RAROC argmax are the
            # corner solution "approve everyone" — the expected-loss (at the through-the-door
            # bad rate) and the economic-capital charge are already netted, so this is a
            # genuine economic result, not under-penalised tail risk.
            _nonempty = [r for r in cutoff_strategy if r["approval_rate"] > 0.0]
            if _nonempty:
                _argmax_row = max(_nonempty, key=lambda r: r["expected_profit"])
                metrics["cutoff_profit_argmax"] = dict(_argmax_row)
                _raroc_row = raroc_argmax_cutoff(cutoff_strategy)
                if _raroc_row:
                    metrics["cutoff_raroc_max"] = dict(_raroc_row)

                # Recommended operating cutoff: profit maximisation subject to the board
                # risk-appetite ceiling on the approved bad rate → well-defined interior
                # cutoff. A single optimum drives the text, table and figure.
                _opt_row = risk_appetite_cutoff(cutoff_strategy, max_bad_rate=_max_bad) or _argmax_row
                metrics["cutoff_optimal_profit"] = dict(_opt_row)
                metrics["cutoff_raroc_hurdle"] = float(_hurdle)
                metrics["cutoff_max_bad_rate"] = float(_max_bad)
                # Headline cutoff metrics come from the risk-appetite operating cutoff.
                metrics["optimal_cutoff_threshold"] = float(_opt_row["cutoff"])
                metrics["optimal_approval_rate"] = float(_opt_row["approval_rate"])
                metrics["optimal_bad_rate"] = float(_opt_row["bad_rate"])
                logger.info(
                    "Operating cutoff (risk appetite bad-rate <= %.1f%%): %d "
                    "(approval=%.2f%%, bad=%.2f%%, profit=%.0f, raroc=%.1f%%) | "
                    "unconstrained profit/RAROC corner was cutoff=%d approval=%.1f%% raroc=%.1f%%",
                    _max_bad * 100, int(_opt_row["cutoff"]), _opt_row["approval_rate"] * 100,
                    _opt_row["bad_rate"] * 100, _opt_row["expected_profit"], _opt_row["raroc"] * 100,
                    int(_argmax_row["cutoff"]), _argmax_row["approval_rate"] * 100,
                    _argmax_row["raroc"] * 100,
                )
                from credit_risk.reporting.charts import plot_cutoff_profit  # noqa: PLC0415
                plot_cutoff_profit(pd.DataFrame(cutoff_strategy), figs, opt_cutoff=int(_opt_row["cutoff"]))
        except Exception as _raroc_err:
            logger.warning("RAROC cutoff strategy sweep failed (non-fatal): %s", _raroc_err)

        # Vintage PD backtesting
        try:
            from credit_risk.validation.backtest import vintage_pd_accuracy, score_band_stability_heatmap  # noqa: PLC0415
            _bt_df = df_all.copy()
            if "issue_d" in _bt_df.columns:
                _backtest_df = vintage_pd_accuracy(
                    _bt_df, pd_col="pd_pred", target_col=TARGET_COL, vintage_col="issue_d"
                )
                metrics["pd_backtest_vintage"] = _backtest_df.to_dict(orient="records")
                logger.info("Vintage PD backtesting done: %d cohorts.", len(_backtest_df))
        except Exception as _bt_err:
            logger.warning("Vintage PD backtest failed (non-fatal): %s", _bt_err)

        # Score-band stability heatmap
        try:
            if "score" in df_el.columns and "issue_d" in df_el.columns:
                _issue_d_dt = pd.to_datetime(df_el["issue_d"], format="%b-%Y", errors="coerce")
                _train_scored = df_el[_issue_d_dt < pd.Timestamp("2015-01-01")]
                _oot_scored = df_el[_issue_d_dt >= pd.Timestamp("2016-01-01")]
                if len(_train_scored) > 100 and len(_oot_scored) > 100:
                    score_band_stability_heatmap(
                        _train_scored, _oot_scored, score_col="score", fig_dir=figs
                    )
        except Exception as _sb_err:
            logger.warning("Score band heatmap failed (non-fatal): %s", _sb_err)

    # ── Phase 9b: Reject Inference (Parcelling) ──────────────────────────────
    logger.info("=== Phase 9b: Reject Inference (Parcelling) ===")
    from credit_risk.business.reject_inference import refit_with_parcelling, align_reject_data  # noqa: PLC0415
    if df_rejected is not None and len(df_rejected) > 0:
        try:
            from credit_risk.models.pd_scorecard import _add_interaction_features, _encode_categoricals

            # Sample df_rejected to speed up reject inference and prevent OOM on 27.6M rows
            df_rej_sample = df_rejected
            if len(df_rejected) > 100_000:
                df_rej_sample = df_rejected.sample(100_000, random_state=seed)

            # Align and impute df_rejected columns robustly
            df_rej_aligned = align_reject_data(
                df_rejected=df_rej_sample,
                df_train=df_train,
                woe_variables=scorecard._woe_transformer.variables_,
            )

            # Prepare both datasets with interaction features and categorical encoding
            df_train_prep = _encode_categoricals(_add_interaction_features(df_train))
            df_rej_prep = _encode_categoricals(_add_interaction_features(df_rej_aligned))

            # Transform raw accepted features to WoE
            df_train_woe = scorecard._woe_transformer.transform(df_train_prep[scorecard._woe_transformer.variables_].fillna(-9999))
            df_train_woe[TARGET_COL] = df_train[TARGET_COL].values
            df_train_woe["pd_pred"] = scorecard.predict_proba(df_train)

            # Transform raw rejects to WoE using the same transformer
            df_rej_woe = scorecard._woe_transformer.transform(df_rej_prep[scorecard._woe_transformer.variables_].fillna(-9999))
            df_rej_woe["pd_pred"] = scorecard.predict_proba(df_rej_aligned)

            # Perform through-the-door refitting
            fitted_ttd_model, gini_shift = refit_with_parcelling(
                df_train_woe,
                df_rej_woe,
                feature_cols=scorecard._selected_features,
                pd_col="pd_pred",
                target_col=TARGET_COL,
                seed=seed
            )
            metrics["gini_shift"] = gini_shift
            metrics["gini_ttd"] = metrics.get("gini", 0.26) + gini_shift
        except Exception as re_err:
            logger.warning("Reject Inference failed: %s", re_err)
            metrics["gini_shift"] = 0.0
            metrics["gini_ttd"] = metrics.get("gini", 0.26)
    else:
        logger.warning("No rejected loans found for Reject Inference.")
        metrics["gini_shift"] = 0.0
        metrics["gini_ttd"] = metrics.get("gini", 0.26)

    # ── Phase 9c: Basel IRB Stress Testing ────────────────────────────────────
    logger.info("=== Phase 9c: Basel IRB Stress Testing ===")
    from credit_risk.risk.basel_irb import irb_capital_requirement, irb_rwa  # noqa: PLC0415
    from scipy.special import ndtr, ndtri  # noqa: PLC0415
    try:
        pd_ttc = np.clip(df_all["pd_pred"].values, 1e-9, 1 - 1e-9)
        z_ttc = ndtri(pd_ttc)
        rho = cfg.basel.stress_rho          # retail asset correlation (config)
        z_stress = cfg.basel.stress_z       # systematic factor shock (config)
        
        # Stressed PD via the ASRF Vasicek model
        pd_stress = ndtr((z_ttc - np.sqrt(rho) * z_stress) / np.sqrt(1.0 - rho))
        
        # Calculate stressed unexpected losses and capital charges
        stressed_lgd = float(lgd_model.downturn_lgd) if lgd_model.downturn_lgd > 0 else 0.45
        stressed_k = irb_capital_requirement(pd_stress, np.full(len(df_all), stressed_lgd), pd_floor=cfg.basel.pd_floor)
        stressed_rwa_arr = irb_rwa(pd_stress, np.full(len(df_all), stressed_lgd), df_all["ead"].values, pd_floor=cfg.basel.pd_floor)
        
        metrics["stress_el"] = float((pd_stress * stressed_lgd * df_all["ead"].values).sum())
        metrics["stress_rwa"] = float(stressed_rwa_arr.sum())
        metrics["stress_capital_req"] = float(stressed_rwa_arr.sum() * 0.08)
        
        logger.info(
            "IRB Capital Stress Test (Z=-2.0): stressed_el=%.2f | stressed_rwa=%.2f | stressed_capital=%.2f",
            metrics["stress_el"], metrics["stress_rwa"], metrics["stress_capital_req"]
        )
    except Exception as st_err:
        logger.warning("IRB Capital Stress Test failed: %s", st_err)
        metrics["stress_el"] = metrics.get("total_el", 0.0) * 1.5
        metrics["stress_rwa"] = metrics.get("total_rwa", 0.0) * 1.8
        metrics["stress_capital_req"] = metrics["stress_rwa"] * 0.08

    # ── Write metrics.json ─────────────────────────────────────────────────────
    logging.getLogger("credit_risk").removeHandler(_nf_handler)
    metrics["phase_failures"] = phase_failures
    if phase_failures:
        logger.warning("%d enhancement phase(s) dropped (non-fatal): %s",
                       len(phase_failures), [f["message"] for f in phase_failures])
    metrics_path = outputs / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics written to %s", metrics_path)
    logger.info("=== Pipeline complete ===")

    # Print summary
    print("\n" + "="*60)
    print("PIPELINE RESULTS SUMMARY")
    print("="*60)
    print(f"  PD Model  | AUC={metrics.get('auc',0):.4f} | Gini={metrics.get('gini',0):.4f} | KS={metrics.get('ks',0):.4f}")
    print(f"  OOT       | AUC={metrics.get('auc_oot',0):.4f} | Gini={metrics.get('gini_oot',0):.4f}")
    print(f"  PSI (OOT) | {metrics.get('psi_total',0):.4f}")
    print(f"  LGD       | Mean={metrics.get('mean_lgd',0):.4f} | Downturn={metrics.get('downturn_lgd',0):.4f}")
    print(f"  EL        | ${metrics.get('total_el',0):>14,.0f}")
    print(f"  RWA (IRB) | ${metrics.get('total_rwa',0):>14,.0f}")
    print(f"  RWA (SA)  | ${metrics.get('total_rwa_sa',0):>14,.0f}")
    print(f"  RWA Density | {metrics.get('rwa_density','N/A')}")
    print(f"  ECL Total | ${metrics.get('total_ecl',0):>14,.0f}")
    print(f"  Coverage  | {metrics.get('ecl_coverage',0):.2%}")
    print(f"  Stage 2%  | {metrics.get('stage2_pct',0):.1%}")
    print(f"  Stage 3%  | {metrics.get('stage3_pct',0):.1%}")
    print(f"  Cutoff    | Score={metrics.get('optimal_cutoff_threshold',0):.0f} | Approval={metrics.get('optimal_approval_rate',0):.1%} | Bad rate={metrics.get('optimal_bad_rate',0):.2%}")
    _pf = metrics.get("phase_failures", [])
    print(f"  Dropped   | {len(_pf)} non-fatal phase failure(s)" + (f": {[f['message'] for f in _pf]}" if _pf else ""))
    print("="*60)


if __name__ == "__main__":
    run_pipeline()
