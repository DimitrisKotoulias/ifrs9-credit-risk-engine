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
    # Three distinct populations, reported separately so the report can never conflate
    # them (docs/AUDIT.md finding A5):
    #   n_accepted_file  - rows in the accepted-loans source file
    #   n_resolved       - loans with a resolved good/bad outcome (post target definition)
    #   n_modelling      - train + test + OOT, i.e. n_resolved minus the excluded grey zone
    n_accepted_file = int(getattr(split, "n_accepted_file", 0)) or (n_train + n_test + n_oot)
    n_resolved = len(split.full_accepted) if split.full_accepted is not None else (n_train + n_test + n_oot)
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
    # Which binner produced these bins. A silent fallback to the manual binner yields a
    # different model with different numbers, so it must be recorded, not just logged
    # (Flaws.md finding N32).
    _binner_kind = scorecard.binner_kind
    logger.info("WoE binning produced by: %s", _binner_kind)
    _selection_stages = scorecard.selection_stages
    _sc_tables = {
        "scorecard_table": scorecard.scorecard_table.to_dict(orient="records"),
        "iv_table": _iv_tbl.to_dict(orient="records"),
        "logit_coefficients": _coef_rows,
        "selected_features": scorecard.feature_names,
        "binner": _binner_kind,
        "feature_selection_stages": _selection_stages,
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

    # Chronological ordering of the OOT window so the recalibration gate can fit on the
    # earlier vintages and validate on the later ones. Without this the gate falls back to
    # row order, which on this data interleaves the two slices across 23 of 36 months and
    # silently turns an out-of-TIME test into a random split (docs/AUDIT.md finding A22).
    if "issue_d" not in df_oot.columns:
        raise ValueError(
            "df_oot has no 'issue_d'; the recalibration gate cannot split the OOT window "
            "chronologically and would silently degrade to a positional split."
        )
    _oot_order = pd.to_datetime(
        df_oot["issue_d"], format="%b-%Y", errors="coerce"
    ).to_numpy()

    val_metrics, calibrator = run_validation(
        y_train=y_train.values,
        y_pred_train=np.asarray(pd_train, dtype=float),
        y_test=y_test.values,
        y_pred_test=np.asarray(pd_test, dtype=float),
        y_oot=y_oot.values,
        y_pred_oot=np.asarray(pd_oot, dtype=float),
        oot_order_key=_oot_order,
        output_dir=outputs,
        fig_dir=figs / "validation",
    )
    if calibrator is not None:
        # Scope the transform to the era the gate learned it from. It is fitted on the
        # earlier half of the OOT window (2016+), and applying it to the 2007-2014
        # development vintages made them over-predict their realised default rate by up
        # to ~53% (Flaws.md finding N5).
        _oot_start_year = int(str(cfg.split.oot_cutoff)[:4])
        scorecard.set_calibrator(calibrator, min_issue_year=_oot_start_year)
        logger.info(
            "Recalibrator attached with vintage scope: %s", scorecard.calibration_scope
        )

    # Fit Model B (Pure Underwriting Scorecard - Circularity-free)
    logger.info("Fitting Model B (Pure Underwriting Scorecard)...")
    # Exported to metrics so the report enumerates what is actually withheld rather than
    # naming only int_rate/grade (Flaws.md finding N36). Note this list also removes loan
    # size and instalment, which are legitimate application variables, so Model B is a
    # conservative lower bound rather than a pricing-only ablation.
    model_b_excluded = [
        "int_rate", "grade_enc", "grade", "sub_grade", "sub_grade_enc",
        "loan_amnt", "funded_amnt", "funded_amnt_inv", "installment",
    ]
    scorecard_underwriting = PDScorecard(
        pdo=cfg.scorecard.pdo,
        base_score=cfg.scorecard.base_score,
        base_odds=cfg.scorecard.base_odds,
        exclude_features=model_b_excluded,
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
        from credit_risk.validation.calibration import spiegelhalter_test, compute_calibration_intercept_slope  # noqa: PLC0415
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

        # The recalibration decision belongs to the out-of-time gate in run_validation:
        # it triggers on OOT evidence, fits on the earlier OOT slice and accepts only on
        # demonstrated improvement on the later one. The previous shootout here fitted
        # Platt on TRAIN and scored it against raw PDs on TEST -- the wrong evidence for
        # an out-of-time drift -- and could attach a transform the gate had rejected
        # (docs/AUDIT.md finding A1).
        # (the calibrator, if the gate accepted one, is already attached above)
        _gate = val_metrics["calibration"].get("recalibration_gate", {})
        val_metrics["calibration"]["method_chosen"] = (
            _gate.get("chosen_method", "none") if calibrator is not None else "none"
        )

        # Before/after recalibration comparison.
        #
        # This block used to fit a THROWAWAY IsotonicRegression on the in-time test
        # partition and tabulate its effect on the full OOT set — a different transform
        # from the one actually deployed, on a different slice from the one the gate
        # judged. Every row of the resulting table moved the wrong way, directly beside
        # prose (correctly) reporting that the gate had accepted an improvement
        # (Flaws.md finding N6).
        #
        # The table is now built from the gate's own record: the deployed transform,
        # measured on the held-out later OOT slice it was accepted on.
        if _gate.get("eval_before") and _gate.get("eval_after"):
            val_metrics["calibration_comparison"] = {
                "measured_on": "gate_eval_slice",
                "recalibration_fit_on": "earlier_oot_slice",
                "transform": _gate.get("chosen_method", "none"),
                "applied_in_production": scorecard.has_calibrator,
                "n_eval": _gate.get("n_eval"),
                "eval_slice_min_date": _gate.get("eval_slice_min_date"),
                "eval_slice_max_date": _gate.get("eval_slice_max_date"),
                "before": dict(_gate["eval_before"]),
                "after": dict(_gate["eval_after"]),
            }
        else:
            # The gate did not reach the fit stage (not triggered, or slices too small),
            # so there is no before/after to show. Report the OOT position as-is rather
            # than manufacturing a comparison.
            val_metrics["calibration_comparison"] = {
                "measured_on": "full_oot",
                "recalibration_fit_on": "none",
                "transform": "none",
                "applied_in_production": scorecard.has_calibrator,
                "reason": (
                    _gate.get("skip_reason")
                    or _gate.get("trigger_reason")
                    or "recalibration gate produced no before/after evidence"
                ),
                "before": {
                    "auc": float(val_metrics["discrimination"]["oot"]["auc"]),
                    "brier": brier_before,
                    "intercept": intercept_before,
                    "slope": slope_before,
                    "expected_dr": expected_dr_before,
                    "actual_dr": actual_dr_before,
                    "hl_pvalue": float(val_metrics["calibration"]["oot"]["hl_pvalue"]),
                },
                "after": None,
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
        "n_accepted_file": int(n_accepted_file),
        "n_resolved_outcome": int(n_resolved),
        # Provenance of the PD model: which binner ran, and how many features survived
        # each of the four selection stages (Flaws.md findings N32, N29).
        "binner": _binner_kind,
        "feature_selection_stages": _selection_stages,
        "model_b_excluded_features": list(model_b_excluded),
        # Retained for backward compatibility with existing consumers. Historically this
        # key held the resolved-outcome count despite its name.
        "n_accepted_raw": int(n_resolved),
        "n_rejected_raw": int(n_rejected_raw),
        # Split composition — the report derives its percentages from these rather than
        # carrying hard-coded literals (docs/AUDIT.md findings A5, A6).
        "train_bad_rate": float(y_train.mean()),
        "test_bad_rate": float(y_test.mean()),
        "oot_bad_rate": float(y_oot.mean()),
        "underwriting_scorecard": {
            "train": disc_train_uw,
            "test": disc_test_uw,
            "oot": disc_oot_uw,
        },
        **val_metrics,
    }

    try:
        from credit_risk.validation.discrimination import RAGStatus
        # RAGStatus grades train-vs-OOT degradation, so it must be fed the TRAIN Gini.
        # It used to receive metrics["gini"], which is the *test* Gini — making the
        # headline "model stability" verdict a test-vs-OOT comparison that understates
        # the real degradation (Flaws.md finding N43).
        _gini_train = float(
            disc.get("train", {}).get("gini", metrics.get("gini", 0.0))
        )
        _rag = RAGStatus(
            gini_train=_gini_train,
            gini_oot=metrics.get("gini_oot", 0.0),
            psi=metrics.get("psi_total", 0.0),
        )
        metrics["rag_status"] = {
            "gini_rag": _rag.gini_rag,
            "psi_rag": _rag.psi_rag,
            "overall": _rag.overall,
            "gini_train": _gini_train,
            "gini_oot": float(metrics.get("gini_oot", 0.0)),
            "basis": "train_vs_oot",
        }
        logger.info("RAG Status: Gini=%s | PSI=%s | Overall=%s",
                    _rag.gini_rag, _rag.psi_rag, _rag.overall)
    except Exception as _e:
        logger.warning("RAG status failed: %s", _e)

    # ── Phase 2b: Challenger Model (LightGBM) ────────────────────────────────
    logger.info("=== Phase 2b: Challenger Model ===")
    try:
        from credit_risk.models.pd_challenger import PDChallenger  # noqa: PLC0415
        from credit_risk.models.pd_scorecard import (  # noqa: PLC0415
            _add_interaction_features, _encode_categoricals,
        )
        from credit_risk.validation.discrimination import compute_discrimination  # noqa: PLC0415

        # The scorecard's selected predictors include columns engineered inside
        # PDScorecard.fit (grade_enc, term_enc, interaction terms). Those are absent from
        # the raw frames, so the challengers must be handed the same engineered frames or
        # they silently train on a smaller feature set and the comparison is unfair
        # (docs/AUDIT.md A12 / B2). PDChallenger now raises rather than filtering.
        def _prep_features(frame: pd.DataFrame) -> pd.DataFrame:
            return _encode_categoricals(_add_interaction_features(frame))

        df_train_ch = _prep_features(df_train)
        df_test_ch = _prep_features(df_test)
        df_oot_ch = _prep_features(df_oot)
        _missing_ch = [f for f in scorecard.feature_names if f not in df_train_ch.columns]
        if _missing_ch:
            raise ValueError(
                f"Challenger feature parity broken: {_missing_ch} not produced by feature "
                "engineering; champion/challenger comparison would not be like-for-like."
            )
        logger.info(
            "Challenger feature parity: %d scorecard predictors available to challengers.",
            len(scorecard.feature_names),
        )

        challenger = PDChallenger(seed=seed)
        challenger.fit(
            df_train_ch, y_train,
            df_test_ch, y_test,
            feature_names=scorecard.feature_names,
        )
        challenger.save(outputs / "challenger.pkl")

        ch_pd_test = challenger.predict_proba(df_test_ch)
        ch_pd_oot = challenger.predict_proba(df_oot_ch)
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
                df_train_ch, y_train,
                df_test_ch, y_test,
                feature_names=scorecard.feature_names,
            )

            # Extract predictions
            sc_pd_test = np.asarray(pd_test, dtype=float)
            sc_pd_oot = np.asarray(pd_oot, dtype=float)

            lgb_pd_test = benchmark.predict_proba_lgb(df_test_ch)
            lgb_pd_oot = benchmark.predict_proba_lgb(df_oot_ch)

            xgb_pd_test = benchmark.predict_proba_xgb(df_test_ch)
            xgb_pd_oot = benchmark.predict_proba_xgb(df_oot_ch)

            rf_pd_test = benchmark.predict_proba_rf(df_test_ch)
            rf_pd_oot = benchmark.predict_proba_rf(df_oot_ch)

            ens_pd_test = benchmark.predict_proba_ensemble(df_test_ch, sc_pd_test)
            ens_pd_oot = benchmark.predict_proba_ensemble(df_oot_ch, sc_pd_oot)

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
            _shap_sample = df_oot_ch.sample(min(10_000, len(df_oot_ch)), random_state=seed)
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

                # The dual SHAP comparison (full model vs a second, bureau-only challenger)
                # used to be fitted and plotted here. Neither the figure nor the
                # shap_mean_abs_bureau metric was referenced anywhere in the report, so a
                # complete second challenger fit was being paid for nothing
                # (Flaws.md finding N37). plot_shap_comparison() remains in
                # reporting/charts.py. Model B in Section 3.6 already carries the
                # circularity argument this was meant to illustrate.

                # PDP and ICE plots used to be produced here, over 10,000 rows x 4
                # features. Neither figure was ever referenced by the report, so the cost
                # bought nothing (Flaws.md finding N37). plot_pdp_grid/plot_ice remain in
                # validation/interpretability.py for ad-hoc interrogation of a model.
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

    # Exposure is measured at the reporting date, so months on book is each loan's actual
    # age at that date rather than a flat fraction of its term (Flaws.md finding N10).
    ead_model = EADModel(reporting_date=cfg.reporting_date)
    ead_model.fit(df_train)
    metrics["ead_mob_basis"] = ead_model.mob_basis
    metrics["reporting_date"] = str(cfg.reporting_date)

    # ── Compute PD, LGD, EAD on full portfolio ────────────────────────────────
    df_all = pd.concat([df_train, df_test, df_oot], ignore_index=True)

    df_all["pd_pred"] = np.asarray(scorecard.predict_proba(df_all), dtype=float)

    # ── PD horizon: lifetime vs 12-month ─────────────────────────────────────
    #
    # The scorecard target is the loan's TERMINAL resolved status, so pd_pred is a
    # LIFETIME default probability. Three consumers require a ONE-YEAR PD instead:
    # the Basel IRB formula (BCBS §328), the per-annum P&L behind the cutoff sweep, and
    # the IRB stress test. Feeding them the lifetime figure produced a mean PD of ~0.245,
    # an RWA density of 228.8%, and an RWA that FELL under an extreme shock — the last
    # because the IRB capital function is concave and was being evaluated past its peak
    # (Flaws.md findings N1, N2, N18).
    #
    # Converting under a constant marginal-hazard assumption over the loan's term:
    #     PD_12m = 1 - (1 - PD_lifetime) ** (12 / term_months)
    # The lifetime figure is kept as-is for IFRS 9 staging, where it is the right horizon.
    _term_m = pd.to_numeric(
        df_all["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    ).fillna(36.0).to_numpy(dtype=float)
    _pd_life = np.clip(df_all["pd_pred"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    df_all["pd_12m"] = 1.0 - (1.0 - _pd_life) ** (12.0 / np.clip(_term_m, 1.0, None))

    metrics["mean_pd_lifetime"] = float(_pd_life.mean())
    metrics["mean_pd_12m"] = float(df_all["pd_12m"].mean())
    metrics["mean_pd_basel"] = float(df_all["pd_12m"].mean())
    metrics["pd_horizon_basel"] = "12m"
    metrics["pd_horizon_staging"] = "lifetime"
    logger.info(
        "PD horizons: lifetime mean=%.4f -> 12-month mean=%.4f (Basel/EL/RAROC use 12m)",
        metrics["mean_pd_lifetime"], metrics["mean_pd_12m"],
    )

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

    # 12-month PD: EL here is charged against a one-year horizon (Flaws.md N1).
    df_el = run_expected_loss(df_all, pd_col="pd_12m")
    df_el.to_parquet(outputs / "portfolio_el.parquet", index=False)
    el_summary = df_el.attrs.get("el_summary", {})
    metrics["total_el"] = el_summary.get("total_el", 0.0)
    metrics["total_ead_portfolio"] = el_summary.get("total_ead", 0.0)
    metrics["el_rate"] = el_summary.get("el_rate", 0.0)

    # ── Phase 7: Basel IRB ────────────────────────────────────────────────────
    logger.info("=== Phase 7: Basel IRB ===")
    from credit_risk.risk.basel_irb import run_basel_irb  # noqa: PLC0415

    downturn_lgd = float(lgd_model.downturn_lgd) if lgd_model.downturn_lgd > 0 else 0.45
    # BCBS §328 is a one-year formula (Flaws.md N1).
    df_rwa = run_basel_irb(
        df_el, pd_col="pd_12m", lgd_downturn=downturn_lgd, pd_floor=cfg.basel.pd_floor
    )
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

        # Headline EC uses the SAME supervisory correlation curve as the IRB calculation
        # it is compared against, so the EC/RegCap ratio reflects tail fidelity rather
        # than a 5x difference in rho (Flaws.md finding N13). The flat-rho run is retained
        # immediately below as an explicitly labelled correlation sensitivity.
        ec_losses, ec_measures = run_economic_capital(
            df_rwa,
            pd_col="pd_12m",
            rho="supervisory",
            n_sim=cfg.econ_cap.n_simulations,
            alpha=cfg.econ_cap.es_alpha,
            seed=cfg.econ_cap.seed,
            n_buckets=cfg.econ_cap.n_buckets,
        )
        reg_capital = float(metrics.get("total_rwa", 0.0)) * cfg.basel.capital_ratio

        try:
            _, _ec_flat = run_economic_capital(
                df_rwa,
                pd_col="pd_12m",
                rho=cfg.econ_cap.rho,
                n_sim=cfg.econ_cap.n_simulations,
                alpha=cfg.econ_cap.es_alpha,
                seed=cfg.econ_cap.seed,
                n_buckets=cfg.econ_cap.n_buckets,
            )
            metrics["econ_cap_rho_sensitivity"] = {
                "rho": cfg.econ_cap.rho,
                "economic_capital": _ec_flat["economic_capital"],
                "var": _ec_flat["var"],
                "es": _ec_flat["es"],
                "ec_to_reg_ratio": (
                    _ec_flat["economic_capital"] / reg_capital if reg_capital > 0 else 0.0
                ),
            }
        except Exception as _ec_sens_err:  # noqa: BLE001
            logger.warning("EC rho sensitivity failed (non-fatal): %s", _ec_sens_err)

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
            "rho": "supervisory (BCBS Other Retail curve, per PD bucket)",
            # A 99.9% quantile from 50,000 draws rests on ~50 tail observations, so the
            # figure is quoted with its own Monte Carlo standard error rather than to two
            # decimal places of false precision (Flaws.md finding N13, secondary).
            "var_mc_stderr": float(
                np.std(ec_losses[ec_losses >= ec_measures["var"]], ddof=1)
                / np.sqrt(max(1, int((ec_losses >= ec_measures["var"]).sum())))
            ),
            "n_tail_observations": int((ec_losses >= ec_measures["var"]).sum()),
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

    # Discrimination of the CHAMPION lifetime-PD model.
    #
    # This model produces the lifetime PD that drives the entire ECL, yet it carried no
    # discrimination metric at all, while its Cox *challenger* reported a C-index. The
    # champion was the unvalidated one; Section 6.2 only ever validated the PD *level*
    # (lifetime calibration by vintage), never the ranking (Flaws.md finding N44).
    try:
        from credit_risk.validation.discrimination import compute_discrimination  # noqa: PLC0415

        _hz_oot = hazard_model.predict_term_structure(df_oot, macro_shock=0.0)
        _hz_disc = compute_discrimination(
            y_oot.values,
            np.asarray(_hz_oot["pd_lifetime"], dtype=float),
            label="hazard_oot",
        )
        # For a binary outcome the C-index and the AUC coincide, so this is directly
        # comparable to the Cox challenger's reported C-index.
        metrics["hazard_model_discrimination"] = {
            "auc_oot": _hz_disc["auc"],
            "gini_oot": _hz_disc["gini"],
            "ks_oot": _hz_disc["ks"],
            "c_index_oot": _hz_disc["auc"],
            "basis": "lifetime PD, OOT partition, macro_shock=0",
        }
        logger.info(
            "Hazard model (champion) OOT discrimination: AUC=%.4f | Gini=%.4f",
            _hz_disc["auc"], _hz_disc["gini"],
        )
    except Exception as _hz_err:  # noqa: BLE001
        logger.warning("Hazard-model discrimination failed (non-fatal): %s", _hz_err)

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
    # Axes on which the upside and downside scenarios coincide (should always be empty;
    # qa_checks.check_scenario_axes_separate fails the build otherwise).
    metrics["degenerate_scenario_axes"] = list(
        macro_shocks.get("degenerate_scenario_axes", [])
    )
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

    # A term-scaled lifetime PD (pd_lifetime_sc) used to be derived here from the
    # scorecard PD. Its only consumer was the stage-migration reconstruction, which has
    # been withdrawn (Flaws.md finding N9); it was never passed to assign_stages as
    # `pd_orig_lifetime`, because doing so made the relative SICR ratio a comparison of
    # two models' PDs at the same date rather than a measure of credit deterioration
    # (docs/AUDIT.md finding B4). Both it and its inputs are therefore gone.

    # No genuine origination-date PD exists, so the relative SICR trigger is not
    # evaluable. Passing None skips it honestly; the absolute-threshold and delinquency
    # backstop triggers still fire inside assign_stages.
    metrics["sicr_relative_trigger_available"] = False
    df_ecl = run_ifrs9_ecl(
        df_rwa, hazard_model, lgd_arr, ead_arr,
        cfg=ifrs9_cfg, pd_orig_lifetime=None,
    )
    df_ecl.to_parquet(outputs / "ecl.parquet", index=False)

    ifrs9_summary = df_ecl.attrs.get("ifrs9_summary", {})
    metrics["total_ecl"] = ifrs9_summary.get("total_ecl", 0.0)
    metrics["ecl_coverage"] = ifrs9_summary.get("coverage_ratio", 0.0)
    stage_counts = ifrs9_summary.get("stage_counts", {})
    n_total = sum(stage_counts.values()) or 1
    metrics["stage2_pct"] = stage_counts.get(2, 0) / n_total
    metrics["stage3_pct"] = stage_counts.get(3, 0) / n_total
    # EL -> ECL reconciliation inputs (Flaws.md finding N28): the two headline provisions
    # differ by only ~3% while measuring very different things, and the report had no way
    # to show the reader why.
    metrics["ecl_reconciliation"] = {
        "total_el_12m": float(metrics.get("total_el", 0.0)),
        "total_ecl": float(ifrs9_summary.get("total_ecl", 0.0)),
        "staging_scenario": ifrs9_summary.get("staging_scenario"),
        "staging_macro_shock": ifrs9_summary.get("staging_macro_shock"),
        "ecl_by_stage": ifrs9_summary.get("ecl_by_stage", {}),
        "ead_by_stage": ifrs9_summary.get("ead_by_stage", {}),
        "n_by_stage": ifrs9_summary.get("n_by_stage", {}),
        "mean_pd_12m_by_stage": ifrs9_summary.get("mean_pd_12m_by_stage", {}),
        "mean_pd_lifetime_by_stage": ifrs9_summary.get("mean_pd_lifetime_by_stage", {}),
    }

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

    # No stage migration matrix is produced.
    #
    # It used to be built here from an "interpolated" t-12 credit profile,
    #     pd_12m_t0 = 0.5 * pd_12m_current_sc + 0.5 * pd_12m_current
    # where both terms were the SAME column, so the expression was the identity and the
    # t-12 state was just the reporting-date state under a different remaining-term
    # exponent. The published matrix consequently showed more Stage 2 -> Stage 1 cures
    # than Stage 1 -> Stage 2 deteriorations on a book whose default rate was rising, and
    # a structurally empty Stage 3 row. Without a monthly servicing panel no honest
    # migration matrix can be built from this data, so the report explains the absence
    # instead of presenting an artefact (Flaws.md finding N9).
    # stage_migration_matrix() remains in risk/ifrs9_ecl.py for use once panel data exists.

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
        raroc_argmax_cutoff, risk_appetite_cutoff,
    )

    scores = np.asarray(scorecard.predict_score(df_el), dtype=float)
    df_el["score"] = scores
    df_el.to_parquet(outputs / "portfolio_el.parquet", index=False)

    if TARGET_COL in df_el.columns:
        # A first sweep_cutoffs()/optimal_cutoff() pass used to run here over 200
        # thresholds x ~1M rows with cutoff.py's hardcoded profit_good=0.05 /
        # loss_bad=0.45 -- economics entirely disconnected from config.business -- and
        # wrote optimal_cutoff_* keys that the RAROC sweep below then overwrote. When the
        # RAROC block failed (it is wrapped in try/except) the report silently fell back
        # to those hardcoded numbers, with a continuous np.linspace threshold in place of
        # an integer grid cutoff: different economics, different semantics, no warning.
        # The single RAROC sweep below is now the only cutoff optimisation
        # (Flaws.md finding N21).
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
                        "risk_adjusted_return": 0.0, "capital_cost": 0.0,
                        "capital_charge": 0.0, "raroc": 0.0
                    })
                    continue
                    
                ead_app = df_app["ead"].values
                # 12-month PD, to match the one year of interest and fee income below.
                pd_app = (
                    df_app["pd_12m"].values if "pd_12m" in df_app.columns
                    else df_app["pd_pred"].values
                )
                lgd_app = df_app["lgd_pred"].values if "lgd_pred" in df_app.columns else (df_app["lgd"].values if "lgd" in df_app.columns else np.full(n_app, 0.45))
                from credit_risk.risk.ifrs9_ecl import normalize_int_rate_to_fraction  # noqa: PLC0415

                int_rate_app = pd.to_numeric(df_app["int_rate"], errors="coerce").fillna(12.0).values
                int_rate_app = normalize_int_rate_to_fraction(int_rate_app)

                interest_income = float((ead_app * int_rate_app).sum())
                fees = float(_fee_r * ead_app.sum())
                funding_cost = float(_fund_r * ead_app.sum())
                operating_cost = float(_op_r * ead_app.sum())
                # All P&L components are per-annum, so the PD must be too. pd_12m is
                # derived from the scorecard's lifetime PD where the portfolio is scored
                # (see the PD-horizon block above). A previous version of this comment
                # asserted pd_pred was "already a 12-month PD" and pointed at a variable
                # that no longer existed; it was wrong, and charging the full lifetime PD
                # against a single year of income overstated losses by roughly the loan
                # term (Flaws.md findings N1, N2).
                pd_annual = np.clip(pd_app, 0.0, 0.999999)
                el = float((pd_annual * lgd_app * ead_app).sum())

                k_app = df_app["capital_requirement_k"].values if "capital_requirement_k" in df_app.columns else np.zeros(n_app)
                capital_charge = float((k_app * ead_app).sum())
                capital_cost = float(_coc * capital_charge)

                # Two distinct quantities, deliberately kept apart (Flaws.md finding N12):
                #   economic_profit = revenue - costs - EL - cost of capital
                #   RAROC           = (revenue - costs - EL) / economic capital
                # RAROC must NOT net out the cost of capital: the hurdle comparison is
                # what accounts for it. Subtracting it first and then comparing to the
                # hurdle double-counts, and depressed every reported RAROC by exactly the
                # 12pp cost-of-capital rate.
                risk_adjusted_return = (
                    interest_income + fees - funding_cost - operating_cost - el
                )
                profit = risk_adjusted_return - capital_cost
                raroc = (risk_adjusted_return / capital_charge) if capital_charge > 0 else 0.0
                bad_rate = float(df_app[TARGET_COL].mean())
                approval_rate = float(n_app / len(df_ecl_copy))
                
                cutoff_strategy.append({
                    "cutoff": int(thr),
                    "approval_rate": approval_rate,
                    "bad_rate": bad_rate,
                    # expected_profit is ECONOMIC profit: net of the cost of capital.
                    # risk_adjusted_return is the RAROC numerator, before it.
                    "expected_profit": profit,
                    "risk_adjusted_return": risk_adjusted_return,
                    "capital_cost": capital_cost,
                    "expected_loss": el,
                    "capital_charge": capital_charge,
                    "raroc": raroc
                })
            metrics["cutoff_strategy_table"] = cutoff_strategy
            logger.info("RAROC cutoff strategy sweep complete: %d cutoffs.", len(cutoff_strategy))

            # Disclosure: unconstrained optima over the fine grid. Both the total-profit
            # argmax and the RAROC argmax are corner solutions; WHICH corner is empirical.
            # Expected loss (at the approved bad rate) and the economic-capital charge are
            # both netted out, so when every grid RAROC comes out negative the argmax is
            # the most EXCLUSIVE non-empty cutoff (a near-zero-volume book), not "approve
            # everyone". The report derives its wording from approval_rate rather than
            # asserting either case (docs/AUDIT.md finding A2).
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
                # The rate actually charged against economic capital in the profit
                # calculation. Distinct from the hurdle, which is only the comparison
                # threshold (docs/AUDIT.md finding A3).
                metrics["cutoff_cost_of_capital"] = float(_coc)
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
            # Fatal by design. This sweep is now the sole source of every cutoff number
            # in the report; letting it fail quietly would leave Section 9 with either
            # missing or (previously) silently substituted figures (Flaws.md finding N21).
            logger.error("RAROC cutoff strategy sweep failed: %s", _raroc_err)
            raise

        # Vintage PD backtesting
        try:
            from credit_risk.validation.backtest import vintage_pd_accuracy  # noqa: PLC0415
            _bt_df = df_all.copy()
            if "issue_d" in _bt_df.columns:
                _backtest_df = vintage_pd_accuracy(
                    _bt_df, pd_col="pd_pred", target_col=TARGET_COL, vintage_col="issue_d"
                )
                metrics["pd_backtest_vintage"] = _backtest_df.to_dict(orient="records")
                logger.info("Vintage PD backtesting done: %d cohorts.", len(_backtest_df))
        except Exception as _bt_err:
            logger.warning("Vintage PD backtest failed (non-fatal): %s", _bt_err)

        # The score-band stability heatmap used to be produced here. It was never
        # referenced by the report, and the per-feature CSI table (Section 7.6) answers
        # the same question in a form the report actually shows, so the figure is no
        # longer generated (Flaws.md finding N37). score_band_stability_heatmap() remains
        # available in validation/backtest.py for ad-hoc analysis.

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
            fitted_ttd_model, gini_shift, _ri_diag = refit_with_parcelling(
                df_train_woe,
                df_rej_woe,
                feature_cols=scorecard._selected_features,
                pd_col="pd_pred",
                target_col=TARGET_COL,
                seed=seed
            )
            metrics["gini_shift"] = gini_shift
            metrics["gini_ttd"] = metrics.get("gini", 0.26) + gini_shift
            metrics["reject_inference"] = _ri_diag
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
        # The stress test feeds the same one-year IRB formula as the base case, so it must
        # start from the same one-year TTC PD. Starting from the lifetime PD (~0.245) put
        # the concave IRB capital function past its peak, which is why stressing PD up to
        # ~60% made RWA *fall* by 2.6% (Flaws.md findings N1, N18).
        _pd_ttc_col = "pd_12m" if "pd_12m" in df_all.columns else "pd_pred"
        pd_ttc = np.clip(df_all[_pd_ttc_col].values, 1e-9, 1 - 1e-9)
        z_ttc = ndtri(pd_ttc)
        rho = cfg.basel.stress_rho          # retail asset correlation (config)
        z_stress = cfg.basel.stress_z       # systematic factor shock (config)
        
        # Stressed PD via the ASRF Vasicek model
        pd_stress = ndtr((z_ttc - np.sqrt(rho) * z_stress) / np.sqrt(1.0 - rho))
        
        # Calculate stressed unexpected losses and capital charges
        stressed_lgd = float(lgd_model.downturn_lgd) if lgd_model.downturn_lgd > 0 else 0.45
        stressed_k = irb_capital_requirement(pd_stress, np.full(len(df_all), stressed_lgd), pd_floor=cfg.basel.pd_floor)
        stressed_rwa_arr = irb_rwa(pd_stress, np.full(len(df_all), stressed_lgd), df_all["ead"].values, pd_floor=cfg.basel.pd_floor)
        
        # Isolate the PD effect: base EL uses the per-loan predicted LGD, so the stressed
        # EL must too, or the reported increase silently blends in an LGD switch
        # (docs/AUDIT.md finding B6). The downturn LGD stays where it belongs — in the
        # capital calculation, which uses it in the base case as well.
        _lgd_for_el = (
            df_all["lgd_pred"].values if "lgd_pred" in df_all.columns
            else np.full(len(df_all), stressed_lgd)
        )
        metrics["stress_el"] = float((pd_stress * _lgd_for_el * df_all["ead"].values).sum())
        metrics["stress_el_downturn_lgd"] = float(
            (pd_stress * stressed_lgd * df_all["ead"].values).sum()
        )
        metrics["stress_rwa"] = float(stressed_rwa_arr.sum())
        metrics["stress_capital_req"] = float(stressed_rwa_arr.sum() * cfg.basel.capital_ratio)
        
        logger.info(
            "IRB Capital Stress Test (Z=-2.0): stressed_el=%.2f | stressed_rwa=%.2f | stressed_capital=%.2f",
            metrics["stress_el"], metrics["stress_rwa"], metrics["stress_capital_req"]
        )
    except Exception as st_err:
        logger.warning("IRB Capital Stress Test failed: %s", st_err)
        metrics["stress_el"] = metrics.get("total_el", 0.0) * 1.5
        metrics["stress_rwa"] = metrics.get("total_rwa", 0.0) * 1.8
        metrics["stress_capital_req"] = metrics["stress_rwa"] * cfg.basel.capital_ratio

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
