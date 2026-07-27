"""PD Scorecard: WoE logistic regression + points-based scaling.

Implements the credit scorecard methodology:
1. WoE-transform features via optbinning (or fallback)
2. Fit logistic regression using statsmodels (for p-values & CIs)
3. Scale to a points-based scorecard with configurable PDO/base_score/base_odds
4. Provide score ↔ PD ↔ odds converters

Scorecard scaling (Appendix A):
    Score  = Offset + Factor · ln(odds)
    Factor = PDO / ln(2)
    Offset = base_score − Factor · ln(base_odds)

Points for attribute i:
    Points_i = (−(WoE_i · β_i) + α / n) · Factor + Offset / n
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
_HOME_MAP = {"OWN": 1, "RENT": 2, "MORTGAGE": 3, "OTHER": 4, "NONE": 4, "ANY": 4}

_EMP_LENGTH_ORDER = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8, "9 years": 9, "10+ years": 10,
}


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct interaction features capturing compounding credit risk signals.

    Features the loader (``data/loader.py``) already engineered are left alone. This
    function used to silently overwrite two of them: it rebuilt ``loan_to_income`` with
    different NaN handling, and it wrote a ``revol_util * acc_open_past_24mths`` product
    into the column named ``revol_util_x_open_acc`` --- which the loader had defined as
    ``revol_util * open_acc``. The report then interpreted the resulting IV/SHAP row under
    the wrong definition (Flaws.md finding N24). The interaction over recently opened
    accounts now carries its own name, ``revol_util_x_new_acc``.
    """
    out = df.copy()

    # Credit cycle stress: high DTI + low FICO → positive = high risk
    if "dti" in df.columns and "fico_range_low" in df.columns:
        dti_z = (df["dti"].fillna(df["dti"].median()) - df["dti"].median()) / (df["dti"].std() + 1e-9)
        fico_z = (df["fico_range_low"].fillna(df["fico_range_low"].median()) - df["fico_range_low"].median()) / (df["fico_range_low"].std() + 1e-9)
        out["dti_fico_interaction"] = dti_z * (-fico_z)

    # Loan affordability ratio — only if the loader has not already produced it.
    if (
        "loan_to_income" not in out.columns
        and "loan_amnt" in df.columns
        and "annual_inc" in df.columns
    ):
        out["loan_to_income"] = df["loan_amnt"].fillna(0) / (df["annual_inc"].clip(lower=1).fillna(50000))

    # Revolving utilisation × accounts opened in the last 24 months. Distinct from the
    # loader's revol_util_x_open_acc, which multiplies by total open accounts.
    if "revol_util" in df.columns and "acc_open_past_24mths" in df.columns:
        out["revol_util_x_new_acc"] = (
            df["revol_util"].fillna(50) / 100.0 * df["acc_open_past_24mths"].fillna(3)
        )
    return out


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode key categorical features so WoE binner can process them.

    Adds ``_enc`` suffix columns; leaves originals untouched.
    Only adds a column if the source column exists in *df*.
    """
    out = df.copy()

    if "grade" in df.columns:
        out["grade_enc"] = (
            df["grade"].astype(str).str.upper().str.strip()
            .map(_GRADE_MAP).fillna(4).astype(float)
        )

    if "term" in df.columns:
        # term is stored as " 36 months" or " 60 months"
        out["term_enc"] = (
            pd.to_numeric(
                df["term"].astype(str).str.extract(r"(\d+)")[0],
                errors="coerce",
            ).fillna(36.0)
        )

    if "emp_length" in df.columns:
        out["emp_length_enc"] = (
            df["emp_length"].astype(str).str.strip()
            .map(_EMP_LENGTH_ORDER).fillna(5.0).astype(float)
        )

    if "home_ownership" in df.columns:
        out["home_ownership_enc"] = (
            df["home_ownership"].astype(str).str.upper().str.strip()
            .map(_HOME_MAP).fillna(4).astype(float)
        )

    return out


def _select_pd_features(df: pd.DataFrame) -> list[str]:
    """Return numeric columns suitable for PD model (exclude IDs, dates, target).

    Includes ordinal-encoded categorical columns (``_enc`` suffix) if present.
    """
    exclude = {"target", "loan_status", "issue_d", "earliest_cr_line", "id", "member_id"}
    numeric = df.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric if c not in exclude]


class PDScorecard:
    """End-to-end WoE scorecard: feature selection → logistic → point scaling.

    Parameters
    ----------
    pdo:
        Points to double the odds.
    base_score:
        Credit score at base_odds.
    base_odds:
        Good-to-bad ratio at base_score.
    min_iv, max_iv:
        IV band for feature selection.
    max_vif:
        Maximum VIF; features above this are iteratively dropped.
    """

    def __init__(
        self,
        pdo: float = 20.0,
        base_score: float = 600.0,
        base_odds: float = 50.0,
        min_iv: float = 0.02,
        max_iv: float = 0.50,
        max_vif: float = 5.0,
        exclude_features: list[str] | None = None,
    ) -> None:
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.max_vif = max_vif
        self.exclude_features = exclude_features
        # Minimum share of the training rows a feature must actually observe before it can
        # be binned at all (Flaws.md finding N31, follow-on).
        self.min_bin_frac_for_selection = 0.01

        # Computed during fit
        self._factor: float = 0.0
        self._offset: float = 0.0
        self._woe_transformer: Any = None
        self._logit_result: Any = None
        self._selected_features: list[str] = []
        self._scorecard_table: pd.DataFrame = pd.DataFrame()
        self._calibrator: Any = None
        # Vintage scope of the calibrator — see set_calibrator (Flaws.md N5).
        self._calibration_min_issue_year: int | None = None
        # Feature counts surviving each selection stage, so the report can describe the
        # real four-stage funnel instead of the two stages it used to claim
        # (Flaws.md finding N29). Populated by fit().
        self._selection_stages: dict[str, Any] = {}
        self._n_dropped_sparse: int = 0
        self._dropped_sparse: list[str] = []

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> "PDScorecard":
        """Fit the full scorecard pipeline.

        Steps:
        1. Compute WoE/IV binning
        2. IV-band feature selection
        3. VIF filter
        4. ElasticNet (LogisticRegressionCV, SAGA) coefficient filter
        5. Logistic regression (statsmodels)
        6. Sign check, then refit on the survivors
        7. Scorecard scaling

        Stages 2-4 and 6 each drop features; the surviving counts are recorded in
        ``self._selection_stages`` so the report can state the real funnel
        (Flaws.md finding N29).
        """
        from credit_risk.features.selection import filter_by_iv, filter_by_vif, sign_check  # noqa: PLC0415
        from credit_risk.features.woe import WoETransformer  # noqa: PLC0415

        # Interaction features + ordinal encoding
        X_train = _add_interaction_features(X_train)
        X_test = _add_interaction_features(X_test)
        X_train = _encode_categoricals(X_train)
        X_test = _encode_categoricals(X_test)

        candidate_cols = _select_pd_features(X_train)
        if self.exclude_features is not None:
            candidate_cols = [c for c in candidate_cols if c not in self.exclude_features]

        # Drop features with too little observed data to bin at all.
        #
        # The LendingClub file carries columns that are entirely (or almost entirely)
        # empty for this population -- the joint-application and hardship blocks, for
        # instance. While missing values were being filled with the -9999 sentinel these
        # arrived at the binner as a *constant* column and were quietly dropped later by
        # the IV filter, contributing nothing. Passing genuine NaN (Flaws.md finding N31)
        # instead leaves the binner with zero observations to fit, so they are excluded
        # here, explicitly and with a count, rather than crashing the binner or being
        # silently neutralised by a sentinel.
        _min_obs = max(50, int(self.min_bin_frac_for_selection * len(X_train)))
        _sparse = []
        for col in candidate_cols:
            series = pd.to_numeric(X_train[col], errors="coerce")
            if series.notna().sum() < _min_obs or series.nunique(dropna=True) < 2:
                _sparse.append(col)
        if _sparse:
            candidate_cols = [c for c in candidate_cols if c not in _sparse]
            logger.info(
                "PD model: %d candidate feature(s) dropped as unbinnable (fewer than %d "
                "observed values, or constant): %s",
                len(_sparse), _min_obs, _sparse,
            )
        self._n_dropped_sparse = len(_sparse)
        self._dropped_sparse = list(_sparse)

        logger.info("PD model: %d candidate features", len(candidate_cols))

        # Step 1: WoE/IV
        woe_all = WoETransformer(variables=candidate_cols)
        # Missing values are passed through as NaN so the binner can give them their
        # own bin. They used to be filled with -9999 first, which dropped every missing
        # observation into the lowest numeric bin of each feature. Two consequences: the
        # binner's own "Missing" bin was structurally empty (N=0 in the points table, a
        # code path that never fired), and the risk direction implied by the imputation
        # was arbitrary per feature -- missing FICO landed in the WORST band while missing
        # DTI landed in the BEST one, and for mths_since_recent_bc (missing = never held a
        # bankcard) the assignment was actively the wrong sign (Flaws.md finding N31).
        woe_all.fit(X_train[candidate_cols], y_train)

        # Step 2: IV filter
        iv_tbl = woe_all.get_iv_table()
        iv_selected = filter_by_iv(iv_tbl, min_iv=self.min_iv, max_iv=self.max_iv)

        if len(iv_selected) == 0:
            raise ValueError(
                "No features passed IV filter. Check config iv thresholds or data quality."
            )

        # Re-fit WoE transformer on selected features only
        self._woe_transformer = WoETransformer(variables=iv_selected)
        self._woe_transformer.fit(X_train[iv_selected], y_train)
        X_woe = self._woe_transformer.transform(X_train[iv_selected])

        # Step 3: VIF filter
        self._selected_features = filter_by_vif(X_woe, max_vif=self.max_vif, y=y_train)
        X_woe_sel = X_woe[self._selected_features]

        self._selection_stages = {
            "n_dropped_unbinnable": int(self._n_dropped_sparse),
            "dropped_unbinnable": list(self._dropped_sparse),
            "n_candidates": int(len(candidate_cols)),
            "n_after_iv": int(len(iv_selected)),
            "n_after_vif": int(len(self._selected_features)),
            "iv_band": [float(self.min_iv), float(self.max_iv)],
            "max_vif": float(self.max_vif),
            "dropped_by_iv": sorted(set(candidate_cols) - set(iv_selected)),
            "dropped_by_vif": sorted(set(iv_selected) - set(self._selected_features)),
        }

        # Step 4: ElasticNet CV Feature Selection (SAGA)
        from sklearn.linear_model import LogisticRegressionCV  # noqa: PLC0415
        from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

        logger.info("Running ElasticNet CV feature selection on %d features...", len(self._selected_features))
        lr_cv = LogisticRegressionCV(
            Cs=np.logspace(-2, 2, 10),
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            penalty="elasticnet",
            solver="saga",
            l1_ratios=[0.5],
            max_iter=1000,
            n_jobs=-1,
            random_state=42
        )
        lr_cv.fit(X_woe_sel, y_train)

        coef_mask = np.abs(lr_cv.coef_[0]) > 1e-4
        non_zero_feats = [feat for feat, keep in zip(self._selected_features, coef_mask) if keep]
        logger.info("ElasticNet CV selected %d features: %s", len(non_zero_feats), non_zero_feats)

        _pre_elasticnet = list(self._selected_features)
        if len(non_zero_feats) > 0:
            self._selected_features = non_zero_feats
            X_woe_sel = X_woe[self._selected_features]
        self._selection_stages["n_after_elasticnet"] = int(len(self._selected_features))
        self._selection_stages["dropped_by_elasticnet"] = sorted(
            set(_pre_elasticnet) - set(self._selected_features)
        )

        # Step 5: Final statsmodels logistic regression
        self._logit_result = self._fit_logistic(X_woe_sel, y_train)
        logger.info("Logistic regression fitted.\n%s", self._logit_result.summary2())

        # Step 6: Sign check
        # WoE = log(pct_good / pct_bad), so higher WoE = lower risk.
        # In logistic regression predicting P(bad=1), coefficients on WoE
        # features should be NEGATIVE (higher WoE → lower P(bad)).
        coefs = pd.Series(
            self._logit_result.params[self._selected_features].values,
            index=self._selected_features,
        )
        violations = sign_check(coefs, expected_positive=False)
        if violations:
            logger.warning(
                "Dropping %d features with wrong sign: %s. "
                "Consider re-binning in a real project.",
                len(violations), violations,
            )
            self._selected_features = [f for f in self._selected_features if f not in violations]
            if len(self._selected_features) == 0:
                raise ValueError(
                    "All features dropped by sign check. "
                    "Check WoE encoding direction or relax sign constraints."
                )
            X_woe_sel = X_woe[self._selected_features]
            self._logit_result = self._fit_logistic(X_woe_sel, y_train)

        self._selection_stages["n_after_sign_check"] = int(len(self._selected_features))
        self._selection_stages["dropped_by_sign_check"] = sorted(violations)
        self._selection_stages["final_features"] = list(self._selected_features)

        # Step 7: Scorecard scaling
        self._factor = self.pdo / np.log(2)
        self._offset = self.base_score - self._factor * np.log(self.base_odds)
        self._build_scorecard_table()

        logger.info(
            "Scorecard built. Factor=%.3f, Offset=%.3f, features=%d",
            self._factor, self._offset, len(self._selected_features),
        )
        return self

    def _fit_logistic(self, X_woe: pd.DataFrame, y: pd.Series) -> Any:
        import statsmodels.api as sm  # noqa: PLC0415

        X_sm = sm.add_constant(X_woe.astype(float), has_constant="add")
        model = sm.Logit(y.astype(float), X_sm)
        return model.fit(disp=False, maxiter=200)

    def _build_scorecard_table(self) -> None:
        """Build scorecard point table per attribute.

        Handles both binner types:
        - ManualMonotonicBinner: uses woe_maps_ dict directly.
        - OptBinningWrapper: calls get_binning_table(feat) API to retrieve
          per-bin WoE, count, and event_rate from the optbinning process.
        """
        result = self._logit_result
        alpha = float(result.params.get("const", 0.0))
        n = len(self._selected_features)
        factor = self._factor
        offset = self._offset

        binner = self._woe_transformer._binner
        records = []

        for feat in self._selected_features:
            beta = float(result.params[feat])

            # ── ManualMonotonicBinner path ──────────────────────────────────
            woe_map = getattr(binner, "woe_maps_", {}).get(feat, {})
            if woe_map:
                bin_edges = getattr(binner, "bin_edges_", {}).get(feat, [])
                for bin_id, woe_val in woe_map.items():
                    points = (-woe_val * beta + alpha / n) * factor + offset / n
                    # Derive bin label from edges when available
                    try:
                        lo = bin_edges[bin_id]
                        hi = bin_edges[bin_id + 1]
                        bin_label = f"({lo:.2f}, {hi:.2f}]"
                    except (IndexError, TypeError):
                        bin_label = str(bin_id)
                    records.append({
                        "feature": feat,
                        "bin": bin_label,
                        "woe": float(woe_val),
                        "beta": float(beta),
                        "points": float(points),
                        "n_obs": None,
                        "dr": None,
                    })
                continue

            # ── OptBinningWrapper path ──────────────────────────────────────
            process = getattr(binner, "_process", None)
            if process is not None:
                try:
                    ob = process.get_binned_variable(feat).binning_table
                    bt = ob.build()
                    # optbinning table columns include: Bin, Count, WoE, IV, etc.
                    # Filter out summary rows (e.g. "Totals") which have non-string Bin
                    woe_col = next(
                        (c for c in bt.columns if c.lower() in ("woe", "woe value")),
                        None,
                    )
                    bin_col = next(
                        (c for c in bt.columns if c.lower() in ("bin", "bins", "interval")),
                        None,
                    )
                    count_col = next(
                        (c for c in bt.columns if c.lower() in ("count", "n", "total")),
                        None,
                    )
                    event_rate_col = next(
                        (c for c in bt.columns if "event" in c.lower() and "rate" in c.lower()),
                        None,
                    )
                    if woe_col is not None and bin_col is not None:
                        for _, row in bt.iterrows():
                            bin_val = row[bin_col]
                            woe_val = row[woe_col]
                            # Skip totals / special rows
                            if not isinstance(woe_val, (int, float)):
                                continue
                            if pd.isna(woe_val):
                                continue
                            woe_val = float(woe_val)
                            points = (-woe_val * beta + alpha / n) * factor + offset / n
                            records.append({
                                "feature": feat,
                                "bin": str(bin_val),
                                "woe": woe_val,
                                "beta": float(beta),
                                "points": float(points),
                                "n_obs": int(row[count_col]) if count_col and pd.notna(row.get(count_col)) else None,
                                "dr": float(row[event_rate_col]) if event_rate_col and pd.notna(row.get(event_rate_col)) else None,
                            })
                        continue
                except Exception as exc:
                    logger.warning(
                        "OptBinning binning table extraction failed for %s: %s — using placeholder.",
                        feat, exc,
                    )

            # ── Fallback placeholder (should not normally be reached) ────────
            logger.error(
                "Could not extract WoE bins for feature '%s'. "
                "Points will be zero — check binner compatibility.",
                feat,
            )
            records.append({
                "feature": feat, "bin": "all", "woe": 0.0,
                "beta": beta, "points": 0.0,
                "n_obs": None, "dr": None,
            })

        self._scorecard_table = pd.DataFrame(records)
        # Assertion guard
        if not self._scorecard_table.empty:
            n_zero = (self._scorecard_table["woe"] == 0.0).sum()
            n_total = len(self._scorecard_table)
            if n_zero == n_total:
                logger.error(
                    "SCORECARD TABLE: ALL %d bins have WoE=0. "
                    "Binner WoE extraction failed. Check binner type.",
                    n_total,
                )

    # ── Prediction ─────────────────────────────────────────────────────────────

    def _woe_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Ordinal-encode categoricals (adds _enc columns if source cols present)
        X = _encode_categoricals(X)
        # Must pass all variables the WoE transformer was fitted on,
        # then select only the VIF-surviving features from the output.
        woe_vars = self._woe_transformer.variables_
        # Keep only columns that exist in X (safety for inference on slim frames)
        available = [c for c in woe_vars if c in X.columns]
        # NaN is preserved so scoring takes the same Missing bin the fit created.
        X_fill = X[available].reindex(columns=woe_vars)
        return self._woe_transformer.transform(X_fill)[self._selected_features]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Return credit score (higher = less risky).

        The score is the points-based scaling of the *raw* logit output and deliberately
        does not pass through the recalibrator, so that it stays consistent with the
        Appendix A points table. When a calibrator is attached, ``score_to_pd`` and
        ``pd_to_score`` therefore describe the pre-recalibration mapping, not the PD that
        ``predict_proba`` returns; the report states this explicitly
        (Flaws.md finding N4, related item).
        """
        X_woe = self._woe_transform(X)
        import statsmodels.api as sm  # noqa: PLC0415

        X_sm = sm.add_constant(X_woe.astype(float), has_constant="add")
        prob_default = self._logit_result.predict(X_sm)
        # log-odds of GOOD (not default) → higher = safer
        log_odds_good = np.log((1 - prob_default + 1e-15) / (prob_default + 1e-15))
        return self._factor * log_odds_good + self._offset

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return PD (probability of default)."""
        X_woe = self._woe_transform(X)
        import statsmodels.api as sm  # noqa: PLC0415

        X_sm = sm.add_constant(X_woe.astype(float), has_constant="add")
        raw_pd = self._logit_result.predict(X_sm).values

        if self._calibrator is not None:
            raw_pd = self._apply_calibrator_in_scope(raw_pd, X)

        return raw_pd

    def score_to_pd(self, score: float | np.ndarray) -> np.ndarray:
        """Convert credit score to probability of default."""
        odds = np.exp((np.asarray(score) - self._offset) / self._factor)
        return 1.0 / (1.0 + odds)

    def pd_to_score(self, pd_val: float | np.ndarray) -> np.ndarray:
        """Convert PD to credit score."""
        pd_arr = np.asarray(pd_val)
        odds = (1.0 - pd_arr) / (pd_arr + 1e-15)
        return self._factor * np.log(odds) + self._offset

    def _apply_calibrator(self, raw_pd: np.ndarray) -> np.ndarray:
        """Apply the attached calibrator, whichever family it belongs to.

        Isotonic exposes ``.transform``; Platt scaling is a ``LogisticRegression`` and
        exposes ``.predict_proba``. The previous code called ``.transform`` unconditionally,
        so selecting Platt raised AttributeError and aborted the whole pipeline at the
        portfolio scoring step (docs/AUDIT.md finding A23).
        """
        raw_pd = np.asarray(raw_pd, dtype=float)
        if hasattr(self._calibrator, "predict_proba"):
            out = self._calibrator.predict_proba(raw_pd.reshape(-1, 1))[:, 1]
        elif hasattr(self._calibrator, "transform"):
            out = np.asarray(self._calibrator.transform(raw_pd.reshape(-1, 1))).ravel()
        else:
            raise TypeError(
                f"Calibrator {type(self._calibrator).__name__} exposes neither "
                "predict_proba nor transform."
            )
        return np.clip(out, 1e-8, 1 - 1e-8)

    def _apply_calibrator_in_scope(self, raw_pd: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        """Apply the calibrator only to the vintages it was fitted on.

        The gate fits on the earlier half of the OOT window, so the transform it learns
        describes the 2016+ era. Applying it across the whole 2007-2018 book left the
        development vintages over-predicting their realised default rate by up to ~53%,
        and that bias flowed straight into EL, RWA and every cutoff decision
        (Flaws.md finding N5). Out-of-scope rows keep their raw PD.

        With no scope set, or no usable issue date, behaviour is unchanged (global apply).
        """
        raw_pd = np.asarray(raw_pd, dtype=float)
        if self._calibration_min_issue_year is None or "issue_d" not in X.columns:
            return self._apply_calibrator(raw_pd)

        year = pd.to_datetime(X["issue_d"], format="%b-%Y", errors="coerce").dt.year
        in_scope = (year >= self._calibration_min_issue_year).to_numpy(dtype=bool)
        if not in_scope.any():
            return raw_pd

        out = raw_pd.copy()
        out[in_scope] = self._apply_calibrator(raw_pd[in_scope])
        return out

    def set_calibrator(
        self, calibrator: Any, min_issue_year: int | None = None
    ) -> None:
        """Attach the isotonic/Platt recalibrator chosen by the out-of-time gate.

        The gate fits on the earlier half of the OOT window and accepts only on measured
        improvement in the later half (``validation.calibration.select_oot_recalibrator``)
        — not on the test partition, as this docstring used to say.

        Parameters
        ----------
        min_issue_year:
            Restrict the transform to loans originated in or after this year, i.e. the era
            the gate actually learned from. ``None`` applies it to every row.
        """
        self._calibrator = calibrator
        self._calibration_min_issue_year = min_issue_year

    @property
    def calibration_scope(self) -> str:
        """Human-readable description of which vintages the calibrator touches."""
        if self._calibrator is None:
            return "none"
        if self._calibration_min_issue_year is None:
            return "all vintages"
        return f"{self._calibration_min_issue_year}+ vintages only"

    @property
    def has_calibrator(self) -> bool:
        """True when a recalibration transform is actually applied by ``predict_proba``.

        The report keys its calibration narrative off this: when it is False, every
        reported PD --- including those feeding EL, RWA and IFRS 9 staging --- is raw
        model output (docs/AUDIT.md finding A1).
        """
        return self._calibrator is not None

    # ── Utilities ──────────────────────────────────────────────────────────────

    @property
    def scorecard_table(self) -> pd.DataFrame:
        return self._scorecard_table

    @property
    def feature_names(self) -> list[str]:
        return list(self._selected_features)

    @property
    def selection_stages(self) -> dict[str, Any]:
        """Feature counts (and names) dropped at each selection stage — see fit()."""
        return dict(self._selection_stages)

    @property
    def binner_kind(self) -> str:
        """Which binning implementation actually produced the WoE encoding.

        Surfaced so a silent fallback to the manual binner cannot pass unnoticed into the
        report (Flaws.md finding N32).
        """
        from credit_risk.features.binning import binner_kind as _kind  # noqa: PLC0415

        if self._woe_transformer is None or self._woe_transformer._binner is None:
            return "unfitted"
        return _kind(self._woe_transformer._binner)

    @property
    def logit_summary(self) -> str:
        return self._logit_result.summary2().as_text() if self._logit_result else ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Scorecard saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "PDScorecard":
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        logger.info("Scorecard loaded from %s", path)
        return obj
