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
    """Construct interaction features capturing compounding credit risk signals."""
    out = df.copy()

    # Credit cycle stress: high DTI + low FICO → positive = high risk
    if "dti" in df.columns and "fico_range_low" in df.columns:
        dti_z = (df["dti"].fillna(df["dti"].median()) - df["dti"].median()) / (df["dti"].std() + 1e-9)
        fico_z = (df["fico_range_low"].fillna(df["fico_range_low"].median()) - df["fico_range_low"].median()) / (df["fico_range_low"].std() + 1e-9)
        out["dti_fico_interaction"] = dti_z * (-fico_z)

    # Loan affordability ratio
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        out["loan_to_income"] = df["loan_amnt"].fillna(0) / (df["annual_inc"].clip(lower=1).fillna(50000))

    # Revolving utilisation × open account count
    if "revol_util" in df.columns and "acc_open_past_24mths" in df.columns:
        out["revol_util_x_open_acc"] = (
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

        # Computed during fit
        self._factor: float = 0.0
        self._offset: float = 0.0
        self._woe_transformer: Any = None
        self._logit_result: Any = None
        self._selected_features: list[str] = []
        self._scorecard_table: pd.DataFrame = pd.DataFrame()
        self._calibrator: Any = None

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
        4. Logistic regression (statsmodels)
        5. Sign check + re-bin violators
        6. Scorecard scaling
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
        logger.info("PD model: %d candidate features", len(candidate_cols))

        # Step 1: WoE/IV
        woe_all = WoETransformer(variables=candidate_cols)
        woe_all.fit(X_train[candidate_cols].fillna(-9999), y_train)

        # Step 2: IV filter
        iv_tbl = woe_all.get_iv_table()
        iv_selected = filter_by_iv(iv_tbl, min_iv=self.min_iv, max_iv=self.max_iv)

        if len(iv_selected) == 0:
            raise ValueError(
                "No features passed IV filter. Check config iv thresholds or data quality."
            )

        # Re-fit WoE transformer on selected features only
        self._woe_transformer = WoETransformer(variables=iv_selected)
        self._woe_transformer.fit(X_train[iv_selected].fillna(-9999), y_train)
        X_woe = self._woe_transformer.transform(X_train[iv_selected].fillna(-9999))

        # Step 3: VIF filter
        self._selected_features = filter_by_vif(X_woe, max_vif=self.max_vif, y=y_train)
        X_woe_sel = X_woe[self._selected_features]

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

        if len(non_zero_feats) > 0:
            self._selected_features = non_zero_feats
            X_woe_sel = X_woe[self._selected_features]

        # Step 5: Final statsmodels logistic regression
        self._logit_result = self._fit_logistic(X_woe_sel, y_train)
        logger.info("Logistic regression fitted.\n%s", self._logit_result.summary2())

        # Step 5: Sign check
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

        # Step 6: Scorecard scaling
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
        X_fill = X[available].reindex(columns=woe_vars, fill_value=-9999).fillna(-9999)
        return self._woe_transformer.transform(X_fill)[self._selected_features]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Return credit score (higher = less risky)."""
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
            raw_pd = self._calibrator.transform(raw_pd.reshape(-1, 1)).ravel()

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

    def set_calibrator(self, calibrator: Any) -> None:
        """Attach an isotonic/Platt calibrator fitted on the test set."""
        self._calibrator = calibrator

    # ── Utilities ──────────────────────────────────────────────────────────────

    @property
    def scorecard_table(self) -> pd.DataFrame:
        return self._scorecard_table

    @property
    def feature_names(self) -> list[str]:
        return list(self._selected_features)

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
