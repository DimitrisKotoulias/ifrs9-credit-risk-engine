"""LGD (Loss Given Default) model.

Population: defaulted loans only.
Realised LGD = clip(1 − net_recoveries / (funded_amnt − total_rec_prncp), 0, 1),
where net_recoveries = recoveries − collection_recovery_fee. Principal-basis
throughout: total_pymnt (which includes interest/fees) is not used, since
mixing an interest-inclusive numerator with a principal-only denominator
understates LGD for loans that paid interest before charge-off.

Two-stage model:
  Stage 1 (cure): logistic regression for P(LGD > 0) — cure probability.
  Stage 2 (severity): beta regression for E[LGD | LGD > 0] via GLM with
    logit link and Binomial family (fraction response approximation).

Downturn LGD: conservative estimate at the configured percentile of
    the severity distribution — used for Basel IRB capital.

Note on CCF: Lending Club loans are fully drawn term instalment loans, so
EAD ≈ outstanding principal. A Credit Conversion Factor (CCF) approach is
the correct method for revolving/undrawn exposures; that is the Basel standard
but is NOT applicable here. This is an explicitly documented simplification.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_realised_lgd(df: pd.DataFrame) -> pd.Series:
    """Compute realised LGD on a principal-only basis:
    LGD = clip(1 - net_recoveries / (funded_amnt - total_rec_prncp), 0.0, 1.0)
    where net_recoveries = recoveries - collection_recovery_fee (clipped at 0).

    Both the numerator and denominator are principal-basis: total_pymnt is
    never used directly, since it bundles in interest/fee cash flows that
    are not part of principal loss and would otherwise understate LGD.
    """
    funded = pd.to_numeric(df["funded_amnt"], errors="coerce").fillna(1.0).clip(lower=1.0)
    trp = pd.to_numeric(df["total_rec_prncp"] if "total_rec_prncp" in df.columns else pd.Series(0.0, index=df.index), errors="coerce").fillna(0.0)
    rec = pd.to_numeric(df["recoveries"] if "recoveries" in df.columns else pd.Series(0.0, index=df.index), errors="coerce").fillna(0.0)
    crf = pd.to_numeric(df["collection_recovery_fee"] if "collection_recovery_fee" in df.columns else pd.Series(0.0, index=df.index), errors="coerce").fillna(0.0)
    net_recoveries = (rec - crf).clip(lower=0.0)

    ead_proxy = (funded - trp).clip(lower=1.0)
    lgd = (1.0 - net_recoveries / ead_proxy).clip(0.0, 1.0)
    return lgd.rename("lgd")


class LGDModel:
    """Two-stage LGD model: logistic cure + beta-regression severity.

    Parameters
    ----------
    downturn_percentile:
        Percentile of severity distribution used as conservative (downturn) LGD.
    seed:
        Random seed.
    """

    def __init__(self, downturn_percentile: float = 90.0, seed: int = 42) -> None:
        self.downturn_percentile = downturn_percentile
        self.seed = seed
        self._cure_model: object | None = None
        self._severity_model: object | None = None
        self._severity_type: str = "ridge"
        self._severity_scaler: object | None = None
        self._feature_cols: list[str] = []
        self._downturn_lgd: float = 0.0
        self._mean_lgd: float = 0.0
        self._challenger: object | None = None
        self._use_challenger: bool = False  # production predict path: two-stage vs challenger

    # ── Feature engineering ────────────────────────────────────────────────────

    @staticmethod
    def _default_features(df: pd.DataFrame) -> list[str]:
        exclude = {
            "target", "loan_status", "issue_d", "earliest_cr_line",
            "lgd", "recoveries", "collection_recovery_fee",
        }
        return [
            c for c in df.select_dtypes(include="number").columns
            if c not in exclude
        ]

    # ── Fitting ────────────────────────────────────────────────────────────────

    # Top features used for LGD to avoid singularity on small datasets
    _LGD_FEATURES = [
        "funded_amnt", "int_rate", "dti", "annual_inc",
        "fico_range_low", "grade_num", "term_num",
    ]

    def fit(
        self,
        df_defaults: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> "LGDModel":
        """Fit two-stage LGD model on defaulted-loans DataFrame.

        Parameters
        ----------
        df_defaults:
            Defaulted loans with 'recoveries' and 'funded_amnt'.
        feature_cols:
            Features to use. Defaults to a robust small feature set.
        """
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        if len(df_defaults) == 0:
            raise ValueError("df_defaults is empty — no defaulted loans to fit on.")

        lgd_series = compute_realised_lgd(df_defaults)
        y_cure = (lgd_series == 0.0).astype(int)  # 1 = cured (no loss)
        y_sev = lgd_series[lgd_series > 0]

        # Build a numeric grade column if available
        df_work = df_defaults.copy()
        if "grade" in df_work.columns:
            grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
            df_work["grade_num"] = df_work["grade"].map(grade_map).fillna(4.0)
        if "term" in df_work.columns:
            df_work["term_num"] = pd.to_numeric(
                df_work["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            ).fillna(36.0)

        if feature_cols is not None:
            self._feature_cols = [c for c in feature_cols if c in df_work.columns]
        else:
            self._feature_cols = [c for c in self._LGD_FEATURES if c in df_work.columns]
            if len(self._feature_cols) < 2:
                self._feature_cols = self._default_features(df_work)

        X = df_work[self._feature_cols].fillna(0.0).astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._severity_scaler = scaler

        # Stage 1: cure model (Gradient Boosting with balanced sample weights)
        if y_cure.sum() > 5 and (y_cure == 0).sum() > 5:
            from sklearn.ensemble import GradientBoostingClassifier  # noqa: PLC0415
            from sklearn.utils.class_weight import compute_sample_weight  # noqa: PLC0415

            sample_weights = compute_sample_weight("balanced", y_cure.values)
            cure_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.seed
            )
            cure_model.fit(X_scaled, y_cure, sample_weight=sample_weights)
            self._cure_model = cure_model
            logger.info("LGD cure model fitted (GBM with sample weights).")
        else:
            self._cure_model = None
            logger.warning("LGD: not enough cure/loss examples; skipping cure stage.")

        # Stage 2: severity model — fractional logit GLM (Papke & Wooldridge 1996)
        X_sev = X_scaled[lgd_series > 0]
        if len(X_sev) > 5:
            y_sev_clipped = y_sev.clip(0.001, 0.999).values
            try:
                import statsmodels.api as sm  # noqa: PLC0415

                X_sev_sm = sm.add_constant(X_sev, has_constant="add")
                frac_model = sm.GLM(
                    y_sev_clipped,
                    X_sev_sm,
                    family=sm.families.Binomial(link=sm.families.links.Logit()),
                )
                self._severity_model = frac_model.fit(disp=False, maxiter=200)
                self._severity_type = "fractional_logit"
                logger.info("LGD severity: fractional logit GLM fitted.")
            except Exception as glm_err:
                logger.warning("Fractional logit failed (%s); falling back to Ridge.", glm_err)
                from sklearn.linear_model import Ridge  # noqa: PLC0415
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_sev, y_sev_clipped)
                self._severity_model = ridge
                self._severity_type = "ridge"
        else:
            self._severity_model = None
            self._severity_type = "ridge"
            logger.warning("LGD: not enough severity examples; skipping severity stage.")

        # Downturn LGD
        all_preds = self.predict(df_defaults)
        loss_preds = all_preds[lgd_series > 0]
        self._downturn_lgd = float(np.percentile(loss_preds, self.downturn_percentile))
        self._mean_lgd = float(all_preds.mean())

        logger.info(
            "LGD summary: mean=%.4f | downturn (p%d)=%.4f",
            self._mean_lgd, int(self.downturn_percentile), self._downturn_lgd,
        )

        # LightGBM challenger
        self._fit_challenger(X_scaled, lgd_series)

        return self

    def _fit_challenger(self, X_scaled: np.ndarray, lgd: pd.Series) -> None:
        try:
            import lightgbm as lgb  # noqa: PLC0415

            from sklearn.model_selection import train_test_split  # noqa: PLC0415

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_scaled, lgd.values, test_size=0.2, random_state=self.seed
            )
            dtrain = lgb.Dataset(X_tr, y_tr)
            dval = lgb.Dataset(X_val, y_val, reference=dtrain)
            params = {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbose": -1,
                "random_state": self.seed,
            }
            self._challenger = lgb.train(
                params, dtrain, 200,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            logger.info("LGD challenger (LightGBM) fitted.")
        except Exception as exc:
            logger.warning("LGD challenger failed: %s", exc)

    # ── Prediction ─────────────────────────────────────────────────────────────

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        df_work = df.copy()
        if "grade" in df_work.columns:
            grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
            df_work["grade_num"] = df_work["grade"].map(grade_map).fillna(4.0)
        if "term" in df_work.columns:
            df_work["term_num"] = pd.to_numeric(
                df_work["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
            ).fillna(36.0)
        return df_work[self._feature_cols].fillna(0.0).astype(float)

    def predict_challenger(self, df: pd.DataFrame) -> pd.Series:
        """Predict point LGD with the LightGBM challenger (direct regression)."""
        if self._challenger is None:
            raise ValueError("No LGD challenger has been fitted.")
        X_scaled = self._severity_scaler.transform(self._prepare_X(df))  # type: ignore[union-attr]
        pred = np.clip(self._challenger.predict(X_scaled), 0.0, 1.0)  # type: ignore[attr-defined]
        return pd.Series(pred, index=df.index, name="lgd_pred")

    def promote_to_challenger(self, df_defaults: pd.DataFrame) -> None:
        """Switch the production predict path to the challenger and refresh the
        mean / downturn LGD summaries from its predictions on ``df_defaults``."""
        if self._challenger is None:
            raise ValueError("Cannot promote: no challenger fitted.")
        self._use_challenger = True
        lgd_series = compute_realised_lgd(df_defaults)
        preds = self.predict(df_defaults)
        loss_preds = preds[lgd_series > 0]
        if len(loss_preds) > 0:
            self._downturn_lgd = float(np.percentile(loss_preds, self.downturn_percentile))
        self._mean_lgd = float(preds.mean())
        logger.info("LGD challenger promoted: mean=%.4f | downturn=%.4f",
                    self._mean_lgd, self._downturn_lgd)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict point LGD (two-stage: P(loss) × E[severity|loss]).

        Routes to the LightGBM challenger when it has been promoted (better OOS fit).
        """
        if self._use_challenger and self._challenger is not None:
            return self.predict_challenger(df)
        X_raw = self._prepare_X(df)
        X_scaled = self._severity_scaler.transform(X_raw)  # type: ignore[union-attr]

        if self._cure_model is not None:
            p_cure = self._cure_model.predict_proba(X_scaled)[:, 1]
            p_loss = 1.0 - p_cure
        else:
            p_loss = np.full(len(df), 0.7)  # fallback: 70% loss probability

        if self._severity_model is not None:
            if self._severity_type == "fractional_logit":
                import statsmodels.api as sm  # noqa: PLC0415
                X_sm = sm.add_constant(X_scaled, has_constant="add")
                sev = np.clip(self._severity_model.predict(X_sm), 0.001, 0.999)
            else:
                sev = np.clip(self._severity_model.predict(X_scaled), 0.001, 0.999)
        else:
            sev = np.full(len(df), 0.5)  # fallback: 50% severity

        lgd_pred = np.clip(p_loss * sev, 0.0, 1.0)
        return pd.Series(lgd_pred, index=df.index, name="lgd_pred")

    @property
    def downturn_lgd(self) -> float:
        """Conservative downturn LGD for Basel IRB calculation."""
        return self._downturn_lgd

    @property
    def mean_lgd(self) -> float:
        return self._mean_lgd

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("LGD model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "LGDModel":
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301


def lgd_backtest(
    df_defaults: pd.DataFrame,
    lgd_pred: pd.Series,
    vintage_col: str = "issue_d",
) -> pd.DataFrame:
    """Compare predicted mean LGD vs realised LGD by vintage quarter.

    Returns DataFrame: vintage, n_defaults, predicted_mean_lgd, actual_mean_lgd, abs_error.
    """
    lgd_actual = compute_realised_lgd(df_defaults)
    df_work = pd.DataFrame({
        "vintage": pd.to_datetime(df_defaults[vintage_col], format="%b-%Y", errors="coerce").dt.to_period("Q"),
        "lgd_pred": lgd_pred.values,
        "lgd_actual": lgd_actual.values,
    }).dropna(subset=["vintage"])
    result = (
        df_work.groupby("vintage", observed=False)
        .agg(
            n_defaults=("lgd_actual", "count"),
            predicted_mean_lgd=("lgd_pred", "mean"),
            actual_mean_lgd=("lgd_actual", "mean"),
        )
        .reset_index()
    )
    result["abs_error"] = (result["predicted_mean_lgd"] - result["actual_mean_lgd"]).abs()
    result["vintage"] = result["vintage"].astype(str)
    return result
