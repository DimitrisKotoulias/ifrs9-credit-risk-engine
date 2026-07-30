"""LightGBM challenger model for PD benchmarking.

Trained on the same features as the scorecard (raw, not WoE-transformed)
for a fair interpretability vs. performance comparison.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_features(
    feature_names: list[str] | None,
    X_train: pd.DataFrame,
    model_label: str,
) -> list[str]:
    """Resolve the feature list, refusing to silently drop requested columns.

    Requested features that are absent from the frame used to be filtered out here. That
    made the champion-vs-challenger comparison quietly unfair: the scorecard's engineered
    columns (e.g. ``grade_enc``, ``term_enc``) are produced inside ``PDScorecard.fit`` and
    are missing from the raw training frame, so the challengers trained on a strictly
    smaller feature set than the champion while the report claimed parity
    (AUDIT-A12 / B2). Callers must engineer those columns first.
    """
    if feature_names is None:
        feature_names = list(X_train.select_dtypes(include="number").columns)

    # Target/date columns are legitimately excluded, never silently.
    exclude = {"target", "loan_status", "issue_d"}
    requested = [f for f in feature_names if f not in exclude]
    missing = [f for f in requested if f not in X_train.columns]
    if missing:
        raise ValueError(
            f"{model_label}: {len(missing)} requested feature(s) are absent from the "
            f"training frame: {missing}. Engineer them before fitting (see "
            "`_add_interaction_features` / `_encode_categoricals` in models.pd_scorecard) "
            "rather than fitting on a reduced set - that would make the champion/"
            "challenger comparison unfair."
        )
    return requested


class PDChallenger:
    """LightGBM challenger for the PD scorecard.

    Provides SHAP-based feature importance for interpretability analysis.

    Parameters
    ----------
    seed:
        Random seed.
    n_estimators:
        Max number of trees.
    learning_rate:
        LightGBM learning rate.
    num_leaves:
        Number of leaves per tree.
    """

    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
    ) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self._model: object | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: list[str] | None = None,
    ) -> "PDChallenger":
        """Fit LightGBM with early stopping on validation set.

        Parameters
        ----------
        X_train, y_train:
            Training data.
        X_val, y_val:
            Validation data for early stopping.
        feature_names:
            Columns to use; defaults to all numeric columns.
        """
        import lightgbm as lgb  # noqa: PLC0415

        feature_names = _resolve_features(feature_names, X_train, "PDChallenger")
        self._feature_names = feature_names

        X_tr = X_train[feature_names].astype(float)
        X_vl = X_val[feature_names].astype(float)

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "n_jobs": -1,
            "random_state": self.seed,
            "verbose": -1,
        }

        dtrain = lgb.Dataset(X_tr, label=y_train.values)
        dval = lgb.Dataset(X_vl, label=y_val.values, reference=dtrain)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

        self._model = lgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        logger.info(
            "Challenger fitted. Best iteration: %d",
            self._model.best_iteration,  # type: ignore[union-attr]
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return PD predictions."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.predict(  # type: ignore[union-attr]
            X[self._feature_names].astype(float)
        )

    def shap_values(self, X: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
        """Compute SHAP values.

        Parameters
        ----------
        X:
            Feature DataFrame.
        max_rows:
            Subsample for speed on large datasets.

        Returns
        -------
        pd.DataFrame
            SHAP values (one column per feature).
        """
        try:
            import shap  # noqa: PLC0415
        except ImportError:
            logger.warning("shap not installed; returning empty DataFrame.")
            return pd.DataFrame()

        if self._model is None:
            raise RuntimeError("Call fit() first.")

        X_sub = X[self._feature_names].astype(float).head(max_rows)
        explainer = shap.TreeExplainer(self._model)
        shap_arr = explainer.shap_values(X_sub)
        if isinstance(shap_arr, list):
            shap_arr = shap_arr[1]  # class 1 for binary
        return pd.DataFrame(shap_arr, columns=self._feature_names)

    def shap_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean absolute SHAP value per feature (global importance)."""
        sv = self.shap_values(X)
        if sv.empty:
            return pd.DataFrame()
        return (
            sv.abs()
            .mean()
            .reset_index()
            .rename(columns={"index": "feature", 0: "mean_abs_shap"})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Challenger saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "PDChallenger":
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        return obj


class PDMultiModelBenchmark:
    """Benchmark framework that trains, evaluates, and compares multiple models:
    1. Scorecard (Logistic Regression) - baseline
    2. LightGBM Classifier - boosting
    3. XGBoost Classifier - boosting
    4. Random Forest Classifier - bagging
    5. Weighted Ensemble
    """

    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
    ) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self._lgb_model: object | None = None
        self._xgb_model: object | None = None
        self._rf_model: object | None = None
        self._feature_names: list[str] = []
        self._medians: pd.Series | None = None
        self.lgb_train_time: float = 0.0
        self.xgb_train_time: float = 0.0
        self.rf_train_time: float = 0.0

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: list[str] | None = None,
    ) -> "PDMultiModelBenchmark":
        """Fit LightGBM, XGBoost, and RandomForestClassifier."""
        import lightgbm as lgb  # noqa: PLC0415
        import xgboost as xgb  # noqa: PLC0415
        from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415

        feature_names = _resolve_features(feature_names, X_train, "PDMultiModelBenchmark")
        self._feature_names = feature_names

        X_tr = X_train[feature_names].astype(float)
        X_vl = X_val[feature_names].astype(float)

        # 1. Fit LightGBM
        t0 = time.perf_counter()
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "n_jobs": -1,
            "random_state": self.seed,
            "verbose": -1,
        }
        dtrain = lgb.Dataset(X_tr, label=y_train.values)
        dval = lgb.Dataset(X_vl, label=y_val.values, reference=dtrain)
        callbacks = [lgb.early_stopping(50, verbose=False)]

        self._lgb_model = lgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            valid_sets=[dval],
            callbacks=callbacks,
        )
        self.lgb_train_time = time.perf_counter() - t0

        # 2. Fit XGBoost
        #
        # Native NaN, exactly like LightGBM above. It used to be handed `fillna(-9999)`,
        # which is a different missing-value policy from the other two models in a table
        # the report presents as a like-for-like benchmark — and it is the same sentinel
        # pattern the WoE binner was explicitly fixed to stop using, because it drops every
        # missing observation into the lowest numeric band of each feature. XGBoost learns a
        # default split direction for missing values on its own.
        t0 = time.perf_counter()
        X_tr_xgb = X_tr
        self._xgb_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=6,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
        self._xgb_model.fit(X_tr_xgb, y_train.values)
        self.xgb_train_time = time.perf_counter() - t0

        # 3. Fit Random Forest
        #
        # The only model here that cannot take NaN at all, so it gets train-median
        # imputation. That asymmetry is intrinsic to the estimator rather than a choice,
        # and the benchmark table footnote says so.
        t0 = time.perf_counter()
        self._medians = X_tr.median()
        X_tr_rf = X_tr.fillna(self._medians)

        self._rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._rf_model.fit(X_tr_rf, y_train.values)
        self.rf_train_time = time.perf_counter() - t0

        logger.info(
            "PDMultiModelBenchmark fitted successfully. LGB time: %.2fs, XGB time: %.2fs, RF time: %.2fs",
            self.lgb_train_time,
            self.xgb_train_time,
            self.rf_train_time,
        )
        return self

    def predict_proba_lgb(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using LightGBM."""
        if self._lgb_model is None:
            raise RuntimeError("Call fit() first.")
        X_sub = X[self._feature_names].astype(float)
        return self._lgb_model.predict(X_sub)

    def predict_proba_xgb(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using XGBoost."""
        if self._xgb_model is None:
            raise RuntimeError("Call fit() first.")
        # Native NaN at serve time too — the fit no longer uses the -9999 sentinel.
        X_sub = X[self._feature_names].astype(float)
        return self._xgb_model.predict_proba(X_sub)[:, 1]

    def predict_proba_rf(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using Random Forest."""
        if self._rf_model is None:
            raise RuntimeError("Call fit() first.")
        X_sub = X[self._feature_names].astype(float)
        X_sub_rf = X_sub.fillna(self._medians)
        return self._rf_model.predict_proba(X_sub_rf)[:, 1]

    def predict_proba_ensemble(
        self,
        X: pd.DataFrame,
        p_scorecard: np.ndarray,
        w_scorecard: float = 0.3,
        w_lgb: float = 0.3,
        w_xgb: float = 0.2,
        w_rf: float = 0.2,
    ) -> np.ndarray:
        """Weighted average ensemble of scorecard, LightGBM, XGBoost, and Random Forest."""
        p_lgb = self.predict_proba_lgb(X)
        p_xgb = self.predict_proba_xgb(X)
        p_rf = self.predict_proba_rf(X)
        p_ensemble = w_scorecard * p_scorecard + w_lgb * p_lgb + w_xgb * p_xgb + w_rf * p_rf
        return np.clip(p_ensemble, 1e-9, 1.0 - 1e-9)

