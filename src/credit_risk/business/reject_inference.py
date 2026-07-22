"""Reject inference — parcelling method.

Assigns probabilistic good/bad labels to rejected loans using the PD model's
score, then refits to estimate through-the-door population performance.

Parcelling:
    For each rejected loan, compute probability P(bad) from accept-population
    scorecard. Fractionally allocate:
        weight_bad  = P(bad)
        weight_good = 1 − P(bad)

    Replicate each rejected loan twice with these fractional weights.
    Combine with accepted population, refit logistic regression.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parcelling(
    df_accepted: pd.DataFrame,
    df_rejected: pd.DataFrame,
    score_col: str = "score",
    target_col: str = "target",
    pd_col: str = "pd_pred",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply parcelling to rejected loans and combine with accepted.

    Parameters
    ----------
    df_accepted:
        Accepted loans with observed target, score, pd_pred.
    df_rejected:
        Rejected loans with score/pd_pred from scorecard (no observed target).
    score_col:
        Score column name (higher = lower risk).
    target_col:
        Target column in accepted.
    pd_col:
        PD column used to assign fractional labels to rejects.
    feature_cols:
        Features to keep in the combined dataset.

    Returns
    -------
    pd.DataFrame
        Combined dataset with `weight` column and `target` (0 or 1).
        Accepted loans have weight=1; rejected get two rows (good/bad) with
        fractional weights.
    """
    acc = df_accepted.copy()
    acc["weight"] = 1.0
    acc["_source"] = "accepted"

    # Reject p(bad) from scorecard PD
    p_bad = df_rejected[pd_col].clip(0.001, 0.999).values

    rej_good = df_rejected.copy()
    rej_good[target_col] = 0
    rej_good["weight"] = 1.0 - p_bad
    rej_good["_source"] = "rejected_good"

    rej_bad = df_rejected.copy()
    rej_bad[target_col] = 1
    rej_bad["weight"] = p_bad
    rej_bad["_source"] = "rejected_bad"

    combined = pd.concat([acc, rej_good, rej_bad], ignore_index=True)

    logger.info(
        "Parcelling: %d accepted + %d rejected -> %d rows (%.1f%% reject weight)",
        len(acc), len(df_rejected), len(combined),
        (1.0 - p_bad).mean() * 100 + p_bad.mean() * 100,
    )
    return combined


def refit_with_parcelling(
    df_accepted: pd.DataFrame,
    df_rejected: pd.DataFrame,
    feature_cols: list[str],
    pd_col: str = "pd_pred",
    target_col: str = "target",
    seed: int = 42,
) -> tuple[object, float]:
    """Refit logistic model on accepted+parcelled-rejected population.

    Returns
    -------
    (fitted_model, gini_shift)
        gini_shift = Gini(through-door) − Gini(accepted-only).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    combined = parcelling(df_accepted, df_rejected, pd_col=pd_col, target_col=target_col)

    valid_cols = [c for c in feature_cols if c in combined.columns]
    X = combined[valid_cols].fillna(0.0).astype(float).values
    y = combined[target_col].fillna(0).astype(int).values
    w = combined["weight"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=0.5, max_iter=500, random_state=seed)
    model.fit(X_scaled, y, sample_weight=w)

    # Gini on accepted only vs. through-the-door
    X_acc = df_accepted[valid_cols].fillna(0.0).astype(float).values
    X_acc_sc = scaler.transform(X_acc)
    y_acc = df_accepted[target_col].fillna(0).astype(int).values

    if y_acc.sum() > 0:
        pd_acc = model.predict_proba(X_acc_sc)[:, 1]
        auc_acc = float(roc_auc_score(y_acc, pd_acc))
        gini_acc = 2 * auc_acc - 1
    else:
        gini_acc = 0.0

    pd_combined = model.predict_proba(X_scaled)[:, 1]
    if y.sum() > 0:
        auc_comb = float(roc_auc_score(y, pd_combined, sample_weight=w))
        gini_comb = 2 * auc_comb - 1
    else:
        gini_comb = 0.0

    gini_shift = gini_comb - gini_acc
    logger.info(
        "Reject inference: Gini(accepted)=%.4f | Gini(through-door)=%.4f | shift=%.4f",
        gini_acc, gini_comb, gini_shift,
    )
    return model, gini_shift


def get_col_case_insensitive(df: pd.DataFrame, aliases: list[str]) -> pd.Series | None:
    """Find a column case-insensitively, ignoring underscores, hyphens, and spaces."""
    def clean_name(name: str) -> str:
        return "".join(c.lower() for c in name if c.isalnum())

    cleaned_cols = {clean_name(c): c for c in df.columns}
    for alias in aliases:
        cleaned_alias = clean_name(alias)
        if cleaned_alias in cleaned_cols:
            actual_col = cleaned_cols[cleaned_alias]
            return df[actual_col]
    return None


def align_reject_data(
    df_rejected: pd.DataFrame,
    df_train: pd.DataFrame,
    woe_variables: list[str],
) -> pd.DataFrame:
    """Align reject columns case-insensitively and impute other variables using train means/modes."""
    df_rej_aligned = pd.DataFrame(index=df_rejected.index)

    def parse_numeric(series: pd.Series | None, default: float) -> pd.Series:
        if series is None:
            return pd.Series(default, index=df_rejected.index)
        s_str = series.astype(str).str.replace("%", "").str.strip()
        val = pd.to_numeric(s_str, errors="coerce")
        return val.fillna(default)

    # 1. fico_range_low & fico_range_high
    fico_series = get_col_case_insensitive(df_rejected, ["risk_score", "riskscore", "fico", "fico_range_low", "fico_score", "score"])
    if fico_series is None:
        fico_series = df_rejected.get("risk_score", pd.Series(0.0, index=df_rejected.index))
    df_rej_aligned["fico_range_low"] = parse_numeric(fico_series, default=600.0)
    df_rej_aligned["fico_range_high"] = df_rej_aligned["fico_range_low"] + 4

    # 2. dti
    dti_series = get_col_case_insensitive(df_rejected, ["debt_to_income_ratio", "debt_to_income", "debttoincome", "dti"])
    df_rej_aligned["dti"] = parse_numeric(dti_series, default=25.0)

    # 3. emp_length
    emp_len_series = get_col_case_insensitive(df_rejected, ["employment_length", "employmentlength", "emp_length", "emplength"])
    if emp_len_series is not None:
        df_rej_aligned["emp_length"] = emp_len_series.fillna("< 1 year")
    else:
        df_rej_aligned["emp_length"] = pd.Series("< 1 year", index=df_rejected.index)

    # 4. annual_inc
    annual_inc_series = get_col_case_insensitive(df_rejected, ["annual_inc", "annualinc", "annual_income", "annualincome"])
    df_rej_aligned["annual_inc"] = parse_numeric(annual_inc_series, default=45000.0)

    # 5. loan_amnt & funded_amnt
    loan_amnt_series = get_col_case_insensitive(df_rejected, ["loan_amnt", "loanamnt", "loan_amount", "amount_requested", "loan_amount_requested"])
    df_rej_aligned["loan_amnt"] = parse_numeric(loan_amnt_series, default=10000.0)
    df_rej_aligned["funded_amnt"] = df_rej_aligned["loan_amnt"]

    # 6. Impute other variables from woe_variables using df_train means/modes
    for col in woe_variables:
        if col not in df_rej_aligned.columns:
            if col in df_train.columns:
                col_series = df_train[col]
                if col_series.dtype == object or isinstance(col_series.dtype, pd.CategoricalDtype):
                    modes = col_series.mode()
                    mode_val = modes.iloc[0] if len(modes) > 0 else ""
                    df_rej_aligned[col] = mode_val
                else:
                    mean_val = pd.to_numeric(col_series, errors="coerce").mean()
                    df_rej_aligned[col] = mean_val if not np.isnan(mean_val) else 0.0
            else:
                df_rej_aligned[col] = 0.0

    return df_rej_aligned
