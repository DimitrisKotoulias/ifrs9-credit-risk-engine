"""Feature selection for PD scorecard.

Implements:
- IV-band filter (0.02 ≤ IV ≤ 0.50)
- VIF (Variance Inflation Factor) check for multicollinearity
- Sign-check after logistic regression (positive coefficients on WoE features)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def filter_by_iv(
    iv_table: pd.DataFrame,
    min_iv: float = 0.02,
    max_iv: float = 0.50,
) -> list[str]:
    """Select features by Information Value band.

    Parameters
    ----------
    iv_table:
        DataFrame with columns ['variable', 'iv'].
    min_iv:
        Minimum IV (below → near-zero predictive power).
    max_iv:
        Maximum IV (above → suspiciously high, possible leakage).

    Returns
    -------
    list[str]
        Variable names passing the IV filter.
    """
    mask = (iv_table["iv"] >= min_iv) & (iv_table["iv"] <= max_iv)
    selected = iv_table.loc[mask, "variable"].tolist()

    rejected_low = iv_table.loc[iv_table["iv"] < min_iv, "variable"].tolist()
    rejected_high = iv_table.loc[iv_table["iv"] > max_iv, "variable"].tolist()

    logger.info(
        "IV filter: %d selected [%.2f, %.2f] | %d dropped (low IV): %s | %d dropped (high IV): %s",
        len(selected), min_iv, max_iv,
        len(rejected_low), rejected_low,
        len(rejected_high), rejected_high,
    )
    return selected


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Compute Variance Inflation Factor for each column.

    Uses OLS R² from regressing each feature on all others.
    VIF = 1 / (1 - R²).

    Parameters
    ----------
    X:
        DataFrame of numeric features (WoE-transformed, no NaN).

    Returns
    -------
    pd.Series
        VIF per column, indexed by column name.
    """
    from sklearn.linear_model import LinearRegression  # noqa: PLC0415

    X_arr = X.values.astype(float)
    vifs = {}

    for i, col in enumerate(X.columns):
        others = np.delete(X_arr, i, axis=1)
        y_col = X_arr[:, i]
        reg = LinearRegression(fit_intercept=True)
        reg.fit(others, y_col)
        r2 = float(reg.score(others, y_col))
        vifs[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf

    return pd.Series(vifs)


def filter_by_vif(
    X_woe: pd.DataFrame,
    max_vif: float = 5.0,
    y: pd.Series | None = None,
) -> list[str]:
    """Iteratively drop the highest-VIF feature, prioritizing high-IV features if y is provided.

    Parameters
    ----------
    X_woe:
        WoE-transformed feature matrix.
    max_vif:
        Maximum acceptable VIF.
    y:
        Binary target series to calculate IVs for feature prioritization.

    Returns
    -------
    list[str]
        Remaining feature names after VIF filtering.
    """
    remaining = list(X_woe.columns)

    # Pre-calculate IV if y is provided
    iv_dict = {}
    if y is not None:
        y_arr = y.values.astype(int)
        n_bad_tot = y_arr.sum()
        n_good_tot = len(y_arr) - n_bad_tot

        for col in remaining:
            df_col = pd.DataFrame({"x": X_woe[col].values, "y": y_arr})
            grp = df_col.groupby("x")["y"].agg(["count", "sum"])
            col_iv = 0.0
            for _, row in grp.iterrows():
                b = row["sum"]
                g = row["count"] - b
                pct_b = (b + 0.5) / (n_bad_tot + 1.0)
                pct_g = (g + 0.5) / (n_good_tot + 1.0)
                woe_val = np.log(pct_g / pct_b)
                col_iv += (pct_g - pct_b) * woe_val
            iv_dict[col] = max(col_iv, 0.001)

    while True:
        if len(remaining) <= 1:
            break
        vifs = compute_vif(X_woe[remaining])
        max_current = vifs.max()
        if max_current <= max_vif:
            break

        if y is not None:
            # Filter among those violating VIF limit
            high_vif_cols = vifs[vifs > max_vif].index.tolist()
            # Drop the one with the highest VIF / IV ratio
            ratios = {col: vifs[col] / iv_dict[col] for col in high_vif_cols}
            worst = max(ratios, key=ratios.get)
            worst_vif = vifs[worst]
        else:
            worst = str(vifs.idxmax())
            worst_vif = max_current

        logger.info("VIF filter dropping '%s' (VIF=%.1f > %.1f)", worst, worst_vif, max_vif)
        remaining.remove(worst)

    logger.info("VIF filter: %d features remaining.", len(remaining))
    return remaining


def sign_check(
    coefs: pd.Series,
    expected_positive: bool = True,
) -> list[str]:
    """Return features whose coefficient sign violates the expected direction.

    In a WoE scorecard, all coefficients should be positive (higher WoE → lower PD,
    but the logistic model is fit on positive coefficients since WoE already encodes
    direction).

    Parameters
    ----------
    coefs:
        Series of logistic regression coefficients, indexed by feature name.
    expected_positive:
        If True, flag features with negative coefficients.

    Returns
    -------
    list[str]
        Feature names with wrong sign.
    """
    if expected_positive:
        violations = coefs[coefs < 0].index.tolist()
    else:
        violations = coefs[coefs > 0].index.tolist()

    if violations:
        logger.warning(
            "Sign check: %d features have wrong coefficient sign: %s",
            len(violations), violations,
        )
    else:
        logger.info("Sign check: all coefficients have correct sign.")

    return violations
