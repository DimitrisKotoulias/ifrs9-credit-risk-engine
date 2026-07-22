"""Basel IRB retail regulatory capital formula.

Implements the 'other retail' (not QRRE, not mortgages) supervisory formula
from BCBS §322–328 (Basel II/III Framework).

Applicable to Lending Club personal loans (fully drawn instalment, not revolving).

Formulas (Appendix B):
    R   = 0.03 · (1 − e^(−35·PD)) / (1 − e^(−35))
         + 0.16 · [1 − (1 − e^(−35·PD)) / (1 − e^(−35))]

    K   = LGD · N[(1−R)^(−0.5) · G(PD) + (R/(1−R))^0.5 · G(0.999)] − PD · LGD

    RWA = K · 12.5 · EAD

Where:
    N  = standard normal CDF
    G  = standard normal inverse CDF (quantile function)
    PD_floor = 0.03% = 0.0003

Notes:
    - Retail exposures have NO maturity adjustment (unlike corporate).
    - Use downturn LGD (conservative) for the capital calculation.
    - PD floor = max(PD, 0.0003).
    - Correlation R is bounded in [0.03, 0.16].
    - QRRE (qualifying revolving retail): fixed R=0.04 — NOT used here.
    - Residential mortgages: fixed R=0.15 — NOT used here.

Standardised approach reference:
    Under the Standardised Approach, retail exposures receive a flat 75% risk weight.
    Capital under SA = 8% × 75% × EAD = 6% × EAD.
    IRB gives risk-sensitive capital reflecting actual PD/LGD.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

_PD_FLOOR = 0.0003  # 0.03% regulatory minimum
_N = norm.cdf     # standard normal CDF
_G = norm.ppf     # standard normal quantile function


def irb_correlation(pd_arr: np.ndarray) -> np.ndarray:
    """Supervisory asset correlation R for 'Other Retail'.
    Formula: BCBS (2006) CRE31.15.
        w = (1 - exp(-35*PD)) / (1 - exp(-35))
        R = 0.03*w + 0.16*(1-w), R in [0.03, 0.16].
    """
    e_neg35 = np.exp(-35.0)
    denom = 1.0 - e_neg35
    weight = (1.0 - np.exp(-35.0 * pd_arr)) / denom

    r = 0.03 * weight + 0.16 * (1.0 - weight)
    return np.clip(r, 0.03, 0.16)


def irb_capital_requirement(
    pd_arr: np.ndarray,
    lgd_arr: np.ndarray,
    pd_floor: float = _PD_FLOOR,
) -> np.ndarray:
    """Compute capital requirement K (retail IRB, no maturity adjustment).

    K = LGD · N[(1−R)^(−0.5) · G(PD) + (R/(1−R))^0.5 · G(0.999)] − PD · LGD

    Parameters
    ----------
    pd_arr:
        Probability of Default (pre-floor; will be floored internally).
    lgd_arr:
        Loss Given Default (use downturn LGD for Basel).
    pd_floor:
        Regulatory PD floor (default 0.0003).

    Returns
    -------
    np.ndarray
        Capital requirement K per exposure (fraction of EAD).
    """
    pd_floored = np.maximum(np.asarray(pd_arr, dtype=float), pd_floor)
    lgd = np.asarray(lgd_arr, dtype=float)

    r = irb_correlation(pd_floored)

    # Vasicek asymptotic single-factor formula
    g_pd = _G(pd_floored)  # G(PD)
    g_999 = _G(0.999)       # G(0.999) ≈ 3.0902

    inner = (1.0 - r) ** (-0.5) * g_pd + (r / (1.0 - r)) ** 0.5 * g_999
    k = lgd * _N(inner) - pd_floored * lgd

    return np.maximum(k, 0.0)


def irb_rwa(
    pd_arr: np.ndarray,
    lgd_arr: np.ndarray,
    ead_arr: np.ndarray,
    pd_floor: float = _PD_FLOOR,
) -> np.ndarray:
    """Compute Risk-Weighted Assets.

    RWA = K · 12.5 · EAD

    Parameters
    ----------
    pd_arr:
        PD per exposure.
    lgd_arr:
        Downturn LGD per exposure (or scalar).
    ead_arr:
        EAD per exposure.
    pd_floor:
        Regulatory PD floor.

    Returns
    -------
    np.ndarray
        RWA per exposure.
    """
    k = irb_capital_requirement(pd_arr, lgd_arr, pd_floor)
    ead = np.asarray(ead_arr, dtype=float)
    return k * 12.5 * ead


def run_basel_irb(
    df: pd.DataFrame,
    pd_col: str = "pd_pred",
    lgd_downturn: float | None = None,
    lgd_col: str | None = None,
    ead_col: str = "ead",
    pd_floor: float = _PD_FLOOR,
    capital_ratio: float = 0.08,
) -> pd.DataFrame:
    """Compute Basel IRB RWA and capital for a portfolio.

    Parameters
    ----------
    df:
        Portfolio DataFrame.
    pd_col:
        PD column name.
    lgd_downturn:
        Scalar downturn LGD applied to all exposures. Mutually exclusive with lgd_col.
    lgd_col:
        Per-loan LGD column. Mutually exclusive with lgd_downturn.
    ead_col:
        EAD column.
    pd_floor:
        Regulatory PD floor.
    capital_ratio:
        Minimum capital ratio (8%).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: rwa, capital_requirement_k, capital_charge.
    """
    if lgd_downturn is None and lgd_col is None:
        raise ValueError("Provide either lgd_downturn (scalar) or lgd_col (column name).")

    out = df.copy()
    pd_arr = out[pd_col].values
    ead_arr = out[ead_col].values

    if lgd_downturn is not None:
        lgd_arr = np.full(len(df), lgd_downturn)
    else:
        lgd_arr = out[lgd_col].values  # type: ignore[index]

    out["capital_requirement_k"] = irb_capital_requirement(pd_arr, lgd_arr, pd_floor)
    out["rwa"] = irb_rwa(pd_arr, lgd_arr, ead_arr, pd_floor)
    out["capital_charge"] = out["rwa"] * capital_ratio

    # Standardised approach comparison
    out["rwa_standardised"] = ead_arr * 0.75  # 75% risk weight for retail

    total_rwa = float(out["rwa"].sum())
    total_ead = float(ead_arr.sum())
    rwa_density = total_rwa / total_ead if total_ead > 0 else 0.0
    total_capital = float(out["capital_charge"].sum())
    total_rwa_sa = float(out["rwa_standardised"].sum())

    logger.info(
        "Basel IRB: total RWA=%.2f | capital=%.2f | RWA density=%.1f%% | "
        "SA reference RWA=%.2f",
        total_rwa, total_capital, rwa_density * 100, total_rwa_sa,
    )

    out.attrs["basel_summary"] = {
        "total_rwa_irb": total_rwa,
        "total_rwa_sa": total_rwa_sa,
        "rwa_density": rwa_density,
        "total_capital": total_capital,
        "pd_floor": pd_floor,
        "capital_ratio": capital_ratio,
    }

    return out
