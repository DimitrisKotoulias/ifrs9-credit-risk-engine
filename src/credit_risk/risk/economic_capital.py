"""Monte Carlo economic capital via the ASRF (Vasicek) single-factor model.

Basel IRB delivers *regulatory* capital under an infinitely-granular, single-factor
assumption. *Economic* capital is read off the full simulated portfolio loss
distribution, which lets us report Value-at-Risk, Expected Shortfall (CVaR) and the
Unexpected Loss buffer that a bank would actually hold.

The systematic-factor convention matches the rest of the engine (see the Phase 9c
stress test in ``pipeline.py`` and ``risk/ifrs9_ecl.py``): a *negative* draw of the
systematic factor ``Z`` is an adverse state that raises every obligor's conditional PD

    p_i(Z) = Phi( (Phi^{-1}(PD_i) - sqrt(rho) * Z) / sqrt(1 - rho) ).

For a ~10^6-loan portfolio a naive ``(n_sim, n_loans)`` draw is infeasible, so obligors
are first aggregated into homogeneous PD buckets; the systematic factor is shared across
buckets each simulation and idiosyncratic risk enters through a per-bucket Binomial draw.
For a handful of loans the bucketing degenerates to one loan per bucket, i.e. exact
name-by-name simulation — which is what the unit tests exercise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

logger = logging.getLogger(__name__)

_PD_CLIP = 1e-9


def _aggregate_buckets(
    pd_arr: np.ndarray,
    lgd_arr: np.ndarray,
    ead_arr: np.ndarray,
    n_buckets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate obligors into EAD-weighted, PD-ranked homogeneous buckets.

    Returns
    -------
    (count, pd_bucket, lgd_bucket, mean_ead) each shape (B,), where ``B`` is the
    effective number of buckets (``<= n_buckets``; equals ``n`` for tiny portfolios).
    """
    n = len(pd_arr)
    n_buckets = int(max(1, min(n_buckets, n)))

    order = np.argsort(pd_arr, kind="stable")
    pd_s = pd_arr[order]
    lgd_s = lgd_arr[order]
    ead_s = ead_arr[order]

    # Contiguous PD-ranked groups of near-equal size.
    edges = np.linspace(0, n, n_buckets + 1).astype(int)

    counts: list[float] = []
    pd_b: list[float] = []
    lgd_b: list[float] = []
    ead_b: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi <= lo:
            continue
        w = ead_s[lo:hi]
        w_sum = float(w.sum())
        if w_sum <= 0.0:  # degenerate zero-EAD slice: fall back to equal weights
            w = np.ones(hi - lo)
            w_sum = float(w.sum())
        counts.append(float(hi - lo))
        pd_b.append(float(np.average(pd_s[lo:hi], weights=w)))
        lgd_b.append(float(np.average(lgd_s[lo:hi], weights=w)))
        ead_b.append(float(ead_s[lo:hi].mean()))

    return (
        np.asarray(counts, dtype=float),
        np.clip(np.asarray(pd_b, dtype=float), _PD_CLIP, 1.0 - _PD_CLIP),
        np.clip(np.asarray(lgd_b, dtype=float), 0.0, 1.0),
        np.asarray(ead_b, dtype=float),
    )


def simulate_portfolio_losses(
    pd_arr: np.ndarray,
    lgd_arr: np.ndarray,
    ead_arr: np.ndarray,
    *,
    rho: float = 0.15,
    n_sim: int = 50_000,
    seed: int = 42,
    n_buckets: int = 50,
) -> np.ndarray:
    """Simulate the portfolio loss distribution under the ASRF single-factor model.

    Parameters
    ----------
    pd_arr, lgd_arr, ead_arr:
        Per-obligor PD, LGD and EAD, each shape ``(n,)``.
    rho:
        Asset correlation to the systematic factor (Basel retail default ~0.15).
    n_sim:
        Number of Monte Carlo scenarios.
    seed:
        RNG seed for reproducibility.
    n_buckets:
        Number of PD-ranked buckets used to keep the simulation tractable.

    Returns
    -------
    np.ndarray shape ``(n_sim,)`` of simulated total portfolio losses.
    """
    pd_arr = np.clip(np.asarray(pd_arr, dtype=float), _PD_CLIP, 1.0 - _PD_CLIP)
    lgd_arr = np.clip(np.asarray(lgd_arr, dtype=float), 0.0, 1.0)
    ead_arr = np.asarray(ead_arr, dtype=float)
    if not (len(pd_arr) == len(lgd_arr) == len(ead_arr)):
        raise ValueError("pd_arr, lgd_arr and ead_arr must have equal length")
    if len(pd_arr) == 0:
        return np.zeros(n_sim, dtype=float)
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho}")

    count, pd_b, lgd_b, ead_b = _aggregate_buckets(pd_arr, lgd_arr, ead_arr, n_buckets)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sim)  # systematic factor, one draw per scenario

    g_pd = ndtri(pd_b)  # (B,)
    sqrt_rho = np.sqrt(rho)
    sqrt_1mrho = np.sqrt(1.0 - rho)

    # Conditional PD per (scenario, bucket): negative Z => higher PD (adverse).
    cond_pd = ndtr((g_pd[None, :] - sqrt_rho * z[:, None]) / sqrt_1mrho)  # (n_sim, B)

    counts_int = count.astype(np.int64)
    defaults = rng.binomial(counts_int[None, :], cond_pd)  # (n_sim, B)
    losses = (defaults * ead_b[None, :] * lgd_b[None, :]).sum(axis=1)  # (n_sim,)
    return losses


def risk_measures(losses: np.ndarray, alpha: float = 0.999) -> dict[str, float]:
    """Compute EL, VaR, Expected Shortfall, Unexpected Loss and Economic Capital.

    Parameters
    ----------
    losses:
        Simulated portfolio losses, shape ``(n_sim,)``.
    alpha:
        Confidence level for VaR / ES (e.g. 0.999).

    Returns
    -------
    dict with keys ``expected_loss, var, es, unexpected_loss, economic_capital, alpha``.
    ``unexpected_loss = VaR - EL`` (regulatory-style buffer); ``economic_capital = ES - EL``
    (the ES-based buffer). By construction ``ES >= VaR >= EL``.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        return {
            "expected_loss": 0.0, "var": 0.0, "es": 0.0,
            "unexpected_loss": 0.0, "economic_capital": 0.0, "alpha": float(alpha),
        }
    el = float(losses.mean())
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size > 0 else var
    return {
        "expected_loss": el,
        "var": var,
        "es": es,
        "unexpected_loss": var - el,
        "economic_capital": es - el,
        "alpha": float(alpha),
    }


def run_economic_capital(
    df: pd.DataFrame,
    *,
    pd_col: str = "pd_pred",
    lgd_col: str = "lgd_pred",
    ead_col: str = "ead",
    rho: float = 0.15,
    n_sim: int = 50_000,
    alpha: float = 0.999,
    seed: int = 42,
    n_buckets: int = 50,
) -> tuple[np.ndarray, dict[str, float]]:
    """Portfolio-level economic-capital driver used by the pipeline.

    Returns the simulated loss array (for plotting) and the risk-measure summary dict.
    """
    losses = simulate_portfolio_losses(
        df[pd_col].to_numpy(dtype=float),
        df[lgd_col].to_numpy(dtype=float),
        df[ead_col].to_numpy(dtype=float),
        rho=rho,
        n_sim=n_sim,
        seed=seed,
        n_buckets=n_buckets,
    )
    measures = risk_measures(losses, alpha=alpha)
    logger.info(
        "Economic capital: EL=%.0f | VaR(%.1f%%)=%.0f | ES=%.0f | EC=%.0f",
        measures["expected_loss"], alpha * 100.0,
        measures["var"], measures["es"], measures["economic_capital"],
    )
    return losses, measures
