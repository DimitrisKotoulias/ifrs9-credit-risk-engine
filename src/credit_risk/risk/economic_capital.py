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
    n_ead_strata: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate obligors into buckets homogeneous in **both** PD and EAD.

    Buckets were formed by PD rank alone, and each one then contributed
    ``defaults x mean_ead`` to the simulated loss. Within a PD bucket, though, exposure
    spans the whole book's range: a $1k loan and a $35k loan with the same PD were treated
    as interchangeable, so which obligors defaulted carried no exposure information at all.
    That suppresses precisely the co-movement that drives the tail --- the scenarios where
    the large exposures are the ones that default --- and the 99.9% VaR and ES the capital
    number rests on came out too low. The docstring also claimed EAD-weighting that only
    ever applied to the PD and LGD aggregates.

    Obligors are now PD-ranked into ``n_buckets / n_ead_strata`` groups and each group is
    then split by EAD quantile into ``n_ead_strata`` strata, so exposure dispersion is
    carried by the bucket structure rather than averaged away. Total bucket count is
    unchanged, so the simulation cost is unchanged.

    Returns
    -------
    (count, pd_bucket, lgd_bucket, mean_ead) each shape (B,), where ``B`` is the
    effective number of buckets (``<= n_buckets``; equals ``n`` for tiny portfolios).
    """
    n = len(pd_arr)
    n_buckets = int(max(1, min(n_buckets, n)))
    n_ead_strata = int(max(1, min(n_ead_strata, n_buckets)))
    n_pd_groups = int(max(1, n_buckets // n_ead_strata))

    order = np.argsort(pd_arr, kind="stable")
    pd_s = pd_arr[order]
    lgd_s = lgd_arr[order]
    ead_s = ead_arr[order]

    # Contiguous PD-ranked groups of near-equal size.
    edges = np.linspace(0, n, n_pd_groups + 1).astype(int)

    counts: list[float] = []
    pd_b: list[float] = []
    lgd_b: list[float] = []
    ead_b: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi <= lo:
            continue
        # Within the PD group, stratify by exposure.
        grp_pd, grp_lgd, grp_ead = pd_s[lo:hi], lgd_s[lo:hi], ead_s[lo:hi]
        ead_order = np.argsort(grp_ead, kind="stable")
        grp_pd, grp_lgd, grp_ead = (
            grp_pd[ead_order], grp_lgd[ead_order], grp_ead[ead_order]
        )
        sub_edges = np.linspace(0, len(grp_ead), n_ead_strata + 1).astype(int)
        for s_lo, s_hi in zip(sub_edges[:-1], sub_edges[1:], strict=True):
            if s_hi <= s_lo:
                continue
            w = grp_ead[s_lo:s_hi]
            if float(w.sum()) <= 0.0:  # degenerate zero-EAD slice: equal weights
                w = np.ones(s_hi - s_lo)
            counts.append(float(s_hi - s_lo))
            pd_b.append(float(np.average(grp_pd[s_lo:s_hi], weights=w)))
            lgd_b.append(float(np.average(grp_lgd[s_lo:s_hi], weights=w)))
            ead_b.append(float(grp_ead[s_lo:s_hi].mean()))

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
    rho: float | str = 0.15,
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
        Asset correlation to the systematic factor: a constant, or ``"supervisory"`` to
        use the BCBS "Other Retail" curve per PD bucket (see the note in the body).
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
    if not (isinstance(rho, str) or 0.0 <= rho < 1.0):
        raise ValueError(f"rho must be in [0, 1) or 'supervisory', got {rho}")

    count, pd_b, lgd_b, ead_b = _aggregate_buckets(pd_arr, lgd_arr, ead_arr, n_buckets)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sim)  # systematic factor, one draw per scenario

    g_pd = ndtri(pd_b)  # (B,)

    # Correlation. Passing "supervisory" uses the same BCBS "Other Retail" curve the IRB
    # capital calculation applies, evaluated per PD bucket. That matters because the
    # economic-capital and regulatory-capital figures are presented side by side: with a
    # flat rho=0.15 against a supervisory R that collapses to ~0.03 at this book's PDs,
    # the headline EC/RegCap ratio was driven mostly by a 5x correlation difference, not
    # by the tail fidelity the report attributed it to (FLAWS-N13).
    if isinstance(rho, str):
        if rho != "supervisory":
            raise ValueError(f"unknown rho mode {rho!r}; expected a float or 'supervisory'")
        from credit_risk.risk.basel_irb import irb_correlation  # noqa: PLC0415

        rho_b = np.asarray(irb_correlation(pd_b), dtype=float)  # (B,)
    else:
        rho_b = np.full_like(pd_b, float(rho))

    sqrt_rho = np.sqrt(rho_b)
    sqrt_1mrho = np.sqrt(1.0 - rho_b)

    # Conditional PD per (scenario, bucket): negative Z => higher PD (adverse).
    cond_pd = ndtr(
        (g_pd[None, :] - sqrt_rho[None, :] * z[:, None]) / sqrt_1mrho[None, :]
    )  # (n_sim, B)

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
    rho: float | str = 0.15,
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
