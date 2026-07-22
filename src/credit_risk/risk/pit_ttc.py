"""Point-in-Time vs Through-the-Cycle PD decomposition via Vasicek inversion.

Basel IRB uses a Through-the-Cycle (TTC) PD --- a macro-neutral long-run average --- while
IFRS 9 requires a forward-looking Point-in-Time (PiT) PD. Inverting the Vasicek single
factor model on the observed quarterly default-rate series recovers the long-run TTC PD
and the implied systematic factor Z for each quarter, making the cyclical PiT/TTC
distinction explicit and quantifiable.

Convention (matching the rest of the engine): Z < 0 marks an adverse quarter (realised
default rate above the TTC average); Z > 0 marks a benign quarter.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.special import ndtri

logger = logging.getLogger(__name__)

_DR_CLIP = 1e-6


def decompose_pit_ttc(
    observed_default_rates: np.ndarray,
    rho: float = 0.15,
) -> tuple[float, np.ndarray]:
    """Invert the Vasicek model to recover the TTC PD and per-period systematic factor Z.

    Given observed default rates DR_t, with
    ``DR_t = Phi((Phi^{-1}(PD_TTC) - sqrt(rho) Z_t) / sqrt(1 - rho))``,
    the long-run PD is the mean default rate and

        Z_t = (Phi^{-1}(PD_TTC) - Phi^{-1}(DR_t) sqrt(1 - rho)) / sqrt(rho).

    Returns ``(ttc_pd, z_factors)``.
    """
    dr = np.clip(np.asarray(observed_default_rates, dtype=float), _DR_CLIP, 1.0 - _DR_CLIP)
    if dr.size == 0:
        return 0.0, np.asarray([], dtype=float)
    if not 0.0 < rho < 1.0:
        raise ValueError(f"rho must be in (0, 1), got {rho}")
    ttc_pd = float(np.mean(dr))
    ttc_pd = min(max(ttc_pd, _DR_CLIP), 1.0 - _DR_CLIP)
    z = (ndtri(ttc_pd) - ndtri(dr) * np.sqrt(1.0 - rho)) / np.sqrt(rho)
    return ttc_pd, z


def run_pit_ttc(
    quarterly_df: pd.DataFrame,
    *,
    dr_col: str = "default_rate",
    quarter_col: str = "quarter",
    rho: float = 0.15,
) -> dict[str, object]:
    """Decompose a quarterly default-rate frame into TTC PD + PiT Z factors.

    Returns a dict with ``ttc_pd, rho, quarters, default_rates, z_factors``.
    """
    df = quarterly_df.dropna(subset=[dr_col]).copy()
    dr = df[dr_col].to_numpy(dtype=float)
    ttc_pd, z = decompose_pit_ttc(dr, rho=rho)
    quarters = (
        df[quarter_col].astype(str).tolist() if quarter_col in df.columns
        else [str(i) for i in range(len(dr))]
    )
    logger.info(
        "PiT/TTC decomposition: TTC PD=%.4f | Z in [%.3f, %.3f] over %d quarters",
        ttc_pd, float(z.min()) if z.size else 0.0, float(z.max()) if z.size else 0.0, len(dr),
    )
    return {
        "ttc_pd": ttc_pd,
        "rho": float(rho),
        "quarters": quarters,
        "default_rates": dr.tolist(),
        "z_factors": z.tolist(),
    }
