"""Single source of truth for published literature reference ranges.

Each :class:`Benchmark` ties a metric computed by the pipeline (a key into
``outputs/metrics.json``) to a published reference range carrying a citation and a
verifiable source locator. ``reports/render_latex.py`` renders BOTH the "Published
Benchmark" cell string AND the pass/fail verdict from the *same* object, so the range
can never drift between the LaTeX table and the comparison logic (the old design kept
them in two places and they disagreed). ``reports/qa_checks.py`` asserts every
benchmark row is registry-backed and driven by a live metric, structurally preventing
a re-introduction of a hand-typed/fabricated row.

Important framing: these are *reference ranges drawn from the literature*, not a
reproduction of the papers' experiments. We compare our own computed values against
published ranges; we do not re-run the cited papers' datasets. The report wording
reflects this ("comparison against published reference ranges", not "verification").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ── shared driver comments (surface when a value falls outside its band) ──────────
_LGD_NOTE = (
    "Unsecured P2P LGD spans mixed recovery regimes; cash-based realised LGD on this "
    "fully amortising book sits at the lower end of published bands"
)
_RWA_NOTE = (
    "Published 75--100\\% is the flat SA retail risk weight; the risk-sensitive IRB "
    "density for this book differs by construction --- documented, not an inconsistency"
)
_ECL_NOTE = (
    "The published band is the EBA \\emph{NPL} coverage ratio (allowance on "
    "\\emph{non-performing} exposure only); the project figure is whole-book lifetime "
    "ECL over total EAD, which blends performing Stage~1 exposure and therefore sits "
    "below impaired-only coverage --- not a like-for-like gap"
)
_LGD_R2_NOTE = (
    "LGD is strongly bimodal (mass at 0 and near 1), so $R^2$ is an unforgiving metric "
    "and is routinely low or negative out-of-time; \\textcite{loterman2012benchmarking} "
    "themselves report a wide $0.04$--$0.43$ spread. The negative value reflects a "
    "2016--2018 severity regime shift versus the fitting vintages"
)
_STAGE2_LOW_NOTE = (
    "Resolved-outcome book: most deteriorated loans have already defaulted into Stage~3 "
    "rather than lingering in Stage~2"
)
_STAGE2_HIGH_NOTE = (
    "Stage~2 share well above the EBA EU-aggregate ($\\approx$9--10\\%): expected for an "
    "unsecured, high-yield US consumer book with materially higher SICR incidence than the "
    "diversified EU bank averages"
)


@dataclass(frozen=True)
class Benchmark:
    """A published reference range for one pipeline metric."""

    key: str            # token stem; report uses __VERDICT_<KEY>__ / __COMMENT_<KEY>__
    label: str          # human label (also used in QA diagnostics)
    metric_key: str     # key into the render-time values dict / outputs/metrics.json
    low: float
    high: float
    source_bibkey: str  # biblatex key of the citing work
    source_locator: str # verifiable pointer (journal/vol/report), not an invented page
    unit: str = "ratio"          # "ratio" -> $l - h$ ; "pct" -> l\% -- h\%
    abs_value: bool = False      # compare abs(value) (e.g. reject-inference dGini)
    below_comment: str = ""
    above_comment: str = ""

    def range_tex(self) -> str:
        """LaTeX string for the 'Published Benchmark' cell."""
        if self.unit == "pct":
            return f"{self.low * 100:.0f}\\% -- {self.high * 100:.0f}\\%"
        return f"${self.low:.2f} - {self.high:.2f}$"

    def verdict(self, value: object) -> tuple[str, str]:
        """Numerically compare *value* against [low, high]; return (verdict, comment)."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "N/A", "Value unavailable at build time"
        if math.isnan(v):
            return "N/A", "Value unavailable at build time"
        if self.abs_value:
            v = abs(v)
        if v < self.low:
            return "Below typical range", self.below_comment
        if v > self.high:
            return "Above typical range", self.above_comment
        return "Within range", ""


# ── the registry ─────────────────────────────────────────────────────────────────
# Reconciliations applied vs the old hand-typed tables:
#   * ONE Lessmann Gini band (0.30-0.45), internally consistent with the AUC band
#     0.65-0.73 since Gini = 2*AUC - 1. The old tables disagreed (0.35-0.50 vs 0.30-0.45).
#   * AUC band 0.65-0.73 is now sourced here (partner of the Gini band), not smuggled in
#     via an apologetic in-code retrofit comment.
#   * Downturn LGD compared against the same unsecured-retail band (0.25-0.45,
#     Bellotti & Crook 2012); the old 0.35-0.55 was not stated by that paper. The 90th-pct
#     downturn uplift keeps the value inside the unsecured band.
#   * LGD R^2 is now a LIVE metric (was a fabricated static "0.09-0.15/Within range").

_BENCHMARKS: list[Benchmark] = [
    Benchmark(
        key="AUC_OOT", label="PD AUC (OOT)", metric_key="auc_oot",
        low=0.65, high=0.73,
        source_bibkey="lessmann2015benchmarking",
        source_locator="consumer-credit AUC benchmarks, EJOR 247(1):124--136",
        below_comment=("Below the consumer-credit band; challenger models plateau at the "
                       "same level, indicating a dataset discrimination ceiling"),
    ),
    Benchmark(
        key="GINI_OOT", label="PD Gini (OOT)", metric_key="gini_oot",
        low=0.30, high=0.45,
        source_bibkey="lessmann2015benchmarking",
        source_locator="Gini = 2*AUC-1 on the consumer-credit AUC band, EJOR 247(1)",
    ),
    Benchmark(
        key="MEAN_LGD", label="LGD Mean", metric_key="mean_lgd",
        low=0.25, high=0.45,
        source_bibkey="bellotti2012",
        source_locator="unsecured retail LGD, IJF 28(1):171--182",
        below_comment=_LGD_NOTE,
        above_comment=(
            "The OOS-validated LightGBM severity model puts mean LGD above both cited "
            "bands, consistent with low post-charge-off recovery on unsecured instalment "
            "loans once EAD is principal-only; the rejected two-stage model materially "
            # R^2 driven by the live metric (lgd_model_comparison.champion.r2) via the
            # __LGD_R2_TWOSTAGE__ token render_latex substitutes, so it can never drift
            # from the value quoted in the §7.7 body again.
            "under-predicted severity (OOS $R^2 \\approx __LGD_R2_TWOSTAGE__$)"
        ),
    ),
    Benchmark(
        key="LGD_R2", label="LGD $R^2$ (OOS)", metric_key="lgd_r2",
        low=0.04, high=0.43,
        source_bibkey="loterman2012benchmarking",
        source_locator="LGD regression $R^2$ across models, IJF 28(1):161--170",
        below_comment=_LGD_R2_NOTE,
    ),
    Benchmark(
        key="RWA_DENSITY", label="RWA Density", metric_key="rwa_density",
        low=0.75, high=1.00, unit="pct",
        source_bibkey="bcbs2017",
        source_locator="SA flat retail risk weight (75\\%), Basel III finalisation",
        below_comment=_RWA_NOTE,
        above_comment=(
            "Risk-sensitive IRB density exceeds the flat SA weight: the OOS-validated LGD "
            "(mean $\\approx 0.89$, above published unsecured LGD ranges --- driver "
            "discussed in the LGD Mean row above) and high empirical default rate on this "
            "unsecured, high-yield book drive a capital \\emph{surcharge} --- the "
            "economically expected outcome, not an inconsistency"
        ),
    ),
    Benchmark(
        key="GINI_SHIFT", label="Reject Inference $\\Delta$Gini", metric_key="gini_shift",
        low=0.0, high=0.10,
        source_bibkey="crook2004reject",
        source_locator="reject-inference Gini impact, JBF 28(4):857--874",
        below_comment=(
            "A negative through-the-door $\\Delta$Gini means parcelling reject inference "
            "slightly \\emph{reduced} discrimination rather than improving it --- itself the "
            "central finding of \\textcite{crook2004reject}, who show reject inference often "
            "fails to improve application-scorecard performance"
        ),
    ),
    Benchmark(
        key="PSI", label="Score PSI", metric_key="psi_total",
        low=0.0, high=0.10,
        source_bibkey="siddiqi2017",
        source_locator="population-stability threshold ($<0.10$), Intelligent Credit Scoring",
    ),
    # ── Table 18-only rows ────────────────────────────────────────────────────────
    Benchmark(
        key="LIT_LGBM", label="LightGBM Gini (OOT)", metric_key="lgbm_gini_oot",
        low=0.35, high=0.55,
        source_bibkey="lessmann2015benchmarking",
        source_locator="tree-ensemble Gini on consumer-credit data, EJOR 247(1)",
    ),
    Benchmark(
        key="LIT_DLGD", label="Downturn LGD", metric_key="downturn_lgd",
        low=0.25, high=0.45,
        source_bibkey="bellotti2012",
        source_locator="unsecured retail LGD band; 90th-pct downturn uplift, IJF 28(1)",
        below_comment=_LGD_NOTE,
        above_comment="Conservative 90th-percentile downturn uplift above the mean band",
    ),
    Benchmark(
        key="LIT_ECL_COV", label="IFRS 9 ECL Coverage", metric_key="ecl_coverage",
        low=0.40, high=0.45, unit="pct",
        source_bibkey="eba2022",
        source_locator="NPL coverage ratio 43.4\\%, EBA Risk Dashboard Q4 2022",
        below_comment=_ECL_NOTE,
        above_comment="Coverage above published range; driven by portfolio mix",
    ),
    Benchmark(
        key="LIT_STAGE2", label="IFRS 9 Stage 2 \\%", metric_key="stage2_pct",
        low=0.08, high=0.11, unit="pct",
        source_bibkey="eba2022",
        source_locator="Stage 2 share (EU aggregate 9.4\\%), EBA Risk Dashboard Q4 2022",
        below_comment=_STAGE2_LOW_NOTE,
        above_comment=_STAGE2_HIGH_NOTE,
    ),
]

# key -> Benchmark, plus the ordering each report table renders.
BENCHMARKS: dict[str, Benchmark] = {b.key: b for b in _BENCHMARKS}

# Table 13 ("Comparison against published reference ranges"). IRB-vs-SA is a *direction*
# check (not a numeric range) and is handled specially in render_latex.py.
TABLE13_KEYS: list[str] = [
    "AUC_OOT", "GINI_OOT", "MEAN_LGD", "LGD_R2", "RWA_DENSITY", "GINI_SHIFT", "PSI",
]

# Table 18 (results vs published literature). Reuses GINI_OOT / MEAN_LGD bands.
TABLE18_KEYS: list[str] = [
    "GINI_OOT", "LIT_LGBM", "MEAN_LGD", "LIT_DLGD", "LIT_ECL_COV", "LIT_STAGE2",
]

# Every metric_key a registry-backed row depends on (used by qa_checks to confirm the
# value was actually available at build time, i.e. not silently defaulted).
REQUIRED_METRIC_KEYS: frozenset[str] = frozenset(
    BENCHMARKS[k].metric_key for k in set(TABLE13_KEYS) | set(TABLE18_KEYS)
)
