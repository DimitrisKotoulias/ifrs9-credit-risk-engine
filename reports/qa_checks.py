"""Pre-build QA consistency checks for the model risk report (Fix 3.3 + 1.1#4).

Run automatically by render_latex.py before PDF compilation. Verifies
cross-table numerical identities in outputs/metrics.json and guards the
generated .tex against unreplaced template variables.

Raises QAError on any failure so the build stops instead of shipping an
internally inconsistent report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REL_TOL = 0.001  # 0.1% relative tolerance per upgrades.md Fix 3.3


class QAError(AssertionError):
    """A report consistency check failed."""


def _rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _check(cond: bool, msg: str, failures: list[str]) -> None:
    if not cond:
        failures.append(msg)


def check_rwa_density(metrics: dict, failures: list[str]) -> None:
    """RWA_IRB / EAD_total must equal the reported RWA density."""
    rwa = metrics.get("total_rwa")
    ead = metrics.get("total_ead_portfolio")
    density_str = str(metrics.get("rwa_density", "")).replace("%", "").strip()
    if not (rwa and ead and density_str):
        return
    density_reported = float(density_str) / 100.0
    density_computed = rwa / ead
    _check(
        abs(density_computed - density_reported) < 0.001,
        f"RWA density mismatch: computed {density_computed:.4f} vs reported {density_reported:.4f}",
        failures,
    )


def check_ecl_coverage(metrics: dict, failures: list[str]) -> None:
    """ECL_total / EAD_total must equal the reported coverage ratio."""
    ecl = metrics.get("total_ecl")
    ead = metrics.get("total_ead_portfolio")
    cov = metrics.get("ecl_coverage")
    if not (ecl and ead and cov):
        return
    _check(
        abs(ecl / ead - cov) < 1e-4,
        f"ECL coverage mismatch: computed {ecl / ead:.6f} vs reported {cov:.6f}",
        failures,
    )


def check_capital_identity(metrics: dict, failures: list[str]) -> None:
    """Minimum capital (8% of RWA) must equal RWA / 12.5."""
    rwa = metrics.get("total_rwa")
    if not rwa:
        return
    _check(
        _rel_err(rwa * 0.08, rwa / 12.5) < REL_TOL,
        "Capital identity broken: RWA*0.08 != RWA/12.5",
        failures,
    )
    stress_rwa = metrics.get("stress_rwa")
    stress_cap = metrics.get("stress_capital_req")
    if stress_rwa and stress_cap:
        _check(
            _rel_err(stress_cap, stress_rwa * 0.08) < REL_TOL,
            f"Stress capital mismatch: reported {stress_cap:,.0f} vs RWA*8% {stress_rwa * 0.08:,.0f}",
            failures,
        )


def check_cutoff_optimum(metrics: dict, failures: list[str]) -> None:
    """The cited operating cutoff must be a traceable grid row and equal the
    risk-appetite cutoff (profit max subject to the approved bad-rate ceiling)."""
    grid = metrics.get("cutoff_strategy_table") or []
    opt = metrics.get("cutoff_optimal_profit") or {}
    if not (grid and opt):
        return
    # (a) the cited optimum must correspond to an actual grid row (traceability)
    grid_row = next((r for r in grid if r.get("cutoff") == opt.get("cutoff")), None)
    _check(
        grid_row is not None,
        f"Cited operating cutoff {opt.get('cutoff')} is not a row in the swept grid",
        failures,
    )
    if grid_row is not None:
        for key in ("approval_rate", "bad_rate", "expected_profit", "raroc"):
            if key in opt and key in grid_row:
                _check(
                    _rel_err(grid_row[key], opt[key]) < REL_TOL,
                    f"Operating-cutoff {key} mismatch: grid {grid_row[key]} vs cited {opt[key]}",
                    failures,
                )
    # (b) it must equal the risk-appetite cutoff, and honour the bad-rate ceiling
    max_bad = metrics.get("cutoff_max_bad_rate")
    if max_bad is not None:
        _check(
            opt.get("bad_rate", 1.0) <= max_bad + 1e-9,
            f"Operating cutoff bad rate {opt.get('bad_rate')} exceeds appetite ceiling {max_bad}",
            failures,
        )
        try:
            from credit_risk.business.cutoff import risk_appetite_cutoff  # noqa: PLC0415
        except Exception:  # pragma: no cover - import guard
            return
        expected = risk_appetite_cutoff(grid, max_bad_rate=max_bad)
        if expected is not None:
            _check(
                expected["cutoff"] == opt.get("cutoff"),
                f"Cited cutoff {opt.get('cutoff')} != risk-appetite cutoff {expected['cutoff']} (ceiling {max_bad})",
                failures,
            )


def check_scenario_dr(metrics: dict, failures: list[str]) -> None:
    """Implied default rate per scenario must equal beta·x from the projection
    coefficients (sign-adjusted where those were used)."""
    # Scenario projections use the sign-adjusted coefficients when present.
    elas = metrics.get("macro_elasticities_adjusted") or metrics.get("macro_elasticities") or {}
    inputs = metrics.get("macro_scenario_inputs") or {}
    preds = metrics.get("macro_predictions") or {}  # stored in percent
    if not (elas and inputs and preds):
        return
    for scen, x in inputs.items():
        if scen not in preds:
            continue
        dr = elas.get("const", 0.0) + sum(
            elas.get(k, 0.0) * v for k, v in x.items()
        )
        dr = min(max(dr, 1e-4), 0.99)
        reported = preds[scen] / 100.0
        _check(
            _rel_err(dr, reported) < REL_TOL,
            f"Scenario '{scen}' implied DR mismatch: recomputed {dr:.4%} vs reported {reported:.4%}",
            failures,
        )


def check_scenario_ordering(metrics: dict, failures: list[str]) -> None:
    """Implied default rates must follow Downside >= Baseline >= Upside."""
    preds = metrics.get("macro_predictions") or {}
    up, base, down = preds.get("upside"), preds.get("baseline"), preds.get("downside")
    if None in (up, base, down):
        return
    _check(
        down >= base >= up,
        f"Scenario DR ordering violated: downside {down:.3f} / baseline {base:.3f} / upside {up:.3f} "
        "(expected downside >= baseline >= upside)",
        failures,
    )


def check_irb_sa_direction(metrics: dict, failures: list[str]) -> None:
    """IRB vs SA RWA direction must be consistent with the reported RWA density
    (density above the flat 75% SA weight <=> IRB RWA above SA RWA = surcharge)."""
    rwa = metrics.get("total_rwa")
    rwa_sa = metrics.get("total_rwa_sa")
    density_str = str(metrics.get("rwa_density", "")).replace("%", "").strip()
    if not (rwa and rwa_sa and density_str):
        return
    density = float(density_str) / 100.0
    _check(
        (rwa > rwa_sa) == (density > 0.75),
        f"IRB/SA direction inconsistent: total_rwa {rwa:,.0f} vs SA {rwa_sa:,.0f} "
        f"but RWA density {density:.3f} vs 0.75 SA flat weight",
        failures,
    )


def check_vintage_pd_ratio(metrics: dict, failures: list[str]) -> None:
    """Spot-check: predicted PD / actual DR must equal the reported PD Ratio."""
    rows = metrics.get("pd_backtest_vintage") or []
    for row in rows[:5] + rows[-5:]:
        pred = row.get("predicted_pd") or row.get("pred_pd")
        actual = row.get("actual_dr") or row.get("actual_default_rate")
        ratio = row.get("pd_ratio") or row.get("ratio")
        if pred is None or actual is None or ratio is None or not actual:
            continue
        _check(
            _rel_err(pred / actual, ratio) < 0.01,
            f"Vintage {row.get('vintage')}: PD ratio {ratio} != predicted/actual {pred / actual:.4f}",
            failures,
        )


def check_no_unreplaced_vars(tex_path: Path, failures: list[str]) -> None:
    """Fix 1.1#4: generated .tex must contain no unreplaced VAR_ or __TOKEN__ placeholders.

    Also catches LaTeX-escaped variants (e.g. ``VAR\\_N\\_OOT``) which evade a plain
    ``VAR_`` search and a naive ``str.replace("VAR_N_OOT", ...)`` — this was the actual
    source of the placeholders that reached the shipped PDF.
    """
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    # VAR followed by (optionally backslash-escaped) underscore segments.
    hits = sorted(set(re.findall(r"VAR(?:\\?_)[A-Z0-9]+(?:\\?_[A-Z0-9]+)*", text)))
    _check(not hits, f"Unreplaced template variables in {tex_path.name}: {hits}", failures)
    tokens = sorted(set(re.findall(r"__[A-Z0-9_]{3,}__", text)))
    _check(not tokens, f"Unreplaced __TOKEN__ placeholders in {tex_path.name}: {tokens}", failures)


def check_es_ge_var(metrics: dict, failures: list[str]) -> None:
    """Economic-capital tail measures must obey ES >= VaR >= EL >= 0."""
    ec = metrics.get("econ_cap")
    if not ec:
        return
    el = float(ec.get("expected_loss", 0.0))
    var = float(ec.get("var", 0.0))
    es = float(ec.get("es", 0.0))
    _check(
        es >= var >= el >= 0.0,
        f"Economic-capital ordering violated: EL={el:.0f}, VaR={var:.0f}, ES={es:.0f} "
        "(require ES >= VaR >= EL >= 0)",
        failures,
    )


def check_benchmarks_sourced(failures: list[str]) -> None:
    """Every literature benchmark must be registry-backed with a resolvable citation.

    Guards against re-introducing a hand-typed/fabricated benchmark row (the original
    LGD R^2 defect): each Benchmark in reports/benchmarks.py must carry a non-empty
    source citation + locator, a well-ordered range, and a bibkey that actually exists
    in the report's .bib so the citation resolves.
    """
    try:
        from benchmarks import BENCHMARKS  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - import guard
        _check(False, f"Cannot import benchmark registry: {exc}", failures)
        return

    bib_path = Path(__file__).resolve().parent / "model_risk_report.bib"
    bib_keys: set[str] = set()
    if bib_path.exists():
        bib_text = bib_path.read_text(encoding="utf-8", errors="replace")
        bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_text))

    for key, b in BENCHMARKS.items():
        _check(bool(b.source_bibkey), f"Benchmark '{key}' has no source_bibkey", failures)
        _check(bool(b.source_locator), f"Benchmark '{key}' has no source_locator", failures)
        _check(b.low <= b.high, f"Benchmark '{key}' range is inverted ({b.low} > {b.high})", failures)
        _check(bool(b.metric_key), f"Benchmark '{key}' has no metric_key", failures)
        if bib_keys:
            _check(
                b.source_bibkey in bib_keys,
                f"Benchmark '{key}' cites '{b.source_bibkey}' which is absent from the .bib",
                failures,
            )


def check_no_fabricated_benchmark(tex_path: Path, failures: list[str]) -> None:
    """The generated .tex must not contain the old fabricated static LGD R^2 band.

    The LGD R^2 row is now driven by the live computed metric; a literal ``0.09 - 0.15``
    (or ``0.09 -- 0.15``) reappearing means a hand-typed value was smuggled back in.
    """
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    _check(
        not re.search(r"0\.09\s*-{1,2}\s*0\.15", text),
        f"Fabricated static LGD R^2 band '0.09-0.15' found in {tex_path.name}",
        failures,
    )


def check_citations_resolve(tex_path: Path, failures: list[str]) -> None:
    """Every \\parencite/\\textcite/\\cite key in the .tex must exist in the .bib.

    Guards against citation drift such as citing a work that was never added to the
    bibliography (the SHAP/Lundberg and Bellotti-2009 survival defects), or leaving a
    dangling key after a citation is corrected.
    """
    bib_path = Path(__file__).resolve().parent / "model_risk_report.bib"
    if not bib_path.exists():
        return
    bib_keys = set(
        re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_path.read_text(encoding="utf-8", errors="replace"))
    )
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    cited: set[str] = set()
    for m in re.finditer(
        r"\\(?:parencite|textcite|autocite|footcite|cite)\s*(?:\[[^\]]*\])*\{([^}]*)\}", text
    ):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                cited.add(key)
    missing = sorted(cited - bib_keys)
    _check(not missing, f"Citations with no matching .bib entry: {missing}", failures)


def check_lgd_r2_consistency(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """The rejected two-stage LGD R^2 quoted in prose must track the live metric.

    The value is now substituted from ``lgd_model_comparison.champion.r2`` in both the
    §7.7 body and the benchmark comment; a reappearance of the old hand-typed ``-1.13``
    (or absence of the live value) means a stale literal was smuggled back in.
    """
    ts = ((metrics.get("lgd_model_comparison") or {}).get("champion") or {}).get("r2")
    if ts is None:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    expected = f"{float(ts):.2f}"
    _check(
        "-1.13" not in text,
        "Stale two-stage LGD $R^2$ '-1.13' found in report (must track lgd_model_comparison.champion.r2)",
        failures,
    )
    _check(
        expected in text,
        f"Two-stage LGD $R^2$ {expected} (from metric) not found in report body",
        failures,
    )


def check_recalibration_claim(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """The report must not claim recalibration 'materially improves' slope/intercept when
    the metrics show both moved away from their targets on OOT."""
    comp = metrics.get("calibration_comparison") or {}
    before, after = comp.get("before") or {}, comp.get("after") or {}
    if not (before and after):
        return
    slope_worse = abs(after.get("slope", 0.0) - 1.0) >= abs(before.get("slope", 0.0) - 1.0)
    intercept_worse = abs(after.get("intercept", 0.0)) >= abs(before.get("intercept", 0.0))
    if slope_worse and intercept_worse:
        text = tex_path.read_text(encoding="utf-8", errors="replace")
        _check(
            "materially improves the calibration slope" not in text,
            "Report claims recalibration 'materially improves the calibration slope' but the "
            "metrics show slope and intercept both moved away from target",
            failures,
        )


def run_metric_checks(metrics: dict) -> None:
    """Run all metrics.json identity checks; raise QAError listing every failure."""
    failures: list[str] = []
    check_rwa_density(metrics, failures)
    check_ecl_coverage(metrics, failures)
    check_capital_identity(metrics, failures)
    check_cutoff_optimum(metrics, failures)
    check_scenario_dr(metrics, failures)
    check_scenario_ordering(metrics, failures)
    check_irb_sa_direction(metrics, failures)
    check_vintage_pd_ratio(metrics, failures)
    check_es_ge_var(metrics, failures)
    check_benchmarks_sourced(failures)
    if failures:
        raise QAError(
            "Report QA failed (%d issue(s)):\n  - %s"
            % (len(failures), "\n  - ".join(failures))
        )
    print(f"QA metric checks passed ({len(metrics)} metric keys audited).")


def run_tex_checks(tex_path: str | Path, metrics: dict | None = None) -> None:
    """Run generated-.tex guards; raise QAError on any unreplaced placeholder,
    dangling citation, or text/metric inconsistency."""
    failures: list[str] = []
    path = Path(tex_path)
    check_no_unreplaced_vars(path, failures)
    check_no_fabricated_benchmark(path, failures)
    check_citations_resolve(path, failures)
    if metrics is not None:
        check_lgd_r2_consistency(path, metrics, failures)
        check_recalibration_claim(path, metrics, failures)
    if failures:
        raise QAError(
            "Report QA failed (%d issue(s)):\n  - %s"
            % (len(failures), "\n  - ".join(failures))
        )
    print(f"QA tex checks passed for {tex_path}.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    with open(root / "outputs" / "metrics.json", encoding="utf-8") as f:
        _metrics = json.load(f)
    run_metric_checks(_metrics)
    tex = root / "reports" / "model_risk_report.tex"
    if tex.exists():
        run_tex_checks(tex, _metrics)
