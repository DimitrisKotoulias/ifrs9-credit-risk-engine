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


def check_recalibration_applied(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """The report must not describe a recalibration that was never attached.

    The calibrator is fitted only when the in-time test Hosmer-Lemeshow test rejects. When
    it does not, ``predict_proba`` returns raw PDs and the before/after table is a
    diagnostic. The shipped report previously claimed the transform was "applied to the OOT
    set", "retained", and feeding EL/RWA/staging (docs/AUDIT.md finding A1).
    """
    cal = metrics.get("calibration") or {}
    if cal.get("isotonic_applied") is None and "recalibration_applied" not in cal:
        return
    comp = metrics.get("calibration_comparison") or {}
    applied = bool(
        cal.get("recalibration_applied",
                comp.get("applied_in_production", cal.get("isotonic_applied", False)))
    )
    # The gate's own record must agree with the attach state.
    gate = cal.get("recalibration_gate") or {}
    if gate:
        _check(
            bool(gate.get("accepted", False)) == applied,
            f"recalibration_gate.accepted={gate.get('accepted')} contradicts "
            f"recalibration_applied={applied}",
            failures,
        )
        _check(
            applied or cal.get("method_chosen", "none") == "none",
            f"method_chosen={cal.get('method_chosen')!r} while no transform is attached",
            failures,
        )
    if applied:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    banned = [
        "and applied to the OOT set",
        "we retain the recalibrated PDs",
        "corrected via the isotonic regression recalibration",
    ]
    for phrase in banned:
        _check(
            phrase not in text,
            f"Report claims recalibration is deployed ({phrase!r}) but no calibrator is "
            "attached to the production scorecard (calibration.isotonic_applied is false)",
            failures,
        )
    _check(
        cal.get("method_chosen") != "isotonic" or cal.get("isotonic_applied", False),
        "metrics.json reports method_chosen='isotonic' while isotonic_applied is false; "
        "record 'none' when nothing is attached",
        failures,
    )


def check_cutoff_corner_direction(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """"Approve everyone" wording is only valid if the unconstrained argmax really is
    full approval. On a book where every grid RAROC is negative the argmax is the most
    *exclusive* non-empty cutoff (docs/AUDIT.md finding A2)."""
    corner = metrics.get("cutoff_raroc_max") or metrics.get("cutoff_profit_argmax") or {}
    if not corner:
        return
    approval = float(corner.get("approval_rate", 0.0))
    if approval >= 0.99:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    banned = [
        "approving the entire through-the-door population",
        "implies 100\\% approval",
        "(full approval)",
    ]
    for phrase in banned:
        _check(
            phrase not in text,
            f"Report describes the unconstrained cutoff corner as full approval ({phrase!r}) "
            f"but the argmax row approves only {approval:.4%} of the population",
            failures,
        )


def check_capital_charge_rate(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """The charge netted out of expected profit is the cost of capital, not the RAROC
    hurdle (docs/AUDIT.md finding A3)."""
    coc = metrics.get("cutoff_cost_of_capital")
    hurdle = metrics.get("cutoff_raroc_hurdle")
    if coc is None or hurdle is None or abs(float(coc) - float(hurdle)) < 1e-9:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    hurdle_pct = f"{float(hurdle) * 100:.2f}\\%"
    _check(
        f"a {hurdle_pct} charge on economic capital" not in text,
        f"Report states a {hurdle_pct} charge on economic capital, but the cost of capital "
        f"actually charged is {float(coc) * 100:.2f}%",
        failures,
    )


def check_population_counts(tex_path: Path, metrics: dict, failures: list[str]) -> None:
    """Every large loan count in the prose must trace to a metrics.json population.

    The abstract previously carried these as hard-coded literals, and the pipeline exposed
    only a mis-named `n_accepted_raw` (actually the resolved-outcome count), so a drifted
    figure could not be detected (docs/AUDIT.md finding A5).
    """
    n_train = int(metrics.get("n_train", 0))
    n_test = int(metrics.get("n_test", 0))
    n_oot = int(metrics.get("n_oot", 0))
    n_resolved = int(metrics.get("n_resolved_outcome", metrics.get("n_accepted_raw", 0)))
    n_modelling = n_train + n_test + n_oot
    known = {
        int(metrics.get("n_accepted_file", 0)),
        n_resolved,
        int(metrics.get("n_rejected_raw", 0)),
        n_train,
        n_test,
        n_oot,
        n_modelling,
        max(0, n_resolved - n_modelling),  # grey-zone loans dropped by the split
    }
    known.discard(0)
    if not known:
        return
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    # Comma-grouped integers with 2+ groups: portfolio-scale populations. Currency figures
    # in this report are always prefixed with \$ (optionally followed by a minus sign), so
    # they are stripped out first rather than matched around.
    text_no_currency = re.sub(r"\\?\$-?\s?[\d,]+(?:\.\d+)?", " ", text)
    for m in re.finditer(r"\b(\d{1,3}(?:,\d{3}){2,})\b", text_no_currency):
        value = int(m.group(1).replace(",", ""))
        # Only police the population range; monetary totals are billions and are checked
        # by the identity tests instead.
        if not (100_000 <= value <= 50_000_000):
            continue
        _check(
            value in known,
            f"Population count {m.group(1)} in the report does not match any metrics.json "
            f"population {sorted(known)}",
            failures,
        )


def check_challenger_feature_parity(metrics: dict, failures: list[str]) -> None:
    """Champion and challenger must consume the same number of predictors.

    ``PDChallenger`` used to silently drop requested features missing from the raw frame,
    so the tree models trained on 13 of the scorecard's 15 predictors while the report
    claimed parity (docs/AUDIT.md findings A12 / B2).
    """
    shap_rows = (metrics.get("challenger") or {}).get("shap_mean_abs") or []
    if not shap_rows:
        return
    root = Path(__file__).resolve().parent.parent
    sc_path = root / "outputs" / "scorecard_tables.json"
    if not sc_path.exists():
        return
    with open(sc_path, encoding="utf-8") as f:
        selected = json.load(f).get("selected_features", [])
    if not selected:
        return
    _check(
        len(shap_rows) == len(selected),
        f"Challenger feature parity broken: challenger used {len(shap_rows)} features vs "
        f"the scorecard's {len(selected)}; the champion/challenger comparison is not "
        "like-for-like",
        failures,
    )



def check_scenario_axes_separate(metrics: dict, failures: list[str]) -> None:
    """Upside and downside must differ on every macro axis.

    FEDFUNDS previously floored to 0.1 in both scenarios and CPI_inflation carried no
    delta, so two of five axes contributed nothing to scenario separation while the
    report tabulated them as assumptions (docs/AUDIT.md finding C5).
    """
    inputs = metrics.get("macro_scenario_inputs") or {}
    up, down = inputs.get("upside") or {}, inputs.get("downside") or {}
    if not (up and down):
        return
    identical = [
        k for k in up
        if k in down and abs(float(up[k]) - float(down[k])) < 1e-9
    ]
    _check(
        not identical,
        f"Macro scenario axes identical in upside and downside: {identical}; "
        "they contribute nothing to scenario separation",
        failures,
    )



def check_no_unescaped_percent(tex_path: Path, failures: list[str]) -> None:
    """No inline unescaped ``%`` in the generated .tex.

    A percent sign is a LaTeX comment: it deletes the rest of the source line from the
    PDF with no error and no warning. Because the template writes each paragraph as one
    long line, a single stray ``%`` in a value injected from Python can silently remove
    hundreds of characters of evidence -- which is exactly what happened to the
    recalibration decision in an earlier build (docs/AUDIT.md finding A1).

    Whole-line comments (optionally indented) are legitimate and ignored.
    """
    failures_before = len(failures)
    for lineno, line in enumerate(tex_path.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue  # a real comment line
        for col, ch in enumerate(line):
            if ch != "%":
                continue
            if col > 0 and line[col - 1] == "\\":
                continue  # escaped
            _check(
                False,
                f"Unescaped '%' at {tex_path.name}:{lineno}:{col + 1} truncates the rest "
                f"of the line in the PDF: ...{line[max(0, col - 60):col + 1]!r}",
                failures,
            )
            break
        if len(failures) - failures_before >= 5:
            failures.append("... further unescaped '%' occurrences suppressed")
            return


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
    check_challenger_feature_parity(metrics, failures)
    check_scenario_axes_separate(metrics, failures)
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
    check_no_unescaped_percent(path, failures)
    check_no_fabricated_benchmark(path, failures)
    check_citations_resolve(path, failures)
    if metrics is not None:
        check_lgd_r2_consistency(path, metrics, failures)
        check_recalibration_claim(path, metrics, failures)
        check_recalibration_applied(path, metrics, failures)
        check_cutoff_corner_direction(path, metrics, failures)
        check_capital_charge_rate(path, metrics, failures)
        check_population_counts(path, metrics, failures)
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
