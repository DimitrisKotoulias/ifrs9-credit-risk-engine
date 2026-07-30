# Finding IDs referenced in the code

Comments throughout `src/`, `reports/`, `config/` and `tests/` cite finding IDs such as
`FLAWS-N31` or `AUDIT-C4`. This file is the published index for them.

Those citations used to name two files directly — `docs/AUDIT.md` and `Flaws.md` — both of
which are gitignored as internal working notes. 163 references across 30 files therefore
pointed at documents nobody reading the published repository could open. The comments now
carry stable IDs and this index explains what each one refers to; the internal logs remain
unpublished, but the IDs are stable and every one of them is pinned by a regression test.

## Where the tests are

| Prefix | Review round | Regression tests |
|---|---|---|
| `AUDIT-*` | First internal audit (A/B/C series) | `tests/test_audit_regressions.py` |
| `FLAWS-*` | Second review round (N series) | `tests/test_flaws_regressions.py` |
| `AUDIT_FINDINGS *` | Third-party review, July 2026 (`AUDIT_FINDINGS.md`, published in-repo) | `tests/test_audit_findings_2026_07.py`, `tests/test_hazard_timing.py` |

## Reading a citation

A comment like

```python
# ... the interaction over recently opened accounts now carries its own name (FLAWS-N24).
```

means: this code shape is the *fix*, the ID identifies the defect it fixes, and there is a
test that fails if the defect returns. To find the test, grep the ID across `tests/`:

```bash
grep -rn "FLAWS-N24" tests/
```

## Index of cited IDs

Each ID below appears in at least one source comment. The location given is one
representative site, not the only one.

| ID | Subject | First reference |
|---|---|---|
| AUDIT-A1 | Recalibration gate must test out-of-time evidence, not the in-time partition | `reports/qa_checks.py` |
| AUDIT-A2 | Unconstrained cutoff corner is data-dependent; wording must be derived | `reports/qa_checks.py` |
| AUDIT-A3 | Cost of capital is a charge, not the hurdle — double-counting in RAROC | `reports/qa_checks.py` |
| AUDIT-A5 | Population counts in the report must reconcile with the pipeline's funnel | `reports/qa_checks.py` |
| AUDIT-A12 | Challenger feature parity: no silent training on a reduced feature set | `reports/qa_checks.py` |
| AUDIT-A22 | Model B (underwriting-only) exclusion list must be exported, not implied | `src/credit_risk/pipeline.py` |
| AUDIT-A23 | Platt calibrator exposes `predict_proba`, not `transform` | `src/credit_risk/models/pd_scorecard.py` |
| AUDIT-B3 | Downturn LGD from realised severity, not from shrunk predictions | `src/credit_risk/models/lgd.py` |
| AUDIT-B4 | Relative SICR trigger is skipped when no origination PD exists, not faked | `src/credit_risk/pipeline.py` |
| AUDIT-B5 | Macro scenario configuration | `config/config.yaml` |
| AUDIT-B6 | Phase failures surfaced in `metrics.json` rather than only logged | `src/credit_risk/pipeline.py` |
| AUDIT-C1 | Stage 3 is a hindsight classification taken from resolved status | `src/credit_risk/risk/ifrs9_ecl.py` |
| AUDIT-C4 | Hazard-model event timing and censoring | `src/credit_risk/models/pd_term_structure.py` |
| AUDIT-C5 | Macro scenario axes must separate upside from downside | `reports/qa_checks.py` |
| AUDIT-C7 | Train/OOT grey-zone embargo | `src/credit_risk/data/split.py` |
| FLAWS-N1 | Basel IRB requires a 12-month PD, not the lifetime PD | `reports/qa_checks.py` |
| FLAWS-N2 | Champion/challenger narrative must follow the measured result | `reports/render_latex.py` |
| FLAWS-N3 | Scenario-implied default rates must reconcile with the projection | `reports/qa_checks.py` |
| FLAWS-N4 | Score ↔ PD mapping is pre-recalibration; the report says so | `src/credit_risk/models/pd_scorecard.py` |
| FLAWS-N5 | Recalibrator scoped to the vintages the gate learned from | `reports/qa_checks.py` |
| FLAWS-N6 | Calibration table direction | `reports/qa_checks.py` |
| FLAWS-N7 | Challenger attribution in the report | `reports/qa_checks.py` |
| FLAWS-N8 | Vintage drift claim must match the backtest | `reports/qa_checks.py` |
| FLAWS-N9 | Report build/QA wiring | `reports/render_latex.py` |
| FLAWS-N10 | Vintage backtest aggregation to annual cohorts | `reports/render_latex.py` |
| FLAWS-N11 | Target equation rendered from config, not hand-typed | `reports/render_latex.py` |
| FLAWS-N12 | ECL / staging regression | `tests/test_flaws_regressions.py` |
| FLAWS-N13 | Economic capital uses the same correlation curve as the IRB figure | `tests/test_flaws_regressions.py` |
| FLAWS-N14 | Fallback binner must not pass unnoticed into the report | `reports/qa_checks.py` |
| FLAWS-N15 | Cumulative PD capped at 1.0 under stress | `tests/test_flaws_regressions.py` |
| FLAWS-N16 | Macro overlay anchored at the lifetime horizon | `src/credit_risk/risk/ifrs9_ecl.py` |
| FLAWS-N17 | TTC anchor for the Vasicek inversion | `src/credit_risk/risk/ifrs9_ecl.py` |
| FLAWS-N18 | Stress direction check | `reports/qa_checks.py` |
| FLAWS-N19 | README metrics block regenerated from `metrics.json` | `scripts/update_readme_metrics.py` |
| FLAWS-N21 | Cutoff economics on a consistent per-annum basis | `src/credit_risk/pipeline.py` |
| FLAWS-N22 | Tests must not patch the production path they are testing | `tests/conftest.py` |
| FLAWS-N24 | `revol_util_x_new_acc` naming vs the loader's own interaction | `src/credit_risk/models/pd_scorecard.py` |
| FLAWS-N25 | Hazard term-structure feature preparation | `src/credit_risk/models/pd_term_structure.py` |
| FLAWS-N26 | Hosmer-Lemeshow evaluated sample size is reported | `tests/test_validation.py` |
| FLAWS-N28 | EL/ECL reconciliation table | `reports/render_latex.py` |
| FLAWS-N29 | Four-stage feature selection funnel recorded and reported | `reports/qa_checks.py` |
| FLAWS-N30 | Binning implementation provenance | `src/credit_risk/features/binning.py` |
| FLAWS-N31 | Missing values get their own WoE bin instead of a −9999 sentinel | `tests/test_flaws_regressions.py` |
| FLAWS-N32 | Binner kind surfaced into `metrics.json` | `reports/qa_checks.py` |
| FLAWS-N33 | LGD validation on disjoint select/report halves | `tests/test_flaws_regressions.py` |
| FLAWS-N34 | Model claims in the report checked against the code | `reports/qa_checks.py` |
| FLAWS-N35 | Cutoff grid hurdle verdict derived from the grid | `reports/render_latex.py` |
| FLAWS-N36 | Excluded-feature list exported for Model B | `src/credit_risk/pipeline.py` |
| FLAWS-N37 | Orphan figures: nothing rendered that the report does not reference | `reports/qa_checks.py` |
| FLAWS-N38 | Scorecard points table rendering | `reports/render_latex.py` |
| FLAWS-N39 | ECL macro sensitivity must respond to the shock | `reports/qa_checks.py` |
| FLAWS-N42 | Validation report structure | `src/credit_risk/validation/report.py` |
| FLAWS-N43 | Challenger/champion comparison on identical engineered frames | `src/credit_risk/pipeline.py` |
| FLAWS-N44 | Lifetime-PD model needs its own discrimination evidence | `src/credit_risk/pipeline.py` |
| FLAWS-N45 | Dead WoE helper documented as such | `src/credit_risk/features/woe.py` |

## July 2026 review

The third review round is published in full as [`AUDIT_FINDINGS.md`](../AUDIT_FINDINGS.md)
at the repository root, together with the verdict on each finding (valid / already fixed /
overstated). Its regressions live in `tests/test_audit_findings_2026_07.py` and
`tests/test_hazard_timing.py`.
