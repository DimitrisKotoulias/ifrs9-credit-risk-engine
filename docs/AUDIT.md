# Model Risk Report & Codebase Audit

Audit of the `credit-risk-ecl` project: `src/credit_risk/**`, the report generator
(`reports/render_latex.py`), the QA layer (`reports/qa_checks.py`), `config/config.yaml`,
`outputs/metrics.json` and the generated `reports/model_risk_report.tex`.

**Method.** The report is generated: `render_latex.py` substitutes live metrics into a
hard-coded narrative. Every numeric claim in the `.tex` was recomputed against
`outputs/metrics.json`, and every methodological claim was traced to the code that produces
it. The numeric identities already guarded by `qa_checks.py` (RWA density, ECL coverage,
capital = RWA x 8%, scenario DR ordering, ES >= VaR >= EL) all hold. The defects are in the
prose, in three computations that move headline numbers, and in disclosure.

Severity: **High** = wrong number or a claim that misrepresents what the model does ·
**Medium** = internally inconsistent or misleading · **Low** = cosmetic / hygiene.

---

## Tier A — Report claims contradicted by the code

### A1 · High · Recalibration is described as applied; it never is

*Claim* (§7.2): "Post-model recalibration (isotonic regression) was therefore fitted
out-of-sample on the in-time test partition and **applied to the OOT set**"; "we **retain
the recalibrated PDs**". §7.3: the drift is "corrected via the isotonic regression
recalibration … which **feeds Expected Loss, Basel RWA, and SICR-origination staging**".

*Reality*: `validation/report.py:110` fits a calibrator only when the **in-time test**
Hosmer-Lemeshow p-value is below 0.05. This run: `calibration.test.hl_pvalue = 0.1330` →
`calibrator = None`, `calibration.isotonic_applied = false`. `pipeline.py:198-212` then
compares a Platt calibrator (fitted on **train**) against the *raw* PDs on test and attaches
neither. `scorecard._calibrator` stays `None`, so `predict_proba` returns raw PDs
(`models/pd_scorecard.py:432`) for EL, RWA, staging and the cutoff sweep.

The before/after table (§7.2) is produced by a *separate* throwaway `IsotonicRegression`
built at `pipeline.py:228-230` purely for the table. It never touches production.

*Impact*: the reported correction of the 2016-2018 under-prediction does not exist. The
documented drift (PD ratio ~0.65 across every 2016-2018 vintage) flows uncorrected into EL,
RWA and the ECL staging thresholds.

### A2 · High · The unconstrained cutoff corner is inverted

*Claim* (§9): "the unconstrained profit-maximising *and* RAROC-maximising cutoff is the same
corner solution — **approving the entire through-the-door population**, at a portfolio RAROC
of -11.75% — because higher-risk grades carry interest rates high enough to remain
RAROC-accretive". "Because unconstrained optimisation therefore implies **100% approval** …".

*Reality*: `cutoff_profit_argmax` and `cutoff_raroc_max` are both **cutoff 600 with
`approval_rate = 0.00015`** (0.015%) and `expected_profit = -14,473`. Every RAROC on the
400-800 grid is negative, so the argmax over non-empty rows is the *most exclusive*
threshold, not the most inclusive. The -11.75% figure the report quotes is that corner's
RAROC — the number is live and correct, the sentence around it is not.

The same false claim appears in `business/cutoff.py:148-150` (`raroc_argmax_cutoff`
docstring: "typically the most inclusive cutoff … i.e. a corner") and in the
`pipeline.py:1118-1122` comment ("both … are the corner solution 'approve everyone'").

*Impact*: the report's stated economic rationale for imposing a risk-appetite constraint is
backwards. The constraint is needed because the book is unprofitable at every cutoff, not
because it is profitable at all of them.

### A3 · Medium · Capital charge rate stated as 15%, applied as 12%

§9 opening: "a **15.00%** charge on economic capital … netted out". The bullet list in the
same section: "**Capital Cost:** 12.0% hurdle rate applied to the Minimum Capital Reserve".
`pipeline.py:1099` applies `cfg.business.cost_of_capital` = **0.12**; 0.15 is
`raroc_hurdle`, a comparison threshold that is never charged. The template substitutes
`__RAROC_HURDLE__` into the charge sentence (`render_latex.py:1819`).

### A4 · Medium · "Stratified" holdout is a plain random sample

§2.3: "**Stratified** random 20% sample from the training period … **Stratification
preserves** the 82.9%/17.1% good/bad ratio". `data/split.py:86` is
`train_pool.sample(frac=cfg.holdout_frac, random_state=seed)` — no `stratify` argument, no
class-balanced sampling anywhere in the split path.

### A5 · Medium · Portfolio sizes are hard-coded literals over a mis-named metric

Title page and abstract state "**2,260,701** accepted loans" / "N = 2.26M"; §2.5 repeats
"Full-portfolio statistics (N = 2,260,701 accepted loans) are used for portfolio-level
capital and impairment calculations". These are literals in `render_latex.py:1065,1077` and
`README.md:3,17`. The rejects figure printed immediately beside them (27,648,741) *is*
live, so one half of the sentence tracks the data and the other does not.

2,260,701 does match the accepted-loans file (`loader.py:69` logs
"Loaded 2260701 accepted loans"), so the literal is currently correct — but nothing keeps
it correct, and the pipeline exposed no metric it could be checked against. What the
pipeline *did* export was `n_accepted_raw = 1,369,566` (`pipeline.py:63`), which despite
its name is `len(split.full_accepted)` — the population *after* `define_target` drops
indeterminate statuses, i.e. the resolved-outcome count, not the raw file count.

Three distinct populations were therefore collapsed into one ambiguous name:

| Population | Count | Definition |
|---|---|---|
| Accepted file rows | 2,260,701 | as read from `accepted_2007_to_2018Q4.csv.gz` |
| Resolved outcome | 1,369,566 | after `define_target` drops Current / In Grace Period / Late (16-30) |
| Modelling population | 992,661 | `n_train + n_test + n_oot`, i.e. resolved minus the excluded 2015 grey zone |

§2.5's claim that the 2.26M figure is what feeds "portfolio-level capital and impairment
calculations" is wrong on any reading: EL, RWA and ECL run on 992,661 loans.

### A6 · Medium · Split percentages and bad rate are hard-coded

"36.6% / 9.2% / 54.3% of modelling population", "Training bad rate: 17.1%", "82.9%/17.1%
good/bad ratio" are literals in `render_latex.py:1172-1173`; only the `N` values are
substituted. They happen to match this run and will drift silently on the next one.

### A7 · High · Downturn LGD is the p90 of predictions, not of severity

§4.1: "a conservative **Downturn LGD** is estimated at the **90th percentile of the default
severity distribution**"; §4.2 repeats "applied conservatively at the 90th percentile of the
severity distribution".

`models/lgd.py:197` (and `:269` after challenger promotion) computes
`np.percentile(loss_preds, 90)` where `loss_preds` are the **model's own predictions**.
Predictions are shrunk toward the conditional mean, so their p90 sits just above the mean —
which is exactly what is observed: mean 0.8931, "downturn" 0.9066, a 1.35pp buffer where the
realised severity distribution has mass at 1.0. The Basel downturn add-on is therefore
close to vacuous.

### A8 · Medium · Stage-migration matrix is mislabelled and self-contradicting

Caption: "IFRS 9 Stage Migration Matrix (**Origination** -> Reporting Date)"; text: "Rows
represent the **origination** stage". Footnote justifies the all-zero Stage-3 row: "no loan
originates directly into Stage 3, as Stage 3 classification requires 90+ days past due …
which by definition cannot occur at origination (t=0)".

`pipeline.py:952-969` builds `_stages_t0` as a **simulated 12-months-ago** state
(`pd_12m_t0 = 0.5 * pd_orig + 0.5 * pd_current`, restricted to `mob_months >= 12`), and the
construction only ever assigns stage 1 or 2 — the zero row is a property of the code, not of
IFRS 9. The footnote's own argument also contradicts the 368,500 loans the table shows
"originating" in Stage 2, which is equally impossible at t=0 (SICR is defined relative to
origination). The report's own §10 limitation correctly describes the 12-month
interpolation; the table caption does not.

### A9 · High · SICR triggers described as DPD-based; neither is

§6.1 states the SICR backstop is "**30+ days past due (DPD)**" and Stage 3 is "**90+ DPD or
default**".

- Backstop (`risk/ifrs9_ecl.py:142-146`): `delinq_2yrs >= 1` — the count of delinquencies in
  the **two years before origination**, an application-time bureau field. It is not a
  current-DPD measure and cannot be one.
- Stage 3 (`risk/ifrs9_ecl.py:125-130`): `df["target"]`, the **realised default outcome** —
  the modelling target itself. See C1 for the consequence.

The report's own §10 states the data has no monthly DPD panel, which is precisely why
neither trigger can be what §6.1 claims.

### A10 · Medium · Two tables still describe the rejected LGD model

`lgd_model_comparison.recommended = "challenger"` and `pipeline.py:631-635` calls
`promote_to_challenger`, so the deployed severity model is **LightGBM**. §7.7 and §4.2 say
so. But:

- Tech-stack table (§10 Appendix C): "LGD Modeling — statsmodels GLM (fractional logit) —
  **Two-stage cure + severity modeling**".
- Assumptions table (§10.1): "LGD Model — **Fractional logit** on funded_amnt proxy EAD".

Both describe the model that was rejected for an out-of-sample R^2 of -2.70.

### A11 · Low · KS sample size contradicts its own table

§4.2: "at this sample size (**n ~ 150,000**) the test is hyper-sensitive". The KS in
question is computed on the report half, `lgd_validation.n_test = 74,981` — as the caption of
the very table it refers to states ("n=74,981 held-out defaults"). 150,000 is the size of
both halves combined, only one of which is used.

### A12 · High · The challenger is not trained on the same feature set

§3.6: "A non-linear LightGBM model was trained on the **same feature set** as a challenger".
§7.8 draws the champion conclusion from that comparison.

`models/pd_challenger.py:77` (and `:232` for the multi-model benchmark) filters the requested
feature list with `f in X_train.columns`. `grade_enc` and `term_enc` are engineered *inside*
`PDScorecard.fit` (via `_encode_categoricals`) and are absent from the raw `df_train` passed
to the challenger, so they are **silently dropped**. The challenger, XGBoost, Random Forest
and the ensemble all train on **13 of 15** features — missing the two highest-IV predictors
in the model (`grade_enc` IV 0.396, `term_enc` IV 0.200, ranks 2 and 3 of the whole IV
table). `challenger.shap_mean_abs` confirms 13 entries.

*Impact*: the entire "the interpretable scorecard beats gradient boosting out-of-time"
finding — §3.6, §7.8, Table 16, the DeLong test and the paired bootstrap — is measured
against a handicapped challenger. This must be re-run before the conclusion can stand.

### A13 · Low · AIC lag text claims a sign verification that failed

§6.3: the AIC grid search is described as "**verifying the economically-correct positive
sign** at the selected lag". `macro_ts.aic_lag_selection.unrate_sign_ok = false` with
coefficient -0.0123 at lag 0, and the table row directly below the sentence prints
"UNRATE - (spurious)".

### A14 · Low · Granger Bonferroni base alpha misstated

Footnote: "a minimum p-value above alpha_corr — **even if below the nominal 0.05** — is
reported as no causality, guarding against multiple-testing false positives".
`validation/macro_ts.py:84` is `alpha_corrected = 0.10 / usable` = 0.025. The base level is
**0.10**, not 0.05; a genuine Bonferroni on 0.05 over 4 lags would be 0.0125. As written the
threshold is presented as stricter than 0.05 while in fact being looser.

### A15 · Medium · Lifetime-PD vintage table: two different restrictions, and a hidden gap

- Body text: "restricted to vintages **matured by 2016**". Footnote on the same table:
  "Restricted to vintages **originated in or before 2016**". These are different populations.
- The table includes the 2016 vintage (N = 297,651) whose 60-month loans cannot be matured in
  a 2018Q4 snapshot — its 24.46% "observed default rate" is right-censored.
- Both this table and the vintage PD backtest jump straight from **2014 to 2016** with no
  note. `data/split.py:78-80` drops the entire grey zone between `train_cutoff` (2015-01-01)
  and `oot_cutoff` (2016-01-01) — the whole 2015 cohort. §2.3 defines train, test and OOT as
  100% of the "modelling population" without ever stating that a full origination year was
  excluded.

### A16 · Medium · ECL formula shows time-varying LGD(t) and EAD(t); the code uses constants

§6.2: `ECL = sum_t m(t) * LGD(t) * EAD(t) / (1+EIR)^t`, and §4.3 gives an amortisation
formula for `EAD(t)`. `risk/ifrs9_ecl.py:183-186` broadcasts a **scalar** per-loan LGD and a
**scalar** per-loan EAD across every month; `models/ead.py:196` returns one EAD per loan at a
single estimated MOB. Exposure never amortises inside the ECL sum, which overstates
provisions on amortising loans.

### A17 · Medium · Reject-inference table mixes three populations

The table reads as accepts-only scorecard (0.4011) vs through-the-door scorecard (0.3818).
`business/reject_inference.py:118-136` computes **both** Ginis from the *same* refitted TTD
model, in-sample — one on the train accepts, one on the combined accepts+parcelled set. The
"through-the-door" Gini is also scored against *inferred* labels for the rejects.
`pipeline.py:1224` then adds that shift to `metrics["gini"]`, which is the **original**
scorecard's **test** Gini. Three populations and two models in a two-row table.

### A18 · Low · Cross-reference points at the wrong section

§4.1 and §4.2 cite `\ref{subsec:benchmarks}` for the LGD champion/challenger selection. That
label is §10.3 (the literature comparison table), which does not describe the selection
procedure; the selection is described in §4.2 itself and its result in §7.7.

### A19 · Low · `metrics.json` contradicts itself

`calibration.isotonic_applied: false` sits next to `calibration.method_chosen: "isotonic"`.
The `method_chosen` branch at `pipeline.py:204-212` records a choice even when no calibrator
is attached.

### A20 · Low · The dual-SHAP figure shows the same model twice

`pipeline.py:510-528` fits a "bureau-only" challenger by removing price features
(`int_rate`, `grade_enc`, …) from the feature list. Those columns were already dropped by
A12, so the bureau feature list equals the full one and
`challenger.shap_mean_abs_bureau` is **byte-identical** to `challenger.shap_mean_abs`. The
`shap_comparison.png` figure is two identical panels.

---

## Tier B — Computational defects that move headline numbers

### B1 · High · Lifetime ECL sums 60 months for every loan

`DiscreteHazardModel.predict_term_structure` sizes `marginal_pd` at
`T = min(max(all terms), max_horizon)` = **60** for the whole portfolio
(`models/pd_term_structure.py:208`). `compute_ecl_single_scenario` then sums **all 60
columns** for Stage 2 and Stage 3 lifetime ECL (`risk/ifrs9_ecl.py:191`). A 36-month loan is
charged 60 months of marginal default probability.

The same class gets this right for `pd_lifetime`, which indexes each loan's own term
(`pd_term_structure.py:246-247`) — only the ECL sum is unmasked.

*Impact*: `total_ecl` and `ecl_coverage` are overstated for every 36-month loan in Stage 2.

### B2 · High · Challenger feature loss (see A12)

`pd_challenger.py:77` / `:232` silently drop requested features that are absent from the raw
frame. Fix by engineering the columns before fitting (the pipeline already does this at
`pipeline.py:1202` for reject inference) and by raising instead of filtering.

### B3 · High · Downturn LGD off the wrong distribution (see A7)

`lgd.py:197,269` — take the p90 of `compute_realised_lgd(df_defaults)` on loss-incurring
defaults, not of the predictions. Feeds Basel RWA and the stress test.

### B4 · High · The relative SICR trigger is inert

`pipeline.py:894` sets `pd_12m_orig = df_rwa["pd_pred"]` — the **current** scorecard PD. The
comment above it admits "we have no separate origination snapshot". `pd_orig_lifetime` is
therefore today's PD compounded over the term, and the "lifetime PD has increased 2.5x since
origination" test at `ifrs9_ecl.py:137` degenerates into a comparison between two *different
models'* lifetime PDs (hazard vs scorecard) on the same loan at the same date. It measures
model disagreement, not credit deterioration.

### B5 · Low · `macro_gamma` is dead configuration

`config/config.yaml:129` documents `macro_gamma: 0.8  # hazard scale = exp(gamma * macro_shock)`
and `DiscreteHazardModel.__init__` accepts `macro_gamma`, but the value is never read — the
hazard uses a Vasicek transform instead (`pd_term_structure.py:184`). The module docstring
(`pd_term_structure.py:8`) still shows the `sigmoid(... + gamma*macro_t)` form.
`_hazard_at_t` (`:164-186`) is dead code duplicating the inlined loop. Separately,
`config.yaml:119-128` `scenarios.*.macro_shock` values are always overwritten by
`fit_macro_model` at `pipeline.py:875-880` and only apply on the fallback path.

### B6 · Medium · The stress test changes PD and LGD simultaneously

Base `total_el` uses per-loan `lgd_pred` (mean 0.8931). `stress_el`
(`pipeline.py:1248-1252`) uses `downturn_lgd` (0.9066) **and** the Z=-2.0 stressed PD. The
report attributes the entire +127.5% EL increase to the systematic shock; roughly 1.5pp of
it is the LGD switch.

---

## Tier C — Disclosure

### C1 · High · Stage 3 is the realised outcome, and it drives ~93% of ECL

Stage 3 is assigned from `df["target"]` (A9) and its ECL is `LGD x EAD` with PD forced to 1
and no discounting (`risk/ifrs9_ecl.py:194`). That term is completely insensitive to the PD
model.

The report's own what-if table proves how dominant it is:

| Shock | Delta ECL |
|---|---|
| LGD +10pp | **+11.2%** (exactly linear: 0.10 / 0.8931 = 11.2%) |
| EAD +15% | **+15.0%** (exactly linear) |
| **PD +50%** | **+3.3%** |

ECL is linear in LGD and EAD but almost flat in PD. Solving `0.5 * (1 - f) = 0.033` gives
`f ~ 93%`: about 93% of the reported ECL comes from the PD-independent Stage 3 term. The
headline $3.00bn provision is, to first order, *known defaulters x LGD x EAD* — a hindsight
accounting identity, not a forward-looking estimate. Every forward-looking element of the
engine (macro scenarios, Vasicek Z-shocks, the hazard term structure) moves the remaining
~7%.

This is the single most consequential finding in the audit and belongs in the report's
limitations, not just here.

### C2 · Medium · EL, RWA and ECL are computed in-sample

`pipeline.py:672`: `df_all = concat(df_train, df_test, df_oot)`. The scorecard, hazard model
and LGD model were all fitted on rows inside `df_all`, so the portfolio-level EL, RWA, ECL
and cutoff sweep are partly in-sample. Not wrong for a provisioning exercise on a closed
book, but it is never stated.

### C3 · Medium · The 2015 origination cohort is silently excluded (see A15)

### C4 · High · Every hazard event is pinned to the loan's final month

`models/pd_term_structure.py:144-146`:

```python
y = np.zeros(len(rep_indices), dtype=int)
ends = np.cumsum(T_all) - 1
y[ends] = targets
```

Every loan contributes its **full** term as person-periods with the default event placed at
the **last** period, and there is no censoring. The data has no observed default month, so
this is unavoidable without a servicing panel — but the consequence is that the estimated
`h(t)` rises with MOB largely as an artefact of the construction, and this term structure is
what drives the ECL sum. §3.7 discloses the synthesised duration for the *Cox challenger*;
the production hazard model carries the same limitation and does not disclose it.

### C5 · Medium · Two macro scenario inputs are degenerate

`risk/ifrs9_ecl.py:589,596` floor `FEDFUNDS` at 0.1 in **both** the upside and downside
scenarios, so the two are identical on that axis (`macro_scenario_inputs` confirms
0.1 / 0.1). `CPI_inflation` has no scenario delta at all and sits at the sample mean in all
three. The report's scenario-inputs table prints the identical values without comment. The
sign prior also forces `FEDFUNDS` positive, so the modelled recession (which cuts rates)
*reduces* defaults through that channel.

### C6 · Low · The baseline scenario is Z = -0.19, not Z = 0

`macro_implied_shocks.baseline = -0.1915`. This is mathematically correct — the Vasicek
conditional PD satisfies `p(0) != PD` — and follows from
`fit_macro_model` recentring the baseline default rate onto the TTC rate. But the report
declares "Z < 0 corresponds to an adverse macroeconomic shock (recession)" as a global
convention, so the baseline scenario reads as a mild recession. One clarifying sentence is
needed.

### C7 · Low · `_assert_no_overlap` can never fail

`data/split.py:105-119` compares `set(train.index) | set(test.index)` against
`set(oot.index)`. Those index sets are disjoint by construction of the mutually exclusive
date masks, so the "data leakage bug" guard is vacuous.

---

## What the QA layer already covers, and what it misses

`reports/qa_checks.py` correctly enforces the numeric identities (RWA density, ECL coverage,
capital, cutoff traceability to the grid, scenario DR recomputation and ordering, IRB/SA
direction, vintage PD ratios, ES >= VaR >= EL, benchmark registry sourcing, unreplaced
placeholders, citation resolution) and two prose guards
(`check_lgd_r2_consistency`, `check_recalibration_claim`). All pass.

Every Tier-A finding escaped it because there is no guard tying a *methodological* sentence
to the metric that contradicts it. The remediation adds five: recalibration-applied,
cutoff-corner direction, population-count traceability, challenger feature parity, and
capital-charge rate.

---

## Remediation status

All Tier-A prose fixes are **data-derived**: the sentence is generated from `metrics.json`
rather than asserted, so it cannot drift out of sync on the next run.

| ID | Action | Where |
|---|---|---|
| A1 | Recalibration narrative branches on whether a calibrator is actually attached; when it is not, the report states plainly that all PDs are raw and the before/after table is a diagnostic | `render_latex.py` (`recalib_status`, `recalib_production_note`) |
| A2 | Corner wording derived from `cutoff_profit_argmax.approval_rate`; the "approve everyone" claim now only renders when approval ≥ 99% | `render_latex.py` (`corner_desc`), `cutoff.py`, `pipeline.py` comments |
| A3 | Charge sentence substitutes `cutoff_cost_of_capital`, not the hurdle | `render_latex.py`, new metric in `pipeline.py` |
| A4 | "Stratified" removed; the sample is described as simple random | `render_latex.py` |
| A5 | Three populations exported separately (`n_accepted_file`, `n_resolved_outcome`, modelling total) and all prose counts derived | `loader.py`, `split.py`, `pipeline.py`, `render_latex.py`, `update_readme_metrics.py` |
| A6 | Split percentages and bad rates derived from `n_train/n_test/n_oot` and new `train_bad_rate` | `render_latex.py`, `pipeline.py` |
| A7 | Report describes the downturn figure as a percentile of *predicted* LGD, with the uplift in pp stated; **and** the underlying computation was corrected (B3) | `render_latex.py`, `lgd.py` |
| A8 | Caption and text changed to "12 Months Prior → Reporting Date"; the zero Stage-3 row is attributed to the reconstruction, not to IFRS 9 | `render_latex.py` |
| A9 | Both triggers described as implemented: `delinq_2yrs` proxy and terminal-status Stage 3 | `render_latex.py`, `ifrs9_ecl.py` docstring |
| A10 | Tech-stack and assumptions tables now name LightGBM as deployed | `render_latex.py` |
| A11 | Sample size substituted from `lgd_validation.n_test` | `render_latex.py` |
| A12 | Feature-parity sentence derived from the actual counts; parity enforced in code (B2) | `render_latex.py`, `pd_challenger.py`, `pipeline.py` |
| A13 | Sign-verification claim replaced with what the diagnostic reports | `render_latex.py` |
| A14 | Bonferroni base α stated as 0.10 with the 0.05 comparison spelled out | `render_latex.py` |
| A15 | Restriction wording reconciled; 2015 grey-zone exclusion disclosed in §2.3 and in the table footnote | `render_latex.py` |
| A16 | Implementation note added: LGD(t)/EAD(t) held constant | `render_latex.py` |
| A17 | Measurement caveat added explaining the three populations behind the two rows | `render_latex.py` |
| A18 | Cross-reference retargeted to §4.2 | `render_latex.py` |
| A19 | `method_chosen` records `"none"` when nothing is attached; `applied_in_production` added | `pipeline.py` |
| A20 | Dual-SHAP figure skipped when the two feature sets coincide (and after B2 they no longer do) | `pipeline.py` |
| B1 | `term_horizon_mask` truncates the lifetime sum at each loan's own term | `ifrs9_ecl.py` |
| B2 | `_resolve_features` raises instead of filtering; pipeline hands challengers the engineered frames | `pd_challenger.py`, `pipeline.py` |
| B3 | Downturn LGD taken from realised severity, floored at mean predicted LGD | `lgd.py` |
| B4 | Relative SICR trigger skipped when no origination snapshot exists; absolute + backstop still fire | `ifrs9_ecl.py`, `pipeline.py` |
| B5 | `macro_gamma` removed from config and model; dead `_hazard_at_t` replaced by the used `_apply_macro_shock` | `pd_term_structure.py`, `config.yaml`, `utils/config.py` |
| B6 | `stress_el` uses the same per-loan LGD as base EL; the downturn-LGD variant exported separately | `pipeline.py` |
| C1–C6 | New §10 subsection "Stage 3 Is a Hindsight Classification, and It Dominates the ECL" plus six added limitation bullets, all with derived figures | `render_latex.py` |
| C7 | `_assert_no_overlap` now checks all three pairs and asserts temporal separation | `split.py` |

### New QA guards

`reports/qa_checks.py` gains five guards, each verified to fire against the pre-fix report:

| Guard | Catches |
|---|---|
| `check_recalibration_applied` | Deployment language when no calibrator is attached; `method_chosen` claiming a method that was not applied |
| `check_cutoff_corner_direction` | "Approve everyone" wording when the argmax approves <99% |
| `check_capital_charge_rate` | The hurdle being described as the charge |
| `check_population_counts` | Any loan count in the prose that traces to no `metrics.json` population |
| `check_challenger_feature_parity` | Challenger trained on fewer predictors than the champion |

### Regression tests

`tests/test_audit_regressions.py` — 11 tests pinning B1 (per-term horizon truncation),
B2 (feature resolution raises), B3 (realised-severity percentile) and B4 (absolute SICR
fires without an origination PD; relative fires only with one).

---

## Post-fix results

Full pipeline re-run after the Tier-B fixes. Movements are confined to where they were
predicted, with one consequential surprise.

| Metric | Before | After | Driver |
|---|---:|---:|---|
| `total_ecl` | $2,999,114,462 | $2,371,560,537 | **B1** — 36-month loans no longer charged 60 months of marginal PD |
| `ecl_coverage` | 31.78% | 25.13% | B1 |
| `downturn_lgd` | 0.9066 | 1.0000 | **B3** — p90 of realised severity, not of shrunk predictions |
| `total_rwa` | $17,756,509,185 | $19,585,499,179 | B3 (downturn LGD feeds the IRB capital formula) |
| RWA density | 188.2% | 207.5% | B3 |
| `stress_el` | $3,714,353,414 | $3,645,431,011 | **B6** — LGD held at the base-case value, so the delta is now pure PD |
| `stage2_pct` / `stage3_pct` | 29.38% / 21.52% | 29.38% / 21.52% | **B4** — unchanged, which *confirms* the relative SICR trigger was inert |
| `method_chosen` | `"isotonic"` | `"none"` | A19 |
| `total_el`, scorecard AUC/Gini/KS/PSI | — | unchanged | no PD-model change was made |

`stage2_pct` holding exactly constant after the relative SICR trigger was removed is the
cleanest confirmation of finding B4: a trigger that changes nothing when deleted was never
firing.

### The champion–challenger conclusion reverses

Restoring feature parity (B2) changed the headline model-selection finding. On an identical
15-predictor set, **every tree model now beats the scorecard out-of-time**:

| Model | OOT AUC (13 feats, before) | OOT AUC (15 feats, after) |
|---|---:|---:|
| Logistic Scorecard | 0.6989 | 0.6989 |
| Weighted Ensemble | 0.6943 | **0.7030** |
| LightGBM | 0.6784 | **0.7027** |
| XGBoost | 0.6776 | **0.7020** |
| Random Forest | 0.6727 | **0.7006** |

The paired bootstrap on Gini (challenger − champion) flips sign from **−0.0410** to
**+0.0078**, and remains significant. The DeLong test agrees.

The shipped report's central modelling claim — that the interpretable WoE scorecard
generalises better than gradient boosting across the 2016–2018 regime shift — was an
artefact of the challengers missing the two strongest predictors. It was not a finding.

§3.6, §7.8 and the executive-summary "discrimination ceiling" paragraph are now generated
from the benchmark table: the report states plainly that the challengers win on
discrimination, quantifies the gap (0.0041 AUC), and retains the scorecard as champion on
explicit *governance* grounds — points attribution for adverse-action reasons, a monotone
auditable form — rather than on a discrimination claim it does not support. If the ranking
reverses again on a future run, the prose follows the data automatically.

### Second-order effect on C1

The Stage-3 share of ECL implied by the PD sensitivity moves from ~93% to ~86%. The PD
shock now propagates further because the shorter, correct horizon leaves cumulative PD
below the `_cap_cumulative_pd` ceiling that previously absorbed part of the stress. Stage 3
still dominates, and the C1 conclusion is unchanged — it is simply measured more accurately.

### Build state

- `pytest tests/` — 247 passed (including 11 new regression tests).
- `reports/qa_checks.py` — all checks pass, 85 metric keys audited.
- `xelatex` — 4-pass build, **0 LaTeX errors**, 0 undefined references or citations, 39 pages.
