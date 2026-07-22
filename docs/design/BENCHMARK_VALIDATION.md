# Benchmark Validation — How It Works

## Document Control
- **Data Scope:** Lending Club 2007–2018 portfolio (accepted + rejected applications)
- **Target System:** Retail Credit Risk Engine (PD, LGD, EAD, ECL, Basel III RWA)
- **Authoritative values:** the compiled report `reports/model_risk_report.pdf`
  (Tables 13 & 18), rendered live from `outputs/metrics.json`.

> **Important.** This document does **not** hand-maintain benchmark numbers. An earlier
> version of this file cached a static results table whose figures (e.g. LGD ≈ 0.87, ECL
> coverage ≈ 30%) predated the LGD EAD-proxy correction and no longer matched the engine
> (current mean LGD ≈ 0.30, coverage flagged below the published range). Hand-typed
> benchmark tables drift from the code and were removed. **Do not re-introduce them.**

---

## Single source of truth

All literature reference ranges live in **`reports/benchmarks.py`** — one `Benchmark`
object per metric, each carrying its published range, a citation (`source_bibkey`), and a
verifiable locator. At every report build, `reports/render_latex.py`:

1. reads the project's *computed* value from `outputs/metrics.json`;
2. renders the "Published Benchmark" cell **and** the pass/fail verdict from the *same*
   `Benchmark` object, so the range and the comparison can never disagree; and
3. is gated by `reports/qa_checks.py::check_benchmarks_sourced`, which fails the build if
   any benchmark row is unsourced or hand-typed, and by `check_no_fabricated_benchmark`,
   which fails the build if a static (fabricated) range reappears.

These are **comparisons against published reference ranges**, not a reproduction of the
cited studies' experiments. Verdicts of "Below/Above typical range" are reported honestly
and are expected for a high-risk, unsecured peer-to-peer book (e.g. IFRS 9 ECL coverage
sits below mixed-book survey ranges; LGD out-of-sample $R^2$ is low/negative because LGD
is strongly bimodal — see `\textcite{loterman2012benchmarking}`).

To read the current verdicts, open the compiled report or inspect the tokens produced by
`render_latex.py`; do not copy numbers here.

---

## Reference ranges and their sources

| Metric | Published range | Source (bibkey) |
| :--- | :---: | :--- |
| PD AUC (OOT) | 0.65 – 0.73 | `lessmann2015benchmarking` |
| PD / Scorecard Gini (OOT) | 0.30 – 0.45 | `lessmann2015benchmarking` |
| LightGBM Gini (OOT) | 0.35 – 0.55 | `lessmann2015benchmarking` |
| Mean LGD (unsecured retail) | 0.25 – 0.45 | `bellotti2012` |
| Downturn LGD | 0.25 – 0.45 (+ 90th-pct uplift) | `bellotti2012` |
| LGD $R^2$ (OOS) | 0.04 – 0.43 | `loterman2012benchmarking` |
| RWA density vs SA flat weight | 75% – 100% | `bcbs2017` |
| Reject-inference $\Delta$Gini | ≈ 0 (\|Δ\| ≤ 0.10) | `crook2004reject` |
| Score PSI | < 0.10 | `siddiqi2017` |
| IFRS 9 ECL coverage | 25% – 40% | `bcbs2021` |
| IFRS 9 Stage 2 % | 20% – 35% | `eba2022` |

Every bibkey above resolves in `reports/model_risk_report.bib` (enforced by the QA guard).
The Gini band is derived from the AUC band via $\text{Gini} = 2\,\text{AUC} - 1$, keeping
the two internally consistent (previously they disagreed across the two report tables).

---

*Benchmark verification is automated at report-build time. The report is the deliverable;
this file only documents the mechanism.*

