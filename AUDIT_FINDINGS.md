# Έλεγχος Codebase — IFRS 9 Credit-Risk & ECL Engine

**Ημερομηνία:** 2026-07-29 · **Commit:** `91f7d80` · **Branch:** `claude/codebase-structure-review-n7wr9u`

Ενδελεχής έλεγχος δομής και μεθοδολογίας, με εντοπισμό λαθών και ασυνεπειών μεταξύ
κώδικα, report και README.

**Πηγές τεκμηρίωσης.** Όλος ο κώδικας (`src/`, `reports/`, `scripts/`, `tests/`,
`config/`), το committed `reports/model_risk_report.tex` (rendered — περιέχει τα
πραγματικά νούμερα του τελευταίου full run), το committed
`outputs/scorecard_tables.json`, και τα committed PNG figures. Τα
`outputs/metrics.json` και τα `.parquet` είναι gitignored, οπότε κάθε αριθμητική
επαλήθευση παρακάτω προέρχεται από τα παραπάνω committed artifacts.

**Έκταση:** ~21.400 γραμμές — 49 modules, 27 test files, 1.434 γραμμές rendered
LaTeX, 3.303 γραμμές report renderer, 1.102 γραμμές QA suite.

**Σύνολο ευρημάτων: 58.**

| # | Κατηγορία | Πλήθος |
|---|---|---|
| [Α](#α-κρίσιμα) | Κρίσιμα — αλλοιώνουν δημοσιευμένα αποτελέσματα | 6 |
| [Β](#β-σοβαρά--αντιφάσεις-και-ανακριβείς-ισχυρισμοί-στο-report) | Αντιφάσεις / ανακριβείς ισχυρισμοί report | 8 |
| [Γ](#γ-μοντέλο-και-στατιστική) | Μοντέλο & στατιστική | 9 |
| [Δ](#δ-πληθυσμός-και-δεδομένα) | Πληθυσμός & δεδομένα | 6 |
| [Ε](#ε-νεκρός-κώδικας-και-ανενεργά-config-knobs) | Νεκρός κώδικας & ανενεργά config | 8 |
| [ΣΤ](#στ-ποιότητα-και-διαδικασία) | Ποιότητα & διαδικασία | 9 |
| [Ζ](#ζ-τρίτο-πέρασμα--reporting-layer-δεδομένα-test-suite) | Τρίτο πέρασμα — charts, EDA, synthetic, macro, tests | 12 |

**Κάλυψη.** Τα Α–ΣΤ προέκυψαν από δύο περάσματα που κάλυψαν γραμμή-γραμμή την αλυσίδα
μοντελοποίησης και δειγματοληπτικά τα τρία μεγαλύτερα αρχεία. Το [Ζ](#ζ-τρίτο-πέρασμα--reporting-layer-δεδομένα-test-suite)
είναι τρίτο πέρασμα στα αρχεία που είχαν μείνει εκτός: `reporting/charts.py`,
`data/eda.py`, `data/synthetic.py`, `data/download_macro.py`,
`validation/interpretability.py`, `validation/ab_test.py` και το test suite (282
tests). **Τίποτα δεν εκτελέστηκε** — ούτε pipeline, ούτε tests, ούτε build του report·
όλα είναι στατική ανάλυση διασταυρωμένη με τα committed artifacts.

---

## Α. ΚΡΙΣΙΜΑ

### A1 · Το cutoff/RAROC sweep χρεώνει Expected Loss = $0

**Αιτία.** `risk/ifrs9_ecl.py:337` εκτελεί `out["pd_12m"] = pd_12m` πάνω σε frame
(`df_rwa`) που **ήδη περιέχει** στήλη `pd_12m` παραγόμενη από το scorecard
(`pipeline.py:750`). Η στήλη αντικαθίσταται με το 12μηνο PD του hazard model, το
οποίο είναι δομικά ≈ 0 (βλ. A2). Στη συνέχεια:

```
pipeline.py:1187    df_ecl_copy = df_ecl.copy()             # φέρει το hazard pd_12m
pipeline.py:1215-8  pd_app = df_app["pd_12m"]               # ≈ 0
pipeline.py:1237    el = (pd_annual*lgd_app*ead_app).sum()  # ≈ 0
```

Το σχόλιο ακριβώς από πάνω (`pipeline.py:1229-1235`) δηλώνει ρητά ότι το `pd_12m`
«προέρχεται από το scorecard» — ίσχυε στη Φάση 5, δεν ισχύει πια στη Φάση 9.

**Απόδειξη** — `model_risk_report.tex:1255-1263`, στήλη *Expected Loss*:

| Cutoff | Approval | Bad Rate | Expected Profit | Expected Loss | RAROC |
|---:|---:|---:|---:|---:|---:|
| 500 | 95,6% | 20,16% | $283,7m | **$0** | 78,39% |
| **530** | **64,4%** | **14,22%** | **$98,5m** | **$0** | **58,80%** |
| 540 | 47,1% | 11,49% | $53,0m | **$0** | 50,05% |
| 580 | 3,5% | 3,39% | $1,1m | **$0** | 24,86% |

Επιβεβαιώνεται και οπτικά στο `reports/figures/cutoff_profit_curve.png`: η καμπύλη
Expected Profit είναι **μονότονα φθίνουσα** από 400 έως 610. Χωρίς πιστωτική ζημιά
κάθε επιπλέον εγκεκριμένο δάνειο προσθέτει καθαρό κέρδος, οπότε δεν υπάρχει κανένα
trade-off κινδύνου — η καμπύλη δεν έχει εσωτερικό μέγιστο, όπως θα είχε ένα
πραγματικό profit curve.

**Επίπτωση.** Ολόκληρο το Section 9 (operating cutoff 530, approval 64,4%, expected
profit $98,5m, RAROC 58,80%, το σχήμα, ο πίνακας και η γραμμή «Operating cut-off»
του README) παράγεται από οικονομικά **χωρίς αναμενόμενη ζημιά**. Το «risk-adjusted
return on capital» δεν είναι risk-adjusted. Δευτερεύουσα ασυνέπεια: ο παρονομαστής
(`capital_requirement_k`, `pipeline.py:1239`) προέρχεται από το *scorecard* PD μέσω
της Φάσης 7, οπότε αριθμητής και παρονομαστής του RAROC βασίζονται σε δύο
διαφορετικά μοντέλα PD.

**Κανένα QA check δεν το πιάνει.** Το `check_cutoff_optimum` (`qa_checks.py:84-126`)
ελέγχει μόνο ιχνηλασιμότητα γραμμής και το ceiling του bad rate· δεν κοιτάζει ποτέ
το `expected_loss`.

---

### A2 · Το Stage 1 ECL είναι δομικά μηδέν και δημοσιεύεται ως αποτέλεσμα

**Αιτία.** Ο `DiscreteHazardModel` χτίζει person-period panel στο οποίο **κάθε**
default event τοποθετείται στον τελευταίο μήνα του δανείου, χωρίς censoring:

```python
# models/pd_term_structure.py:169-171
y = np.zeros(len(rep_indices), dtype=int)
ends = np.cumsum(T_all) - 1
y[ends] = targets
```

Άρα το σωρευτικό hazard στους πρώτους 12 μήνες είναι ≈ 0 για κάθε δάνειο. Ο ίδιος ο
κώδικας το παραδέχεται ρητά (`pd_term_structure.py:270-275`: *«the 12-month
cumulative hazard is ~0 for every loan»*) και το `qa_checks.py:957-966` επίσης.

**Απόδειξη** — `model_risk_report.tex:1296-1298`:

| Stage | Loans | EAD | Mean PD (12m) | Mean PD (life) | ECL |
|---|---:|---:|---:|---:|---:|
| Stage 1 (12-month ECL) | 483.685 | $1,24bn | **0,00%** | 10,91% | **$0,00bn** |
| Stage 2 (lifetime ECL) | 295.312 | $1,31bn | **0,00%** | 24,09% | $0,17bn |
| Stage 3 (credit-impaired) | 213.664 | $1,17bn | **0,00%** | 23,32% | $1,04bn |

**Επίπτωση.** Το performing book — 49% των δανείων, $1,24bn έκθεση — φέρει
**μηδενική πρόβλεψη**. Ένα IFRS 9 engine που βγάζει μηδενικό Stage 1 ECL δεν εκτελεί
την κύρια λειτουργία του. Το Section 10.3 αποκαλύπτει τιμίως το hindsight του
Stage 3, αλλά **πουθενά** δεν δηλώνεται ότι το Stage 1 είναι δομικά μηδέν· ο πίνακας
παρουσιάζεται ως «reconciliation» ενώ μια ολόκληρη στήλη του είναι τεχνούργημα.

---

### A3 · Δύο αντικρουόμενες τιμές του «12-month PD» στο ίδιο report

- `model_risk_report.tex:597` (Section 5.1): *«a mean of 20.96% lifetime against
  **6.64%** over twelve months»* — από το scorecard, `pipeline.py:750`.
- `model_risk_report.tex:1296-1298` (Section 10.2): Mean PD (12m) = **0,00%** και
  στα τρία stages — από το hazard model, μετά την αντικατάσταση του A1.

Το ίδιο ονομαζόμενο μέγεθος εμφανίζεται με δύο τιμές που διαφέρουν κατά τάξη
μεγέθους, σε δύο πίνακες του ίδιου εγγράφου, χωρίς καμία επισήμανση ότι πρόκειται
για δύο διαφορετικούς estimators. Είναι η άμεση, ορατή συνέπεια των A1+A2.

---

### A4 · Λάθος πρόσημο στον τύπο των scorecard points — ο πίνακας δεν αθροίζει στο score

Ο κώδικας (`models/pd_scorecard.py:392` και `:445`), το docstring (`:15`) και το
report (`model_risk_report.tex:290`) χρησιμοποιούν όλα:

```
Points_j = ( −(WoE_j · β_j) + α/n ) · Factor + Offset/n
```

Ο καθιερωμένος τύπος (Siddiqi 2017 Ch.5 / Anderson 2007 Ch.5 — **που παρατίθενται ως
πηγή**) είναι:

```
Points_j = −( WoE_j · β_j + α/n ) · Factor + Offset/n
```

Το `α/n` φέρει λάθος πρόσημο. Άθροισμα points = `Factor·(α − Σ woe·β) + Offset`, ενώ
το `predict_score()` (`pd_scorecard.py:517`) επιστρέφει
`Factor·(−α − Σ woe·β) + Offset`. Σταθερή διαφορά = `2·α·Factor`.

**Αριθμητική επαλήθευση** από τα committed outputs (α = −1,61097, Factor = 28,8539,
17 features):

```
2·α·Factor            = −92,97 πόντοι
εύρος πίνακα points   : 367,6 .. 521,1
εύρος score μοντέλου  : 460,6 .. 614,1
```

**Επίπτωση.** Ο Appendix A points table **δεν μπορεί να αναπαράγει** τον operating
cutoff 530: το μέγιστο δυνατό άθροισμα πόντων είναι 521. Ένας ελεγκτής που αθροίσει
τον δημοσιευμένο πίνακα θα συμπεράνει ότι κανένας αιτών δεν εγκρίνεται. Καταρρίπτεται
ακριβώς το επιχείρημα auditability / adverse-action codes που το report (`tex:411`)
επικαλείται ως λόγο διατήρησης του scorecard έναντι του LightGBM.

Δεν υπάρχει κανένα test που να συμφιλιώνει points ↔ score.

---

### A5 · Train/serve skew — interaction feature δεν ξαναφτιάχνεται στο scoring

`PDScorecard.fit` (`:208-211`) καλεί `_add_interaction_features` **και**
`_encode_categoricals`. Το `_woe_transform` (`:488-498`), που χρησιμοποιείται από
`predict_proba` και `predict_score`, καλεί **μόνο** `_encode_categoricals`.

Το `revol_util_x_new_acc` παράγεται αποκλειστικά από το `_add_interaction_features`
(`pd_scorecard.py:71`)· ο loader φτιάχνει το διαφορετικά ορισμένο
`revol_util_x_open_acc`. Είναι στα τελικά features (IV = 0,0969 — 8ο υψηλότερο,
β = −0,0487, p = 0,043).

Στο pipeline όλες οι κλήσεις scoring γίνονται με raw frames
(`predict_proba(df_train / df_test / df_oot / df_all / df_rej_aligned)`,
`predict_score(df_el / df_ecl_copy)`), οπότε:

```python
# pd_scorecard.py:497
X_fill = X[available].reindex(columns=woe_vars)   # → στήλη all-NaN
```

Όλες οι γραμμές πέφτουν στο Missing bin του optbinning, το οποίο είχε **0
παρατηρήσεις** στο fit (WoE = 0).

**Επίπτωση.** Το feature ουδετεροποιείται σιωπηλά: το deployed μοντέλο τρέχει σε 16
ενεργά features, ενώ το report δημοσιεύει 17 με συντελεστή, p-value και ladder πόντων
για ένα predictor που δεν φτάνει ποτέ στο scoring. Το `dti_fico_interaction` έχει το
ίδιο πρόβλημα αλλά κόπηκε από το ElasticNet, οπότε δεν εκδηλώνεται.

---

### A6 · Ψευδής ισχυρισμός ότι τα δύο unconstrained optima «coincide»

`model_risk_report.tex:1234`: *«the unconstrained profit-maximising **and**
RAROC-maximising cutoff **coincide** at the most exclusive non-empty cutoff on the
grid (score 610, approving only 0.001% of the population)»*.

Ο κώδικας υπολογίζει δύο διαφορετικές γραμμές:

- `pipeline.py:1283` `_argmax_row = max(_nonempty, key=expected_profit)` → `cutoff_profit_argmax`
- `pipeline.py:1285` `_raroc_row = raroc_argmax_cutoff(...)` → `cutoff_raroc_max`

Το `render_latex.py:1742` όμως χρησιμοποιεί **μόνο** το RAROC argmax:

```python
_corner_row = metrics.get("cutoff_raroc_max") or metrics.get("cutoff_profit_argmax", {})
```

και η λέξη «coincide» είναι hardcoded στο template. Δεν γίνεται ποτέ σύγκριση των δύο
γραμμών.

Από το `cutoff_profit_curve.png`: το expected profit είναι μονότονα φθίνον, με μέγιστο
≈ $345m στο cutoff 400 (έγκριση ~100%). Το RAROC argmax είναι στο 610 (έγκριση
0,001%). Τα δύο optima βρίσκονται στις **αντίθετες γωνίες** του grid.

---

## Β. ΣΟΒΑΡΑ — αντιφάσεις και ανακριβείς ισχυρισμοί στο report

### B1 · «All coefficients negative» ↔ θετικός συντελεστής 23 γραμμές παρακάτω

- `tex:353`: *«As expected ... **all** feature coefficients carry a **negative sign**»*
- `tex:376`: `mo_sin_rcnt_tl & 0.0182 & 0.0396 & 0.460 & 0.6452` — **θετικός** (και μη
  σημαντικός, p = 0,65)

**Αιτία.** Το sign check (`pd_scorecard.py:321-343`) τρέχει **μία φορά**: εντοπίζει
παραβάτες, τους αφαιρεί, κάνει refit — και δεν ξαναελέγχει. Το refit αλλάζει τους
συντελεστές των επιζώντων και μπορεί να γεννήσει νέες παραβάσεις, όπως εδώ.
Χρειάζεται loop μέχρι σύγκλισης.

Ίδιο θέμα και στο `tex:288`: *«all coefficients must be negative»* ως μεθοδολογική
αρχή που το ίδιο το παραδοτέο μοντέλο παραβιάζει.

### B2 · «Every cutoff returns a negative RAROC» ↔ «22 of 22 clear the hurdle»

Στην **ίδια ενότητα**:

- `tex:1234`: *«every cutoff on the 400–800 grid returns a **negative RAROC**»*
- `tex:1268`: *«**22 of the 22** non-empty cutoffs on the 400–800 grid **clear the
  15.00% hurdle**, the strongest being cutoff 610 at a RAROC of 185.33%»*
- Ο πίνακας δείχνει RAROC 78,39% / 58,80% / 50,05% / 24,86%.

**Αιτία.** `render_latex.py:1758-1766` — η φράση είναι hardcoded στο `else` branch που
επιλέγεται όταν `corner_approval < 0.99`, χωρίς κανέναν έλεγχο των προσήμων. Το
`__GRID_HURDLE_VERDICT__` (γραμμές 1713-1735) είναι data-driven και λέει το σωστό,
οπότε οι δύο προτάσεις της ίδιας παραγράφου παράγονται με ασύμβατη λογική.

### B3 · «Within a few percent» ↔ 4,3× διαφορά

`tex:1284`: *«two headline provisions that land **within a few percent** of one
another»*. Ο πίνακας από κάτω: EL = **$0,28bn**, ECL = **$1,21bn** → λόγος 4,3×.
Το ίδιο stale σχόλιο υπάρχει και στον κώδικα (`pipeline.py:1077-1079`: *«differ by
only ~3%»*). Η φράση είναι σταθερό κείμενο του template, όχι παραγόμενη.

### B4 · 58 hardcoded stale fallbacks στον report renderer

Το `render_latex.py` περιέχει **58** αναζητήσεις της μορφής
`metrics.get("<key>", <σταθερά>)` όπου η σταθερά είναι νούμερο από παλιό run:

| Κλειδί | Hardcoded default | Πραγματική τιμή |
|---|---:|---:|
| `mean_lgd` | 0,1178 | 0,893 |
| `gini_oot` | 0,2651 | 0,401 |
| `ks_oot` | 0,1984 | 0,289 |
| `total_ecl` | 2.428.522 | 1.210.000.000 |
| `total_el` | 2.474.806 | 279.690.000 |
| `total_rwa` | 67.238.352 | 6.030.000.000 |
| `stage3_pct` | 0,0635 | 0,215 |
| `optimal_cutoff_threshold` | 550,0 | 530 |
| `optimal_approval_rate` | 0,871 | 0,644 |

Χειρότερα, **το ίδιο κλειδί έχει διαφορετικό default σε διαφορετικά σημεία**:
`total_rwa` → `0`, `0.0`, `67238352`· `mean_lgd` → `0.1178`, `0.0`, `0`·
`total_rwa_sa` → `0`, `0.0`, `244894720`. Αν λείψει ένα κλειδί, δύο ενότητες του
report μπορούν να τυπώσουν **διαφορετικά** νούμερα για το ίδιο μέγεθος, σιωπηλά.

Αντιβαίνει ρητά τη δηλωμένη φιλοσοφία του ίδιου του repo (`reports/benchmarks.py:8-10`:
*«structurally preventing a re-introduction of a hand-typed/fabricated row»*).

### B5 · Τρεις διαφορετικές ζώνες ανοχής για το ίδιο vintage backtest

| Πηγή | Ζώνη «pass» |
|---|---|
| `tex:956` (εξίσωση + κείμενο) | 0,50 – 1,50 («50% tolerance band») |
| `tex:957` (επόμενη παράγραφος) | 0,80 – 1,25 |
| `validation/backtest.py:63` (production flag) | 0,80 – 1,20 (amber 0,60–1,50) |
| `render_latex.py:409` (δεύτερη υλοποίηση) | 0,80 – 1,25 |

Δύο ανεξάρτητες υλοποιήσεις του ίδιου flag με **διαφορετικά κατώφλια**, και τρεις
διαφορετικές περιγραφές στο κείμενο.

### B6 · Το vintage scope του recalibrator δεν δηλώνεται πουθενά

`pipeline.py:176` προσαρτά τον calibrator **μόνο** για vintages ≥ 2016:

```python
scorecard.set_calibrator(calibrator, min_issue_year=_oot_start_year)
```

Είναι σκόπιμη απόφαση, αλλά:

- `pipeline.py:177-179` την **καταγράφει μόνο στο log** — δεν γράφεται στο
  `metrics.json`, οπότε το report δεν μπορεί να την αναφέρει.
- `tex:906` δηλώνει ανεπιφύλακτα: *«The transform **is** attached to the production
  scorecard, so the PDs feeding Expected Loss, Basel RWA and IFRS 9 staging are
  recalibrated»* — χωρίς να πει ότι ~45% του χαρτοφυλακίου (vintages 2007–2014)
  παραμένει **μη** αναβαθμονομημένο.
- `tex:956` λέει ότι οι λόγοι του vintage backtest είναι *«post-recalibration, i.e.
  they describe the PDs the pipeline actually deploys»* — αληθές μόνο για τις γραμμές
  2016+.

Αποτέλεσμα: το ίδιο χαρτοφυλάκιο φέρει ασυνέχεια επιπέδου PD στο όριο του 2016, χωρίς
αποκάλυψη.

### B7 · README ≠ κώδικας για το SICR

README, *Methodology Notes*: *«3-stage model with SICR (**2.5× PD uplift + 30 DPD
backstop**)»*.

Πραγματικότητα:

- Το relative 2,5× trigger είναι **απενεργοποιημένο** — `pipeline.py:1066` περνάει
  `pd_orig_lifetime=None` (ορθώς, δεν υπάρχει origination snapshot).
- Ο 30-DPD backstop **δεν υλοποιείται**: `assign_stages` (`ifrs9_ecl.py:161-165`)
  χρησιμοποιεί `delinq_2yrs >= 1`, δηλαδή bureau flag για καθυστερήσεις **πριν** την
  εκταμίευση.
- Το absolute threshold 0,20 — ο πραγματικός driver του Stage 2 (29,7% του βιβλίου) —
  δεν αναφέρεται καθόλου στο README.

Το `model_risk_report.tex:736, 1324-1325` είναι απολύτως τίμιο εδώ. Το **README** είναι
που παραπλανά.

### B8 · Το ECL what-if και το tornado αγκυρώνονται σε Z = 0, που δεν είναι σενάριο

`ifrs9_ecl.py:523`: `base_ts = hazard_model.predict_term_structure(df, macro_shock=0.0)`.

Είναι **ακριβώς** το ελάττωμα που ο ίδιος ο κώδικας διόρθωσε για το staging
(`ifrs9_ecl.py:316-328`): η συνάρτηση conditional PD του Vasicek δεν επιστρέφει το
unconditional PD στο Z = 0, οπότε το baseline σενάριο κάθεται σε μη-μηδενικό Z.

- `reports/figures/ecl_tornado.png` επιγράφει τη γραμμή αναφοράς **«Z = 0.0
  (Baseline)» — $1.184,2M**.
- `tex:1028`: *«Baseline-only ECL ... excludes the Upside/Downside macro scenario
  shocks»* — αλλά δεν είναι ούτε το baseline σενάριο· είναι ένα τέταρτο,
  μη-τιμολογημένο σημείο.
- Το headline weighted ECL είναι $1.210M.

Επιπλέον, το tornado μετρά ευαισθησία ±8–13% πάνω σε βάση που περιέχει το
PD-ανεξάρτητο Stage 3 ($1,04bn). Η πραγματική ευαισθησία του PD-εξαρτώμενου μέρους
είναι ~7× μεγαλύτερη. Το report κάνει αυτή τη διόρθωση για το what-if (Section 10.3,
f ≈ 88%) αλλά **όχι** για το macro tornado της Section 6.

---

## Γ. Μοντέλο και στατιστική

### Γ1 · Το «DeLong test» αγνοεί τη συνδιακύμανση των δύο AUC

`validation/discrimination.py:332-354`:

```python
se = float(np.sqrt(_variance(auc_a) + _variance(auc_b)))
```

Χρησιμοποιείται η προσέγγιση διακύμανσης Hanley–McNeil και τα δύο AUC
αντιμετωπίζονται ως **ανεξάρτητα**. Τα δύο μοντέλα σκοράρονται στα **ίδια** δάνεια,
οπότε η συσχέτισή τους είναι μεγάλη και θετική· η ουσία του DeLong (1988) είναι
ακριβώς ο όρος συνδιακύμανσης που λείπει.

Το SE υπερεκτιμάται σοβαρά, το z συρρικνώνεται, το τεστ είναι πολύ συντηρητικό. Το
report το παρουσιάζει ως «DeLong test» (Section 7.8) και βγάζει συμπέρασμα
μη-σημαντικότητας από αυτό.

### Γ2 · Cox χωρίς τυποποίηση + ridge penalty → μη συγκρίσιμα hazard ratios

`models/survival.py:127-128`: `CoxPHFitter(penalizer=0.1)` σε μη τυποποιημένα
covariates με εντελώς διαφορετικές κλίμακες:

| Covariate | Εύρος | Coef | HR (`tex:483-486`) |
|---|---|---:|---:|
| `grade_num` | 1 – 7 | +0,12773 | 1,13625 |
| `int_rate` | **0,05 – 0,35** (κλάσμα) | +0,03797 | 1,03870 |
| `dti` | 0 – 40 | +0,01317 | 1,01325 |
| `term_num` | 36 – 60 | +0,00314 | 1,00314 |

Η L2 ποινή εφαρμόζεται στους ακατέργαστους συντελεστές, οπότε μια μεταβλητή μικρής
κλίμακας (που χρειάζεται μεγάλο β για δεδομένη επίδραση) τιμωρείται δυσανάλογα. Το
`int_rate` — σε κλίμακα κλάσματος — συνθλίβεται σχεδόν στο μηδέν.

Επίδραση σε όλο το πραγματικό εύρος κάθε μεταβλητής:

- grade: 6 × 0,1277 = **+77%** hazard
- int_rate: 0,30 × 0,0380 = **+1,1%** hazard

Το report (`tex:472`) γράφει: *«The dominant hazard multipliers attach to credit grade
**and interest rate**»*. Το επιτόκιο έχει ουσιαστικά μηδενική επίδραση στο fitted
μοντέλο, και τα HR δεν είναι συγκρίσιμα μεταξύ τους ούτως ή άλλως.

### Γ3 · Το bug «fillna(12.0) σε κλάσμα» επιβιώνει στο `survival.py`

Το `pd_term_structure.py` διορθώθηκε ώστε να κανονικοποιεί πριν το fillna. Το
`survival.py:73` **όχι**:

```python
out["int_rate"] = pd.to_numeric(df.get("int_rate", 12.0), errors="coerce").fillna(12.0)
```

Το `full_accepted` φέρει `int_rate` ως **κλάσμα** (`loader.py:102-106` διαιρεί δια
100), άρα κάθε ελλείπουσα τιμή γίνεται 12,0 — δηλαδή 1200% — μέσα στο Cox.

### Γ4 · Το gains chart είναι αντεστραμμένο

`compute_decile_table(..., score_is_pd=True)` ταξινομεί **αύξουσα ως προς PD**, οπότε
το decile 1 είναι το **λιγότερο** επικίνδυνο. Το `plot_gains_chart`
(`discrimination.py:211-254`) χτίζει τη σωρευτική καμπύλη από εκεί.

`reports/figures/validation/gains_chart.png`: η καμπύλη «Model» τρέχει **κάτω** από τη
διαγώνιο «Random» σε όλο το εύρος (6,6% των bads στο 20% του πληθυσμού). Οπτικά
διαβάζεται ως μοντέλο χειρότερο του τυχαίου. Το συμβατικό cumulative gains/lift chart
ταξινομεί **υψηλότερο κίνδυνο πρώτα** και η καμπύλη είναι πάνω από τη διαγώνιο.

Ομοίως η στήλη `lift` του decile table είναι αντεστραμμένη ως προς τη σύμβαση.

### Γ5 · Sign check ενός περάσματος

Βλ. B1. Ο έλεγχος προσήμου δεν επαναλαμβάνεται μετά το refit, οπότε το τελικό μοντέλο
μπορεί να — και εδώ όντως — παραβιάζει τον περιορισμό που το report διαφημίζει ως
μεθοδολογική εγγύηση.

### Γ6 · Το `last_fico_range_*` λείπει από το leakage deny-list

Το `config/config.yaml:34-71` δεν περιέχει `last_fico_range_low` /
`last_fico_range_high` — το FICO του δανειολήπτη κατά την **τελευταία** άντληση
πιστωτικού αρχείου, δηλαδή καθαρά post-origination και η κλασικότερη διαρροή στα
δεδομένα LendingClub. Ούτε καλύπτονται από κάποιο prefix του deny-list (υπάρχει
`last_credit_pull_d`, όχι `last_fico`).

Τα `outputs/scorecard_tables.json` δείχνουν και τις δύο μεταβλητές στο `dropped_by_iv`
— δηλαδή **μπήκαν** στον υποψήφιο πληθυσμό και εξαιρέθηκαν μόνο από το φίλτρο IV
(σχεδόν βέβαια από το άνω όριο `max_iv = 0.50`, που είναι ρυθμιζόμενο config knob).

Το README δηλώνει: *«Post-origination features are excluded via an explicit
deny-list»*. Στην πράξη, γι' αυτές τις δύο, τις σώζει μια ευρετική IV — όχι η πολιτική.

### Γ7 · Downturn LGD = 1,000 — κορεσμένο στο άνω όριο

`metrics.downturn_lgd` = 1,0000, `mean_lgd` = 0,8931. Το p90 της πραγματοποιηθείσας
severity χτυπά το cap του `compute_realised_lgd` (`lgd.py:52` κάνει `.clip(0,1)`),
οπότε ολόκληρο το άνω decile βρίσκεται στο 1,0.

- Το κεφάλαιο Basel τρέχει με LGD = 100% (`pipeline.py:804, 807`) — το «downturn
  stress» δεν φέρει καμία διαφοροποιητική πληροφορία, είναι το μαθηματικό μέγιστο.
- Το `_downturn_from_realised` δίνει `max(p90, mean_pred)`, οπότε οποιαδήποτε τιμή του
  `lgd.downturn_percentile` ≥ ~60 πιθανότατα δίνει το ίδιο 1,0: το config knob είναι
  ουσιαστικά αδρανές.
- `tex:531` περιγράφει *«a buffer of 10,69 pp above the mean»* — ακριβές αριθμητικά,
  αλλά είναι απόσταση από το cap, όχι μέτρο ακραίας συνθήκης.

### Γ8 · Τρεις διαφορετικές πολιτικές ελλειπουσών τιμών στο «benchmark»

`models/pd_challenger.py`, `PDMultiModelBenchmark.fit`:

| Μοντέλο | Missing policy |
|---|---|
| LightGBM | native NaN (γραμμή 276) |
| XGBoost | `fillna(-9999)` (γραμμή 291) |
| Random Forest | `fillna(train medians)` (γραμμές 305-306) |

Τα τρία μοντέλα συγκρίνονται σε πίνακα («Multi-Model ML Benchmark») ως αν ήταν
like-for-like. Είναι η ίδια κατηγορία σφάλματος που το repo διόρθωσε ρητά για τον
binner (sentinel −9999) — εδώ επιβιώνει.

### Γ9 · Το «Weighted Ensemble» περιέχει το ίδιο το scorecard, με αυθαίρετα βάρη

`predict_proba_ensemble(..., w_scorecard=0.3, w_lgb=0.3, w_xgb=0.2, w_rf=0.2)` — βάρη
hardcoded, ποτέ βελτιστοποιημένα. Το report (`tex:418`) γράφει: *«on the out-of-time
set the best challenger, Weighted Ensemble, exceeds the scorecard's discrimination
(OOT AUC 0.7039 vs 0.7004)»*.

Δύο επιφυλάξεις που δεν αναφέρονται: (α) το «challenger» περιέχει 30% του ίδιου του
champion, άρα δεν είναι ανεξάρτητο μοντέλο· (β) το DeLong και το paired bootstrap
(`pipeline.py:433, 442`) τρέχουν έναντι του **LightGBM**, όχι έναντι του ensemble —
άρα η μοναδική διαφορά που το report ονομάζει «exceeds» δεν ελέγχεται στατιστικά
πουθενά.

---

## Δ. Πληθυσμός και δεδομένα

### Δ1 · Το «χαρτοφυλάκιο» είναι κατά ~80% in-sample

`pipeline.py:729`: `df_all = pd.concat([df_train, df_test, df_oot])`. Τα PD για
train+test (~55% των γραμμών) είναι in-sample. Όλα τα downstream — EL, RWA, Economic
Capital, concentration, IFRS 9 ECL, cutoff sweep — τρέχουν πάνω σε μείγμα in-sample και
out-of-sample προβλέψεων. Σε συνδυασμό με το B6, το ίδιο χαρτοφυλάκιο φέρει **δύο**
ασυνέχειες: in/out-of-sample και calibrated/raw, και οι δύο στο ίδιο όριο του 2016.

### Δ2 · Maturity / censoring bias στο OOT — 50% του OOT από 11 μήνες του 2016

Το `define_target` (`data/target.py:74`) κρατά **μόνο** resolved statuses. Δάνεια
2016–2018 που είναι ακόμη *Current* στο snapshot 2018Q4 αποκλείονται. Επειδή τα 60-μηνα
και τα υγιή δάνεια των πρόσφατων vintages δεν έχουν προλάβει να λήξουν, το OOT κρατά
δυσανάλογα όσα «τελείωσαν νωρίς» — δηλαδή τα charged-off.

**Ποσοτική απόδειξη** από το report (`tex:906`): ο χρονολογικός διαχωρισμός του OOT στο
50% πέφτει στο **2016-11**. Δηλαδή οι μισές OOT παρατηρήσεις (277.705) προέρχονται από
11 μήνες του 2016, και οι υπόλοιπες μισές (260.810) από τους επόμενους **25** μήνες —
σε περίοδο κατά την οποία ο όγκος χορηγήσεων της LendingClub **αυξανόταν**.

Το report καλύπτει το 2015 grey zone (`tex:195`) και το LGD censoring 2017–18
(`tex:751`), αλλά **δεν** αναφέρει αυτή την επιλογή δείγματος στον ίδιο τον πληθυσμό
PD, η οποία βαραίνει κάθε OOT μετρική (AUC, PSI, HL, recalibration gate).

### Δ3 · Ο ορισμός target αντιφάσκει με το «terminal resolved status»

Το bad set (`config.yaml:23-27`) περιλαμβάνει `Late (31-120 days)` — μη τερματική,
τρέχουσα κατάσταση καθυστέρησης.

- `model_risk_report.tex:184` το δηλώνει σωστά και το αιτιολογεί.
- README (*Methodology Notes*): *«the scorecard's target is the loan's terminal resolved
  status»* — ανακριβές. Ίδια διατύπωση και στα `risk/ifrs9_ecl.py:16-19`,
  `pd_term_structure.py`.

Δεν είναι μόνο ορολογικό: η μετατροπή lifetime → 12m (`pipeline.py:750`) υποθέτει ότι
το PD είναι lifetime, ενώ για ένα υποσύνολο του bad set είναι σημείο-στιγμιότυπο
καθυστέρησης.

### Δ4 · EAD βιβλιοποιείται σε δάνεια που δεν υπάρχουν

`reporting_date = 2018-12-31` και MOB = χρόνος από τη χορήγηση, capped στο term
(`models/ead.py:119-127`). Εφαρμόζεται **συμβατική** απόσβεση σε δάνεια που έχουν ήδη
αποπληρωθεί πλήρως ή χρεωθεί ως ζημιά χρόνια νωρίτερα — η πραγματική τους έκθεση στις
31/12/2018 είναι μηδέν.

Το README αναφέρει *«Zero-prepayment assumption (conservative)»*, που καλύπτει την
πρόωρη αποπληρωμή αλλά **όχι** το ότι σχηματίζεται πρόβλεψη πάνω σε εκθέσεις που έχουν
πάψει να υφίστανται. Το Stage 3 EAD ($1,17bn) αφορά εξ ορισμού δάνεια που έχουν ήδη
χρεωθεί ως ζημιά.

### Δ5 · Το TTC PD είναι μη σταθμισμένος μέσος τριμηνιαίων ποσοστών

`risk/pit_ttc.py:45`: `ttc_pd = float(np.mean(dr))` — απλός μέσος των τριμηνιαίων
default rates. Τα τρίμηνα 2007–2008 (λίγες εκατοντάδες δάνεια) ζυγίζουν όσο τα τρίμηνα
2014 (δεκάδες χιλιάδες).

Ασυνέπεια με το `fit_macro_model` (`ifrs9_ecl.py:718`), όπου το
`ttc_dr = df_train["target"].mean()` είναι **σταθμισμένο κατά δάνειο**. Δύο διαφορετικά
«TTC PD» στο ίδιο engine, και η OLS τρέχει σε μη σταθμισμένα τριμηνιαία σημεία ενώ
αγκυρώνεται σε σταθμισμένο μέσο.

### Δ6 · Όλο το macro apparatus τρέχει μόνο στο training window

`fit_macro_model(df_train, ...)` (`pipeline.py:968`), `build_quarterly_macro_frame(df_train, ...)`
(`macro_ts.py:31`, καλείται στα `pipeline.py:980, 994`). Άρα η macro παλινδρόμηση, τα
σενάρια, οι Vasicek Z-shocks, τα ADF/Granger/AIC/Johansen και η PiT/TTC αποσύνθεση
βασίζονται **αποκλειστικά** σε vintages 2007–2014 — τέσσερα χρόνια πριν την ημερομηνία
αναφοράς.

Δευτερεύον: το `build_quarterly_macro_frame` κάνει **σύγχρονο** merge, ενώ η production
`fit_macro_model` χρησιμοποιεί lag 2 τριμήνων. Το docstring του `macro_ts.py:33-35` λέει
ρητά ότι «mirrors the contemporaneous merge inside fit_macro_model» — δεν το κάνει πια,
οπότε το Granger test που υποτίθεται ότι τεκμηριώνει την επιλογή του lag τρέχει σε
διαφορετική σειρά από την παραγωγή.

---

## Ε. Νεκρός κώδικας και ανενεργά config knobs

Ρυθμίσεις που διαβάζονται, επικυρώνονται από το Pydantic, εμφανίζονται στο
`config.yaml` με επεξηγηματικό σχόλιο — και **δεν επηρεάζουν τίποτα**:

| # | Knob | Κατάσταση |
|---|---|---|
| E1 | `ifrs9.sicr_dpd_backstop: 30` | Περνά μέχρι `SICRConfig.dpd_backstop` (`pipeline.py:1046`) και **δεν διαβάζεται ποτέ** από την `assign_stages`. Η αλλαγή του δεν κάνει τίποτα. |
| E2 | `ifrs9.stage3_dpd: 90` | `utils/config.py:84`, μηδέν χρήσεις. |
| E3 | `business.interest_revenue_rate: 0.12` | Μηδέν χρήσεις (το sweep χρησιμοποιεί per-loan `int_rate`). |
| E4 | `business.cost_per_approval: 50.0` | Μηδέν χρήσεις. |
| E5 | `basel.maturity_adjustment: false` | Μηδέν χρήσεις. |
| E6 | `concentration.rho` | `risk/concentration.py:45` — δηλωμένο `# noqa: ARG001`· το pipeline του περνά `cfg.econ_cap.rho` και αγνοείται. |
| E7 | `ManualMonotonicBinner(min_bin_frac=...)` | `features/binning.py:132` — αποθηκεύεται, **ποτέ δεν επιβάλλεται** στο `_fit_one`. Τα bins μπορούν να είναι οσοδήποτε μικρά. |
| E8 | `get_binner(**_extra)` | `binning.py:268` — καταπίνει σιωπηλά άγνωστα kwargs. |

**Επιπλέον νεκρός κώδικας / διπλογραφία:**

- `PDScorecard.fit(X_train, y_train, X_test, y_test)` — τα δύο τελευταία ορίσματα
  μετασχηματίζονται (`:209`, `:211`) και **δεν χρησιμοποιούνται ποτέ**. Το API υπονοεί
  validation/early-stopping που δεν υπάρχει.
- `SurvivalPDModel.monthly_hazard_from_cox` — καλείται μόνο από test. Το module
  docstring όμως το διαφημίζει ως το κύριο παραδοτέο του Cox.
- `features/woe.compute_woe_iv` — νεκρό (τεκμηριωμένο).
- `business/cutoff.sweep_cutoffs` / `optimal_cutoff` / `run_cutoff_analysis` — με
  hardcoded `profit_good=0.05`, `loss_bad=0.45`· αποσυνδέθηκαν από το pipeline αλλά
  παραμένουν και τα tests τα ασκούν.
- `validation/backtest.score_band_stability_heatmap` — νεκρό (τεκμηριωμένο).
- `discrimination.pd_backtest_by_vintage` ↔ `backtest.vintage_pd_accuracy` — **δύο
  υλοποιήσεις του ίδιου πράγματος**· η πρώτη είναι νεκρή και το docstring της λέει
  λανθασμένα «predicted mean **12m** PD» ενώ η στήλη που τροφοδοτείται είναι lifetime.
- `risk/ifrs9_ecl.stage_migration_matrix` — διατηρημένο, μη καλούμενο (τεκμηριωμένο).

**Ασυνέπεια ορίζοντα PD:** `pipeline.py:889` καλεί `run_concentration(df_rwa, ...)` με
το default `pd_col="pd_pred"` = **lifetime** PD, ενώ όλο το υπόλοιπο Basel/EL/EC κομμάτι
χρησιμοποιεί `pd_12m`. Το granularity adjustment υπολογίζεται σε λάθος ορίζοντα σε
σχέση με το κεφάλαιο στο οποίο προστίθεται.

---

## ΣΤ. Ποιότητα και διαδικασία

### ΣΤ1 · Επινοημένα νούμερα σε fallback

`pipeline.py:1436-1440`:

```python
except Exception as st_err:
    metrics["stress_el"]  = metrics.get("total_el", 0.0) * 1.5
    metrics["stress_rwa"] = metrics.get("total_rwa", 0.0) * 1.8
```

Αν το stress test αποτύχει, γράφονται **κατασκευασμένοι** πολλαπλασιαστές που ρέουν στο
report — και **περνούν** το `check_stress_direction` (`qa_checks.py:735`), αφού 1,8 > 1.

### ΣΤ2 · Το `metrics.json` γράφεται δύο φορές

`validation/report.py:216-219` γράφει το αρχείο με **μόνο** τα validation metrics· το
`pipeline.py:1448` το ξαναγράφει πλήρες. Κατάρρευση ενδιάμεσα (Φάσεις 4–9) αφήνει ένα
μερικό αρχείο που μοιάζει πλήρες, και το `make report` θα τρέξει πάνω του γεμίζοντας τα
κενά με τα hardcoded defaults του B4.

### ΣΤ3 · Ο recalibration gate ενεργοποιείται από subsample 5.000 γραμμών

`hosmer_lemeshow_test` (`calibration.py:42-49`) υποδειγματοληπτεί σε 5.000 γραμμές με
σταθερό seed 42. Το trigger του gate (`select_oot_recalibrator:315`) εξαρτάται από αυτό
το p-value — δηλαδή από δείγμα 5k σε slice 277.705 γραμμών. Η απόφαση προσάρτησης
calibrator (που επηρεάζει EL, RWA, staging) κρέμεται από μια υποδειγματοληψία.

### ΣΤ4 · Economic capital: σταθμισμένο PD αλλά μη σταθμισμένο EAD ανά bucket

`economic_capital.py:71-73`:

```python
pd_b.append(np.average(pd_s[lo:hi], weights=w))    # EAD-weighted
lgd_b.append(np.average(lgd_s[lo:hi], weights=w))  # EAD-weighted
ead_b.append(ead_s[lo:hi].mean())                  # UNweighted mean
```

Τα buckets ταξινομούνται **μόνο κατά PD**, οπότε η διασπορά EAD εντός bucket είναι
μεγάλη και αγνοείται πλήρως: η προσομοίωση χρησιμοποιεί `defaults × mean_ead`. Αυτό
υποεκτιμά την ουρά (VaR 99,9%, ES) — ακριβώς το μέγεθος που το κεφάλαιο υποτίθεται ότι
καλύπτει. Το docstring λέει «EAD-weighted, PD-ranked homogeneous buckets», που ισχύει
μόνο για τα δύο από τα τρία μεγέθη.

### ΣΤ5 · Per-frame z-scoring στα interaction features

`_add_interaction_features` (`pd_scorecard.py:56-58`) υπολογίζει z-scores με
`median`/`std` **του ίδιου του frame**. Στο pipeline καλείται ξεχωριστά σε
`df_train_ch`, `df_test_ch`, `df_oot_ch` (`:390-392`), οπότε ο challenger βλέπει
**διαφορετικό μετασχηματισμό ανά partition** — data-dependent preprocessing που ακυρώνει
μερικώς την out-of-time φύση του OOT.

### ΣΤ6 · 156 παραπομπές σε αρχεία που δεν υπάρχουν στο repo

`grep` για `docs/AUDIT.md` και `Flaws.md`: **156 αναφορές** σε 28 αρχεία (`src/`,
`tests/`, `reports/`, `config/`, `scripts/`). Και τα δύο είναι ρητά gitignored
(`.gitignore`: *«Internal design/audit notes - not published»*). Κάθε σχόλιο τύπου
«(Flaws.md finding N16)» είναι ανεπίλυτη παραπομπή για όποιον διαβάσει το δημοσιευμένο
repo.

### ΣΤ7 · Το `sign_check` αντιφάσκει με τον εαυτό του· τα tests ελέγχουν τη λάθος σύμβαση

- `features/selection.py:6` (module docstring): *«positive coefficients on WoE features»*
- `features/selection.py:159-162` (docstring): *«all coefficients should be positive»*,
  default `expected_positive=True`
- `pd_scorecard.py:329` (παραγωγή): `sign_check(coefs, expected_positive=False)` — η
  **αντίθετη** σύμβαση
- `tests/test_features.py:81-92`: ελέγχουν τη σύμβαση του **docstring**, όχι της
  παραγωγής.

Το production path δεν καλύπτεται από test.

### ΣΤ8 · Δεν υπάρχει CI· το badge «tests passing» είναι διακοσμητικό

Δεν υπάρχει κατάλογος `.github/`. Το `README.md:7`
(`![Tests](https://img.shields.io/badge/tests-passing-brightgreen)`) είναι στατικό
shields.io badge, χωρίς καμία σύνδεση με πραγματική εκτέλεση.

Στην ίδια κατηγορία:

- `.pre-commit-config.yaml` τρέχει **και** `ruff-format` **και** `black` — δύο
  ανταγωνιστικούς formatters στην ίδια αλυσίδα.
- `requirements.txt` πινάρει `requests` και `python-dotenv`, που δεν δηλώνονται στο
  `pyproject.toml`.
- Τα lint/test tools (ruff, black, mypy, pytest) βρίσκονται στα **runtime**
  dependencies του `pyproject`, ενώ το `[dev]` extra περιέχει μόνο
  ipykernel/jupyterlab.

### ΣΤ9 · Τα test fixtures δεν αναπαράγουν το σχήμα της παραγωγής

Το `conftest.small_accepted` δεν περιέχει `total_pymnt` ούτε `total_rec_prncp`:

- `compute_realised_lgd` τρέχει με `trp = 0` — ο principal-basis παρονομαστής που
  αποτελεί το κύριο σημείο του module δεν ελέγχεται ποτέ.
- `estimate_months_on_book` χωρίς reporting_date και χωρίς `total_pymnt` πέφτει στο
  fallback `term * 0.4` — τα LGD/EAD/survival tests ασκούν το **fallback branch**, όχι
  το production branch.

---

## Ζ. Τρίτο πέρασμα — reporting layer, δεδομένα, test suite

Τα δύο πρώτα περάσματα κάλυψαν γραμμή-γραμμή την αλυσίδα μοντελοποίησης, αλλά
δειγμάτισαν μόνο τα `reports/render_latex.py`, `reports/qa_checks.py` και το `.tex`,
και δεν άνοιξαν καθόλου τα `reporting/charts.py`, `data/eda.py`, `data/synthetic.py`,
`data/download_macro.py`, `validation/interpretability.py`, `validation/ab_test.py`
και τα 26 test files. Αυτή η ενότητα καλύπτει αυτό το κενό.

### Ζ1 · Το chart συγκέντρωσης δηλώνει «% of Portfolio» ενώ δείχνει «% of top-15»

`reporting/charts.py:338-343`:

```python
s = s.nlargest(top_n) if len(s) > top_n else s
pct = (s / s.sum() * 100.0).sort_values(ascending=True)
...
ax.set_xlabel("% of Portfolio Exposure", fontsize=10)
```

Το `s.sum()` υπολογίζεται **μετά** το `nlargest(top_n)`. Ο παρονομαστής είναι το
άθροισμα των top-15 κατηγοριών, όχι του χαρτοφυλακίου. Η `grouped_exposures`
(`risk/concentration.py:96-99`) επιστρέφει **όλες** τις κατηγορίες, οπότε για το
`addr_state` (50+ πολιτείες) η περικοπή ενεργοποιείται πάντα.

**Επίπτωση.** Κάθε μπάρα στο `reports/figures/concentration_risk.png` είναι
διογκωμένη κατά τον λόγο «σύνολο χαρτοφυλακίου / σύνολο top-15». Ο άξονας λέει ρητά
«% of Portfolio Exposure». Το HHI στο `metrics["concentration"]` υπολογίζεται σωστά
από την πλήρη σειρά — άρα το **σχήμα αντιφάσκει με τον πίνακα** δίπλα του.

### Ζ2 · Το macro fallback δεν είναι ισοδύναμο με το live FRED — άλλο μέγεθος, ίδιο όνομα στήλης

Το report δηλώνει (`model_risk_report.tex:779`): *«sourced **live** from the official
FRED (St.\ Louis Fed) API»*. Το `data/download_macro.py:194-202` όμως έχει σιωπηλό
fallback σε hardcoded πίνακα (`_REAL_HISTORY`, γραμμές 32-119) που ενεργοποιείται με
γυμνό `except Exception` — και **χωρίς `FRED_API_KEY` είναι η προεπιλεγμένη διαδρομή**
(το repo έχει μόνο `.env.example`).

Τα δύο μονοπάτια δεν παράγουν το ίδιο πράγμα. Σύγκριση του committed
`data/processed/macro_quarterly.csv` (live) με τον hardcoded πίνακα:

| Τρίμηνο | GDP_growth live | GDP_growth offline | CPI live | CPI offline |
|---|---:|---:|---:|---:|
| 2007Q1 | 1,25 | 0,30 | 0,98 | 0,60 |
| 2007Q2 | 1,31 | 0,80 | 1,13 | 0,70 |
| 2008Q3 | 0,22 | −0,50 | 1,54 | −0,20 |
| 2009Q1 | −1,21 | −1,10 | −0,69 | 0,30 |

Το live μονοπάτι τραβά τη σειρά FRED **`GDP`** (*ονομαστικό* ΑΕΠ) και παίρνει QoQ
μεταβολή· ο offline πίνακας περιέχει τιμές συμβατές με **πραγματικό** ΑΕΠ —
συστηματικά ~0,7 μονάδες/τρίμηνο χαμηλότερα, δηλαδή ακριβώς ο πληθωρισμός. Ίδιο
όνομα στήλης, διαφορετικό μέγεθος, καμία προειδοποίηση στο report.

**Επίπτωση.** (α) Όποιος κλωνοποιήσει το repo χωρίς κλειδί FRED προσαρμόζει το macro
OLS σε **άλλα δεδομένα** από αυτά του report — οι ελαστικότητες, τα ADF/Granger/Johansen
και τα ECL ανά σενάριο δεν αναπαράγονται. (β) Ακόμη και στο live μονοπάτι, το
ονομαστικό ΑΕΠ μαζί με το `CPI_inflation` στην ίδια παλινδρόμηση εισάγει τον
πληθωρισμό δύο φορές. Το `_REAL_HISTORY` περιγράφεται ως *«high-fidelity»* — για δύο
από τις πέντε σειρές δεν είναι.

*Θετικό:* οι τιμές `UNRATE` και `FEDFUNDS` του offline πίνακα ελέγχθηκαν και είναι
πράγματι σωστές ιστορικά (π.χ. 2009Q1 = 8,3· 2010Q1 = 9,8).

### Ζ3 · Ο QA έλεγχος ορίζοντα PD είναι μονόπλευρος — δεν μπορεί να πιάσει το A2

`qa_checks.py:726-732`:

```python
_check(
    float(mean_pd) <= 0.15,
    "...a lifetime PD is being used where a 12-month PD is required...",
    failures,
)
```

Ο έλεγχος έχει **μόνο άνω φράγμα**. Το A2 είναι η ακριβώς αντίθετη αστοχία: μέσο
12μηνο PD ≈ 0,0000. Περνά θριαμβευτικά. Ο ίδιος ο έλεγχος που γράφτηκε για να
φυλάει τον ορίζοντα PD είναι τυφλός στο πιο σοβαρό εύρημα του engine.

Χρειάζεται και κάτω φράγμα (π.χ. `mean_pd >= 0.002`) — ένα χαρτοφυλάκιο καταναλωτικής
πίστης με 20,96% lifetime PD δεν μπορεί να έχει 12μηνο PD μηδέν.

**Διόρθωση προηγούμενης εκτίμησης:** το `qa_checks.py` ορίζει **37** ελέγχους (όχι 44 —
οι υπόλοιποι είναι helpers/runners) και **όλοι είναι συνδεδεμένοι** στους
`run_metric_checks`/`run_tex_checks`. Δεν υπάρχουν νεκροί έλεγχοι. Το πρόβλημα δεν
είναι η κάλυψη, είναι η *κατεύθυνση* του Ζ3 και η απουσία ελέγχου στο `expected_loss`
(A1).

Δύο έλεγχοι μπορούν να φιμωθούν με σημαία στο `metrics.json`
(`allow_fallback_binner` :591, `vintage_calibration_disclosed` :771). Καμία από τις δύο
δεν τίθεται από το pipeline — είναι χειροκίνητες παρακάμψεις, αλλά τίποτα δεν
καταγράφει ποιος τις έθεσε ή γιατί.

### Ζ4 · 156 γραμμές νεκρού κώδικα με ισχυρισμό κανονιστικής αναγκαιότητας

Το `validation/interpretability.py` δηλώνει στο docstring του: *«PDP/ICE show how a
feature moves the prediction — **essential for regulatory model documentation**»*.

Δεν καλείται από πουθενά. Το `pipeline.py:596-598` το λέει ρητά: *«PDP and ICE plots
used to be produced here... Neither figure was ever referenced by the report»*. Τα
`reports/figures/validation/pdp_grid.png` και `ice_plot.png` **δεν υπάρχουν** και δεν
αναφέρονται στο `.tex`.

Ίδια εικόνα για το `charts.plot_shap_comparison` (`pipeline.py:587-592`): νεκρό, το
`shap_comparison.png` δεν υπάρχει στο `reports/figures/validation/`.

Είναι θεμιτό να κρατά κανείς εργαλεία για ad-hoc χρήση. Δεν είναι θεμιτό να δηλώνει
ένα docstring ότι κάτι είναι «essential for regulatory model documentation» όταν το
regulatory documentation δεν το περιέχει.

### Ζ5 · Τα chart tests ελέγχουν μόνο ότι δημιουργήθηκε αρχείο

Και τα 8 tests στο `tests/test_charts_smoke.py` έχουν ακριβώς μία δήλωση:

```python
charts.plot_concentration(grouped, tmp_path)
assert (tmp_path / "concentration_risk.png").exists()
```

Καμία δήλωση για περιεχόμενο. Συνέπειες:

- Το `test_plot_concentration` (`:49-55`) περνά **3 και 2 κατηγορίες** — κάτω από το
  `top_n=15`. Ο κλάδος περικοπής, δηλαδή το ίδιο το bug Ζ1, **δεν εκτελείται ποτέ**.
- Το `plot_ecl_tornado` και το `plot_cutoff_profit` **δεν έχουν κανένα test**. Είναι
  τα δύο γραφήματα που στηρίζουν το Section 9 και το macro sensitivity — δηλαδή τα A1
  και B8.
- Υπάρχει test για το `plot_shap_comparison`, που είναι νεκρός κώδικας (Ζ4). Το suite
  ελέγχει ό,τι δεν τρέχει και αφήνει αδοκίμαστο ό,τι παράγει τα δημοσιευμένα σχήματα.

Το ίδιο μοτίβο και στο Γ4 (αντεστραμμένο gains chart): κανένα chart test δεν κοιτάζει
τιμές.

### Ζ6 · Ο synthetic generator δεν παράγει ποτέ ανεπίλυτο δάνειο — η λογική grey-zone είναι αδοκίμαστη

`data/synthetic.py:105`:

```python
loan_status = np.where(is_bad, "Charged Off", "Fully Paid")
```

Δύο μόνο τιμές. Τα πραγματικά δεδομένα Lending Club περιέχουν `Current`,
`Late (31-120 days)`, `In Grace Period`, `Default`, `Does not meet the credit policy...`
— και ολόκληρη η λογική του `data/target.py` υπάρχει για να τις χειριστεί.

**Επίπτωση.** Ο διαχωρισμός `n_accepted_file` / `n_resolved` / `n_modelling`
(`pipeline.py:62-73`), η εξαίρεση της grey zone, και το εύρημα Δ3 (αντίφαση ορισμού
target με το «terminal resolved status») **δεν μπορούν να ελεγχθούν από κανένα test**,
επειδή στα συνθετικά δεδομένα κάθε δάνειο είναι εξ ορισμού resolved.

### Ζ7 · Τρεις αποκλίσεις σχήματος synthetic ↔ παραγωγής που ακυρώνουν ελέγχους

1. **`int_rate` ως κλάσμα αντί ποσοστού.** `synthetic.py:50-53` παράγει `0,05–0,35`.
   Τα πραγματικά δεδομένα LC έχουν συμβολοσειρά τύπου `"13.56%"` — γι' αυτό ακριβώς
   υπάρχει η `normalize_int_rate_to_fraction` (`ifrs9_ecl.py`). Η συνάρτηση
   κανονικοποίησης **δεν δοκιμάζεται ποτέ με ρεαλιστική είσοδο**.
2. **Απουσία `last_fico_range_high/low`.** Δεν υπάρχουν στον generator, δεν υπάρχουν
   στο deny-list (Γ6), και δεν υπάρχει κανένα `usecols` allow-list πουθενά στο `src/`.
   Το κενό διαρροής του Γ6 είναι **δομικά μη ανιχνεύσιμο** από το test suite.
3. **Το `generate_rejected` παράγει `annual_inc`** (`:194`). Το πραγματικό αρχείο
   rejected της Lending Club δεν έχει στήλη εισοδήματος. Το reject inference
   δοκιμάζεται με χαρακτηριστικό που στην παραγωγή δεν υπάρχει.

Σημείωση: το docstring του module λέει *«Used for pytest fixtures and CI only»* — δεν
υπάρχει CI (ΣΤ8).

### Ζ8 · Το γράφημα missingness υπολογίζεται μετά το φίλτρο διαρροής, το report το λέει «raw»

Το `filter_origination_features` εφαρμόζεται στο `loader.py:242`, **πριν** τον
διαχωρισμό. Το `run_eda` καλείται στο `pipeline.py:74-75` πάνω στο `split`, δηλαδή σε
ήδη φιλτραρισμένα δεδομένα. Το `render_latex.py:2136` όμως περιγράφει το σχήμα ως
*«the missingness profile of the **raw** feature set»*.

Οι ~30 στήλες με τη μεγαλύτερη απουσία τιμών στο ακατέργαστο αρχείο LC είναι ακριβώς
οι post-origination (`hardship_*`, `settlement_*`, `mths_since_last_*`) — οι
περισσότερες έχουν ήδη αφαιρεθεί. Το δημοσιευμένο σχήμα δείχνει άλλο πληθυσμό στηλών
από αυτόν που περιγράφει το κείμενο.

### Ζ9 · Τα ιστογράμματα good/bad χρησιμοποιούν ανεξάρτητα bin edges

`data/eda.py:328-329`:

```python
ax.hist(good, bins=40, alpha=0.6, color=C_GREEN, label="Good", density=True)
ax.hist(bad,  bins=40, alpha=0.6, color=C_RED,   label="Bad",  density=True)
```

Το `bins=40` υπολογίζει **ξεχωριστά** άκρα για κάθε κλήση, από το εύρος της κάθε
υποομάδας. Οι δύο κατανομές σχεδιάζονται σε διαφορετικό πλέγμα και με `density=True`
σε διαφορετικό πλάτος bin — οπτικά δεν είναι συγκρίσιμες, που είναι ο μοναδικός
σκοπός του σχήματος («Feature Distributions: Good vs Bad»). Σωστό: κοινό
`bins=np.linspace(lo, hi, 41)` από τα συνδυασμένα δεδομένα.

### Ζ10 · Τρία ονόματα για τον ίδιο κανόνα επιλογής cutoff, τα δύο λάθος

Ο κανόνας που όντως εκτελείται είναι `risk_appetite_cutoff(...)` — το πιο inclusive
score του οποίου το approved bad rate μένει κάτω από το ceiling (`pipeline.py:1292`).
Η λεζάντα του πίνακα (`render_latex.py:608`) το περιγράφει **σωστά**. Αλλά:

- Το κλειδί στο `metrics.json` λέγεται **`cutoff_optimal_profit`** (`pipeline.py:1293`)
  — δεν είναι το βέλτιστο κέρδους.
- Το σχόλιο ακριβώς από πάνω (`render_latex.py:598-599`) λέει *«marginal RAROC-hurdle
  rule»* — δεν είναι.
- Το docstring του `plot_cutoff_profit` (`charts.py:104-106`) λέει *«the reconciled
  marginal-RAROC-hurdle optimum»* — δεν είναι.

Δευτερεύον: το fallback `or _argmax_row` (`pipeline.py:1292`). Αν ο κανόνας risk
appetite δεν επιστρέψει γραμμή, το σύστημα πέφτει στο argmax κέρδους — που με
Expected Loss = $0 (A1) είναι η γωνία «έγκριση όλων».

### Ζ11 · Literal backslash στον άξονα του shock tornado

`charts.py:296`: `ax.set_xlabel(r"Change in ECL vs Baseline (\$M)")`. Το `style.py`
δεν ενεργοποιεί `text.usetex`, οπότε το `\$` δεν είναι escape — αποδίδεται κατά
γράμμα. Όλα τα υπόλοιπα γραφήματα γράφουν σκέτο `($M)` (π.χ. `:135`, `:185`), που
επιβεβαιώνει ότι πρόκειται για αβλεψία μεταφοράς LaTeX σύνταξης σε matplotlib.

### Ζ12 · Μικρότερα

- `data/eda.py:52` — `import matplotlib.cm as cm` μέσα στη συνάρτηση, ενώ υπάρχει ήδη
  στη γραμμή 16.
- `validation/ab_test.py:76` — το `"gini_a"` παίρνει το ίδιο το `nan3`, ενώ τα άλλα δύο
  παίρνουν `dict(nan3)`. Αβλαβές σήμερα (τίποτα δεν το μεταλλάσσει), ασύμμετρο όμως.
- `validation/interpretability.py:84` — `feats = [...][:4]` περικόπτει σιωπηλά σε 4
  χαρακτηριστικά ανεξάρτητα από το τι ζητήθηκε.
- `data/eda.py:108` — το `axvspan(2007, 2009, label="GFC Stressed Period")` σκιάζει
  περίοδο στην οποία η Lending Club είχε ελάχιστο όγκο. Το report **δεν** ισχυρίζεται
  κάλυψη GFC (το «GFC-like» του `tex:1041` είναι υποθετικός πολλαπλασιαστής ×3,0),
  οπότε είναι θέμα σχήματος, όχι ψευδής ισχυρισμός.

### Διόρθωση στο A1

Η παραπομπή στο A1 γράφει `risk/ifrs9_ecl.py:337`. Η σωστή γραμμή είναι **336**.
Το εύρημα δεν αλλάζει.

---

## Τι είναι σωστό — για ισορροπία

- **Σύμβαση Vasicek Z συνεπής παντού** — `ifrs9_ecl`, `pd_term_structure`,
  `economic_capital` και το Phase 9c stress test χρησιμοποιούν όλα
  `Φ((Φ⁻¹(p) − √ρ·Z)/√(1−ρ))` με Z<0 = δυσμενές.
- **Σωστός τύπος IRB retail** (BCBS CRE31.15), σωστή correlation curve, PD floor,
  απουσία maturity adjustment, σύγκριση με SA.
- **Recalibration gate** με χρονολογικό διαχωρισμό, fit στο πρώτο μισό, αποδοχή μόνο επί
  αποδεδειγμένης βελτίωσης στο δεύτερο, επιλογή οικογένειας με out-of-fold Brier εντός
  του fit slice.
- **Term truncation** στο lifetime ECL (`term_horizon_mask`) και **cap του σωρευτικού
  PD** στο what-if.
- **Principal-basis realised LGD**, με ρητή απόρριψη του `total_pymnt`.
- **Disjoint select/report halves** στο LGD OOS — αποφυγή selection bias στην promotion.
- **Supervisory ρ στο EC** ώστε να είναι συγκρίσιμο με το RegCap, με flat-ρ sensitivity
  δίπλα.
- **Εκτενές QA suite** (44 checks) και **regression tests** για κάθε προηγούμενο finding
  — ο λόγος που τα Α1–Α6 γλίστρησαν είναι ακριβώς ότι κάθε επιμέρους αριθμός είναι
  *εσωτερικά* συνεπής.
- **Section 10** του report είναι εξαιρετικά τίμιο για το hindsight του Stage 3, το
  απενεργοποιημένο relative SICR trigger, τον proxy backstop και το TTC anchoring.

---

## Προτεινόμενο πλάνο διόρθωσης

### Κύμα 1 — Κρίσιμα (Α1–Α6)

1. `risk/ifrs9_ecl.py:336-337` — να μην πατιέται υπάρχουσα στήλη· γράψιμο σε
   `pd_12m_hazard` / `pd_lifetime_hazard`.
2. `pipeline.py:1215-1218` — ρητή στήλη scorecard με fail-fast αν λείπει.
3. Νέο QA check `check_cutoff_el_nonzero`: το άθροισμα `expected_loss` πάνω από τα μη
   κενά rows του `cutoff_strategy_table` > 0 και συμβατό με το `total_el`.
4. Νέο QA check `check_stage1_ecl_material`: Stage 1 ECL > 0 όταν Stage 1 EAD > 0.
5. Hazard model: είτε (α) censoring / κατανομή του event μέσα στη ζωή του δανείου ώστε
   το 12μηνο hazard να μην είναι μηδέν — η ουσιαστική λύση· είτε (β) ρητή δήλωση
   αδυναμίας στο report και αφαίρεση της μηδενικής στήλης από τον πίνακα reconciliation.
6. `pd_scorecard.py:392, 445` + docstring `:15` + `render_latex.py:2226` + footnote του
   `_scorecard_points_latex` — διόρθωση προσήμου σε `−(woe·β + α/n)·factor + offset/n`.
7. Νέο test `tests/test_scorecard_points_reconcile.py`: άθροισμα points επιλεγμένων bins
   == `predict_score()` εντός 1e-6.
8. `pd_scorecard._woe_transform` — κλήση `_add_interaction_features` πριν το
   `_encode_categoricals`· fail-fast όταν λείπει `woe_variable`. Test:
   `predict_proba(raw) == predict_proba(prepped)`.
9. `render_latex.py:1742` — σύγκριση `cutoff_profit_argmax` με `cutoff_raroc_max` και
   παραγωγή της λέξης «coincide» μόνο όταν όντως συμπίπτουν.

### Κύμα 2 — Αντιφάσεις report (Β1–Β8)

10. `pd_scorecard.py:321-346` — sign check σε loop μέχρι σύγκλισης.
11. `render_latex.py:1758-1766` — η φράση για αρνητικό RAROC να παράγεται από
    `max(raroc) < 0`. Νέο guard `check_raroc_narrative_matches_grid`.
12. Section 10 — η φράση «within a few percent» να παράγεται από τον λόγο EL/ECL. Ίδιο
    και για το stale σχόλιο `pipeline.py:1077-1079`.
13. `render_latex.py` — αντικατάσταση και των 58 hardcoded defaults με fail-fast ή ρητό
    `"n/a"`. Νέο guard: κανένα rendered νούμερο να μην προέρχεται από default.
14. Ενοποίηση της ζώνης ανοχής του vintage backtest σε **μία** πηγή αλήθειας (κατάργηση
    της δεύτερης υλοποίησης στο `render_latex.py:409`) και ευθυγράμμιση του κειμένου.
15. Εξαγωγή του `scorecard.calibration_scope` στο `metrics.json` και ρητή αναφορά του
    scope στο Section 7.2.
16. README — διόρθωση της γραμμής SICR και της φράσης «terminal resolved status».
17. `ifrs9_ecl.py:523` — αγκύρωση του what-if στο baseline scenario Z· διόρθωση της
    ετικέτας «Z = 0.0 (Baseline)» στο tornado.

### Κύμα 3 — Μοντέλο, δεδομένα, καθαριότητα (Γ, Δ, Ε, ΣΤ)

18. `delong_test` — προσθήκη του όρου συνδιακύμανσης (πλήρης DeLong) ή μετονομασία σε
    `hanley_mcneil_independent_test` με ρητή επιφύλαξη στο report.
19. `SurvivalPDModel` — τυποποίηση covariates πριν το Cox (ή `penalizer=0`) και
    κανονικοποίηση του `int_rate`· διόρθωση της πρότασης για «dominant multipliers».
20. `plot_gains_chart` / `compute_decile_table` — ταξινόμηση υψηλότερου κινδύνου πρώτα,
    σύμφωνα με τη συμβατική μορφή lift chart.
21. Προσθήκη `last_fico_range_low/high` στο leakage deny-list.
22. Ενοποίηση της πολιτικής missing values στο `PDMultiModelBenchmark`.
23. `run_concentration` με `pd_col="pd_12m"`.
24. Σύνδεση ή αφαίρεση των 8 ανενεργών config knobs· αφαίρεση των `X_test/y_test` από
    την `PDScorecard.fit`.
25. `pipeline.py:1436-1440` — κατάργηση των επινοημένων fallbacks.
26. Αντικατάσταση των 156 παραπομπών σε gitignored αρχεία με σταθερά IDs (π.χ.
    `AUDIT-B5`) ή δημοσίευση συνοπτικού `docs/FINDINGS.md`.
27. Section 10 — προσθήκη του maturity/censoring bias του OOT (Δ2), της φύσης του EAD σε
    resolved δάνεια (Δ4) και του training-only macro window (Δ6).
28. Προσθήκη GitHub Actions workflow (test + lint + QA)· αφαίρεση του διπλού formatter
    από το pre-commit.

---

## Επαλήθευση διορθώσεων

- `make test` — τα υπάρχοντα 200+ tests πράσινα, συν τα 3 νέα (points reconciliation,
  fit/serve parity, cutoff EL > 0).
- `python -m credit_risk.pipeline` με `data.source: synthetic` — γρήγορο end-to-end χωρίς
  Kaggle. Έλεγχος στο `outputs/metrics.json`:
  `cutoff_strategy_table[*].expected_loss > 0` και
  `ecl_reconciliation.ecl_by_stage.s1 > 0`.
- `python reports/qa_checks.py` — όλα τα checks + τα νέα guards.
- Χειροκίνητη επαλήθευση points: άθροισμα των `points` ενός δανείου από το
  `outputs/scorecard_tables.json` == `predict_score()` για το ίδιο δάνειο.
- Full run σε πραγματικά δεδομένα και σύγκριση headline νούμερων πριν/μετά. **Ο cutoff
  θα μετακινηθεί και το RAROC θα πέσει σημαντικά** μόλις το EL πάψει να είναι μηδέν —
  αυτό είναι το ζητούμενο αποτέλεσμα της διόρθωσης, όχι regression.

---

# Απάντηση — επαλήθευση και αποκατάσταση (2026-07-30)

Κάθε εύρημα ελέγχθηκε γραμμή-γραμμή στο **τρέχον working tree**, όχι στο commit `91f7d80`
στο οποίο γράφτηκε ο έλεγχος. Το tree είχε ήδη μια αδέσμευτη διόρθωση (hazard timing +
`pd_12m` shadowing), οπότε μέρος του ελέγχου ήταν ήδη παρωχημένο. Οι αριθμητικές
επαληθεύσεις έγιναν από `outputs/scorecard_tables.json` και το παραγόμενο `.tex`.

## Ήδη διορθωμένα πριν από αυτό το πέρασμα

A1, A2, A3 (μερικώς), A6, B2, B3, B7, E1–E5, και τμήμα του Ζ3 (νέοι έλεγχοι
`check_stage1_ecl_nonzero`, `check_hazard_pd12m_nondegenerate`, `check_cutoff_el_nonzero`).
Το Stage 1 ECL δεν είναι πλέον μηδέν ($0,03bn), το `pd_12m` δεν σκιάζεται, και η καμπύλη
cutoff χρεώνει πραγματική αναμενόμενη ζημιά.

## Ευρήματα που **δεν** ισχύουν ή είναι υπερβολικά

- **Δ1 «~80% in-sample»** — λάθος. Το OOT είναι 538.515 από 992.661 δάνεια (54%), άρα το
  in-sample μερίδιο είναι ~46%. Το ίδιο το κείμενο του ελέγχου λέει αλλού «~55%», που
  αντιφάσκει με την επικεφαλίδα του. Η ουσία (ανάμεικτα in/out-of-sample aggregates)
  ισχύει και **ήταν ήδη δηλωμένη** στους περιορισμούς της §10. Καμία αλλαγή κώδικα.
- **B4 «58 hardcoded fallbacks»** — υποεκτίμηση: 66 σημεία, 34 με μη μηδενική παλιά
  σταθερά, 9 κλειδιά με **διαφορετικές** σταθερές σε διαφορετικά σημεία. Η ουσία ισχύει.
- **ΣΤ6 «156 αναφορές»** — 162/163 στην πράξη. Η ουσία ισχύει.
- **Γ7 downturn LGD = 1,0** — ισχύει αριθμητικά αλλά **ήταν ήδη δηλωμένο** στους
  περιορισμούς («Downturn LGD sits at the distribution cap»). Παρέμεινε μόνο το αδρανές
  knob, που καλύφθηκε στη διαλογή των config knobs.

## Νέο εύρημα που ο έλεγχος δεν μπορούσε να δει

- **N1.** Το `render_latex.py:2645` (→ `.tex:792`) εξακολουθούσε να δηλώνει ότι το hazard
  panel «τοποθετεί κάθε default στον τελευταίο μήνα» — ψευδές μετά τη διόρθωση, και σε
  ευθεία αντίφαση με τη νέα δήλωση περιορισμού στο `.tex:1327`. Είχε ήδη τυπωθεί στο PDF.

## Τι διορθώθηκε τώρα

**Κύμα 1 — μοντέλο** (μετακινούν κάθε δημοσιευμένο νούμερο)
A4 πρόσημο points (`-(woe·β + α/n)·factor`), A5+ΣΤ5 train/serve skew με fitted z-score
moments, B1/Γ5/ΣΤ7 sign check σε loop μέχρι σύγκλισης, Γ3 `int_rate` σε κλίμακα κλάσματος,
Γ6 `last_fico_range_*` στο deny-list, Γ4 gains chart με σωστή ταξινόμηση, Γ8 native NaN στο
XGBoost, Γ9 paired bootstrap για το ensemble, concentration σε `pd_12m`.

**Κύμα 2 — report**
N1, B4 (accessor `_num` + δύο νέοι QA guards), B5 (μία πηγή αλήθειας για τη ζώνη ανοχής),
B6 (vintage scope του recalibrator στο `metrics.json` και στο κείμενο), B8 (ετικέτες
tornado + ευαισθησία στο PD-εξαρτώμενο μέρος), Γ1 (πλήρης DeLong με όρο συνδιακύμανσης),
Γ2 (τυποποίηση covariates στο Cox, HR ανά τυπική απόκλιση), Ζ1 (παρονομαστής πριν την
περικοπή), Ζ8, Ζ10, Ζ11.

**Κύμα 3 — αποκαλύψεις**: Δ2 (censoring bias στο OOT), Δ3, Δ4 (EAD σε ήδη λυμένα δάνεια),
Δ6 (macro μόνο στο training window + ευθυγράμμιση lag), Δ5 (σταθμισμένο TTC).

**Κύμα 4 — διαδικασία**: ΣΤ1 (κατάργηση επινοημένων stress fallbacks), ΣΤ2, ΣΤ3 (median
p-value σε 9 replicates· ένα draw έδινε p από 0,03 έως 0,72 στα ίδια δεδομένα), ΣΤ4
(bucketing σε PD **και** EAD), ΣΤ6 (163 αναφορές → σταθερά IDs + `docs/FINDINGS.md`), ΣΤ8
(GitHub Actions CI, ένας formatter, καθαρά dev extras), ΣΤ9/Ζ6/Ζ7 (fixtures και synthetic
generator με σχήμα παραγωγής), Ζ5 (chart tests που κοιτούν τιμές), Ζ4, Ζ2 (GDPC1 αντί
ονομαστικού GDP + καταγραφή προέλευσης), Ζ9, Ζ12, E6/E7/E8, υπόλοιπο Ζ3.

Regressions: `tests/test_audit_findings_2026_07.py` (11 tests) + επεκτάσεις στο
`tests/test_charts_smoke.py`.
