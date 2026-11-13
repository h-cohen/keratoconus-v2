# Explainability and evaluation study — design

Date: 2026-11-13. Status: draft for approval.

## Intent

The final model (rank blend of XGBoost, 5-seed LightGBM bag and elastic-net LR on blocks N + R, + H in the H arm;
`kcref/model.py`) is frozen. This study produces every evaluation and explainability result the paper can present,
and adds them to `publication_v2/report.html`, the backbone of the algorithmic part of the paper (new, higher-ranked
journal).

The central question a clinical reviewer must be able to answer after reading:

> Our previous paper used single-eye Pentacam numerics and reached AUC ≈ 0.62. The same kind of input here gives about
> the same. The final model reaches 0.790 (noH, first visit). **Which additional information explains the gap, and
> where on the cornea is it?**

Every figure serves that question, either by measuring the gap (evaluation) or by locating it (explainability).

## Goals

1. Standard ML evaluation figures: ROC curves; AUC per stage (bar chart, stages on the x axis, one panel per method),
   for both staging approaches A (Km F) and B (Rm B).
2. A feature-block ladder that measures how much each added block contributes, starting from the previous paper's
   input.
3. SHAP explanations of the final model, per feature and grouped into clinically meaningful families.
4. Spatial explanation of block R: which corneal regions of which map carry the signal, drawn on the cornea.
5. Three stand-out analyses (user approved "all three"):
   - S1 matched-pair cases,
   - S2 explanations by stage,
   - S3 robustness of the explanation across CV fits and anchoring to Belin ABCD.
6. Every analysis for **both arms**: noH (primary, headline) and H ("visit + fellow-eye treatment history",
   auxiliary), side by side. H never replaces noH as the headline.

## Non-goals

- No new model, no retuning, no change to the frozen blend or its features. This study explains the model; it does
  not improve it.
- No reporting of superseded experiments in the report (the report covers the chosen method only). The ladder rungs
  are ablations of the chosen method, not old experiments.
- No lockbox scoring.

## Anti-goals (failure even if it "works")

- Explanations computed in-sample on the frozen model and presented as if they were held-out. All quantitative
  SHAP is out-of-fold (see below); the frozen model is used only where stated.
- Cherry-picked cases. S1 pairs are chosen by a rule fixed before looking at their maps (below).
- Any patient-identifying content in a figure: names, ids, dates, original filenames, and the header text burned into
  the original Pentacam composite images. Maps are drawn only from the decoded numeric arrays
  (`kcref/data/decoded_hr.npy`), never from the source image files. Cases are labelled "Pair 1 / eye a" etc., not
  even by `pid`.
- Figures that look like default matplotlib. The user asked for figures that are "very very eye-attractive yet
  scientific publication ready, proper colour palettes and designs".

## Constraints (from CLAUDE.md, binding)

- Evaluation via `kcref/evaluate.py`: 5-fold StratifiedGroupKFold by patient × 5 repeats (seeds 2026–2030), 944
  scans; pooled AUC and first-visit AUC (523 eyes, 125 positive), each with patient-cluster bootstrap CI (1000).
- Every new CV run appends to `kcref/results/experiments.csv` (never edit or delete rows). Pre-declare in
  `kcref/PROTOCOL.md` before running; log results in `kcref/RESULTS.md`.
- Staging only via `kcref/staging.py` (A and B, side by side). No KMax, no Amsler–Krumeich.
- Privacy rules above; never send data to an external service.
- Git: author `Hadar Cohen <hal.nls@gmail.com>`, no co-author or session trailers. Do not edit `paper/`.
- Sub-agents on Sonnet (`model: "sonnet"`).
- Keep code minimal (user preference): extend `kcref/model.py` / add one `kcref/explain.py` and extend
  `publication_v2/build_report.py`; no forks of the pipeline.
- `shap` is not installed. Use native exact TreeSHAP (`XGBClassifier.get_booster().predict(..., pred_contribs=True)`,
  `LGBMClassifier.predict(..., pred_contrib=True)`) and exact linear SHAP for the LR (coef × (standardised x − mean of
  standardised training x)). No new dependency is needed. *(My call.)*

## Part 1 — Evaluation

### 1a. ROC (exists; restyle)

ROC curves, all scans and first visit, noH and H, plus age-only and the single-eye baseline (ladder rung L1) as
dashed references. AUC with 95% CI in the legend. A shaded bootstrap band around the noH curve is optional.
`publication_v2/figures/roc_final.png` exists; it gets restyled to the new figure style.

### 1b. AUC per stage

One figure per staging approach (A, B). Each figure has one panel per method. *(My call:* small multiples inside one
figure, sharing the y axis. This reads "1 plot per method" while staying a single journal figure.)

- Methods (panels): age only, L1 single-eye numerics, final noH, final H.
- x axis: stages 1–4; bars = first-visit AUC with 95% cluster-bootstrap CI error bars.
- Annotate n / positives under each bar. A stage with < 10 positives or < 10 negatives gets a hatched bar and no CI,
  with a note. Stage A4 has 5 scans by design.
- Horizontal reference line at 0.5.
- An all-scans version goes to the report appendix as a table (`tables/strata.csv` already exists) and is not drawn.

## Part 2 — Feature-block ladder (new CV runs)

Same blend, same training rows (cohort fold rows + extra exams, first-visit weight 2), same inner tuning, same
folds. Only the columns change. Each rung is run for both arms: the H variant adds block H to that rung.

| rung | columns (noH) | name logged |
|---|---|---|
| L0 | age only | existing `age only (same folds)` result; no new run |
| L1 | own eye: age + 54 numerics + 11 indices (the previous paper's information) | `E29_ladder_L1_{arm}` |
| L2 | L1 + fellow eye + own−fellow asymmetry + worse_* + has_fellow (= full block N) | `E29_ladder_L2_{arm}` |
| L3 | L2 + block R + image flags (= final model) | existing `final_{arm}` / `E28_{arm}_blend`; no new run |

Four new CV runs: L1 and L2, each × noH and H.

- An optional rung L2b (L1 + R own-eye only, without fellow) separates "images" from "fellow eye". *(My call: include
  it, two more runs, because it answers "is it the maps or the other eye?" directly.)*
- Output: Fig. E1, a horizontal dot-and-whisker ladder (first-visit AUC [CI]). Each step is annotated with the paired
  Δ vs the previous rung (`evaluate.paired_delta`, same bootstrap), and noH and H are drawn as two coloured tracks.
- Implementation: add a `blocks` argument to `model.fit` / `F.assemble` (column subset) and a CLI form
  `model.py cv {noH|H} --blocks L1|L2|L2b`. The default stays the final model, and its output must be bit-identical to
  before (verify on fold 1, as done in the consolidation).

## Part 3 — SHAP for the final model

### Computation (out-of-fold, both arms)

- Re-run the final CV (25 fits per arm). For each fit, compute exact SHAP on that fit's test rows for each member, in
  the member's log-odds space (LR: decision function).
- The blend is a mean of ranks, which is not additive. Blend attribution, *(my call)*:
  - standardise each member's SHAP by the SD of that member's log-odds over the test fold;
  - average the three members with equal weights, matching the blend's equal-weight rank averaging;
  - LightGBM bag = mean over its 5 seeds.
- Report the per-member SHAP agreement (Spearman of mean |SHAP| across features) as a sanity check.
- Each scan appears in 5 test folds (one per repeat); average its SHAP over the repeats. Store
  `kcref/results/shap/{arm}_oof.npz`: SHAP matrix (944 × p), feature values (raw, before clipping/imputation, for
  colouring), column names, and per-fit mean |SHAP| (25 × p) for S3.
- This re-run must reproduce `final_{arm}` OOF scores exactly. Assert this; do not log a second experiments.csv row for
  it, and log it as "explanation pass" in RESULTS.md.
- **Missing images:** R columns are median-imputed for scans without images, so they get attributions too. Report R
  SHAP over image-bearing scans only, and say so in the figure caption.

### Feature families

These are used everywhere SHAP is grouped. A family's SHAP is the sum of its members' SHAP per scan.

- own numerics (54 + 11 indices)
- fellow numerics (65)
- own−fellow asymmetry (65 + 8 worse_* + has_fellow)
- R own maps (96)
- R fellow maps (96)
- R own−fellow (96)
- image flags (2)
- age
- H (5, H arm only)

### Figures

- **Fig. E2a — family importance:** horizontal bars of mean |SHAP| per family, noH and H as paired bars.
- **Fig. E2b — top features beeswarm:** top 20 features, noH (H as a supplementary twin). Dots coloured by feature
  value percentile, with a diverging, colour-blind-safe map. Human-readable feature labels are needed: a label
  dictionary that maps column names to clinical names, e.g. `asym_Kmax Front` → "K max front, own − fellow" and
  `EB_r12_I` → "Back elevation, 1–2 mm ring, inferior".
- **Fig. E2c — dependence plots:** for the top 4 features, SHAP vs value, coloured by stage A, with a LOWESS trend.

## Part 4 — Where on the cornea (block R, spatial)

- Geometry from `kcref/radiomics.py`:
  - 4 maps: T thickness, K sagittal front curvature, EF front elevation, EB back elevation;
  - left eyes mirrored, so +x is nasal and +y superior;
  - 16 ring × quadrant sectors per map (rings 0–1, 1–2, 2–3, 3–4 mm; quadrants S/I/N/T);
  - plus extremum value / position, I−S and N−T differences on 2–4 mm, SD within 3 mm, and 4 thickness-progression
    features.
- **Fig. E3 — corneal importance maps:** a 3 × 4 grid. Rows are own / fellow / own−fellow; columns are the maps T, K,
  EF and EB.
  - Each panel is a polar "dartboard" of the 16 sectors (4 rings × 4 quadrants), coloured by mean |SHAP| on a shared
    sequential scale.
  - Orientation labels are S, I, N, T; the scale bar is in mm.
  - Non-sector features of that map (extremum, I−S, N−T, SD, thickness progression) appear as small labelled
    side-bars next to the panel.
- **Fig. E3b — signed version:** sectors coloured by the correlation between SHAP and feature value, i.e. "higher
  value here → higher referral score" (red) vs lower (blue), with a diverging scale centred at 0.
- **Fig. E3c — mean-map context:** the mean decoded map of referred vs not-referred first-visit eyes, and their
  difference, for the top 2 maps, with the top sectors outlined. Means are computed from `decoded_hr.npy`, mirrored as
  in radiomics. Group means only, so no individual is shown.

## Part 5 — Stand-out analyses

### S1. Matched-pair cases (Fig. E4)

- **Pool:** first-visit eyes with images in both eyes.
- **Pairing rule (fixed before any map is viewed):**
  1. For each referred eye, find its nearest non-referred eye in (Km F, age), standardised, within the same stage A,
     from a different patient. Each eye is used once (greedy matching by distance).
  2. Among the matched pairs, rank by (final OOF score gap) − (L1 OOF score gap), largest first. These are pairs the
     final model separates and the single-eye baseline does not.
  3. Show the top 3 pairs in the main figure and the next 5 in the supplement.
- **Also report** (honest framing):
  - how many matched pairs exist;
  - the fraction ordered correctly by L1 vs by the final model, with a sign test.

  This turns the cases from anecdote into a statistic.
- **Per pair panel:**
  - each eye's own and fellow T and EB maps (decoded), with the top-3 SHAP R sectors outlined;
  - a compact waterfall of the 8 largest family-level contributions;
  - the L1 vs final scores as two dots.
- **Labels:** "Pair 1a (referred) / 1b (not referred)", with Km F and age rounded to integers. No ids, no dates.

### S2. Explanations by stage (Fig. E5)

Family-level mean |SHAP| per stage, as a stacked or grouped bar or a heatmap (families × stages), for A and B and both
arms, first-visit eyes. It pairs visually with Fig. 1b: stage-AUC on top, what drives the score in that stage below.
Stages with < 10 eyes are shown greyed.

### S3. Robustness and clinical anchoring (Fig. E6)

- **Rank stability:** for the top 20 features, the median rank and 5–95% interval of the mean-|SHAP| rank across the
  25 CV fits (forest-style plot).
- **Family share stability:** the 25-fit distribution of each family's share of total |SHAP|.
- **Belin ABCD anchoring:** tag each top feature as follows:
  - **A:** anterior curvature/radius around the thinnest point or 3 mm zone (K map sectors; front radius numerics);
  - **B:** posterior radius (EB map sectors; back radius numerics);
  - **C:** thinnest pachymetry (T map, thinnest-point features, progression);
  - **other:** asymmetry, age, indices.

  Show tags as coloured chips on Fig. E2b and E6. The A/B/C mapping table goes in the report. *(My call:* tagging by
  rule from the feature name; borderline cases listed in a footnote.)

## Figure style (applies to every figure)

- One shared style module `publication_v2/style.py` (rcParams + palette), applied to all figures, including the
  restyled existing ones.
- **Sizes:** single column 89 mm, double column 183 mm (Nature/Ophthalmology-type widths). Export PDF (vector) +
  PNG 300 dpi; the report embeds the PNG.
- **Typography:** sans-serif, Arial if present else DejaVu Sans, 7–8 pt text and 9 pt panel titles. Panel letters (a,
  b, c) are bold, top left.
- **Axes:**
  - minimal: no top/right spines, light grid on the value axis only;
  - CI as thin whiskers with caps;
  - no chart junk.
- **Colour:**
  - **Arms (constant across all figures):** noH in deep blue `#1F4E79`-type, H in amber/orange; age-only and L1 in
    neutral greys.
  - **Families:** one fixed Okabe–Ito-based categorical palette, the same colour for a family in every figure.
  - **Sequential (importance):** a perceptually uniform map (viridis/cividis family or a single-hue blue).
  - **Diverging (signed SHAP, feature value):** a balanced blue–red diverging map centred at 0.
  - **Corneal raw maps (E3c, E4):** a perceptually uniform map, not the Pentacam rainbow. Each map type uses a fixed
    range across pairs, so panels compare.
  - Check every palette for colour-blind safety (deuteranopia simulation) and greyscale legibility.
- Every figure has a caption in the report, covering what is shown, n, and how the CIs were made.

## Report integration

- New sections in `publication_v2/report_template.html` after the current results:
  - Evaluation (1a, 1b);
  - Ladder (E1);
  - What the model uses (E2);
  - Where on the cornea (E3);
  - Cases (E4);
  - By stage (E5);
  - Robustness (E6).
- Each new figure gets a CSV of its underlying numbers in `publication_v2/tables/`.
- `build_report.py` reads the stored SHAP / ladder outputs. It does not recompute CV, so the report rebuild stays
  fast.
- After each piece lands:
  - rebuild the report;
  - log in RESULTS.md;
  - commit (author Hadar Cohen, no trailers).

## Order of work

1. Pre-declare E29 (ladder + explanation pass + S1 rule) in PROTOCOL.md.
2. Style module and restyled 1a / 1b. These are quick and give the look early for user feedback.
3. Ladder runs (background, both arms).
4. Explanation pass (OOF SHAP, both arms).
5. Figures E2, E3, then S1, S2, S3.
6. Report sections, RESULTS.md, commit.

## Left to the builder

- Exact plot geometry, annotation placement, and whether E5 is a heatmap or bars (pick the more legible).
- The human-readable label dictionary wording (must be clinically correct; keep units).
- Parallelism / runtime management of the CV runs.
- Whether 1a gets bootstrap bands.

## Decisions after builder review

These settle the questions raised by the builder review. Each is marked as the user's decision or my call.

1. **Population for SHAP summaries** *(my call)*. E2, E3, E5 and E6 summarise first-visit eyes (523 eyes), the
   same population as the headline AUC. S3 fit-level ranks use the first-visit test rows of each fit. An all-scans
   version of E2a goes in a table only. The stored OOF SHAP still covers all 944 scans.

2. **Collinear families** *(my call)*. Own, fellow and own−fellow are linear combinations of each other, so SHAP
   credit between them depends on the encoding.
   - **Primary grouping (main figures):** information *sources*:
     - own eye: own numerics + R own;
     - fellow-derived: fellow, asymmetry, worse_*, has_fellow, and R fellow / own−fellow;
     - age;
     - image flags;
     - H.
   - Each source is split into numerics vs maps (R), so the main view has four bars plus age, flags and H.
   - **Secondary:** the 9-family split, labelled as "how the model encoded it", with a caveat.
   - **Causal wording** ("the fellow eye adds X") is used only for ladder deltas (L1→L2, L1→L2b), never for SHAP bars.

3. **Per-scan SHAP storage** *(my call)*. `kcref/results/shap/` holds raw clinical values, so it is gitignored like
   `results/oof/`.

4. **S1 details** *(my call)*.
   - The noH final OOF score ranks the pairs and feeds the waterfall. The same pairs are shown for H, scored by H.
   - If fewer than 8 pairs exist, show what exists, with a minimum of 3 in the main figure. Matching is not widened
     across stages.
   - Scores are repeat-averaged within-fold ranks. Cross-fold comparability is approximate; state this in the caption.
   - Waterfall units are labelled "relative contribution (member-standardised SHAP)".

5. **Ladder framing** *(my call)*.
   - L1 is reported as found, with no "same as the previous paper" claim in figures.
   - The text states the previous paper's ~0.62 as context only; L1 is not a replication of that learner.
   - L0 (age only) is drawn as a reference line with no paired delta. Deltas start at L1→L2.

6. **Individual maps stay out of the repo until submission** *(user decision 2026-11-13)*.
   - Any figure showing an individual eye's map (E4 / S1 panels) is written only to `publication_v2/local/`, which
     is gitignored. That folder also holds a local-only `report_cases.html`.
   - The committed `report.html` carries S1 as statistics and a table only: the pair count and the fraction ordered
     correctly by L1 vs the final model, with a sign test. The case figures are referenced by name and not embedded.
   - Group-mean maps (E3c) are not individual and stay in the committed report.
   - At submission, the user decides whether to move the case figures in.
