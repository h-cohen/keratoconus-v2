# Explainability and Evaluation (E29) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the evaluation figures, the feature-block ladder, out-of-fold SHAP explanations, corneal-region
maps and the three stand-out analyses (S1–S3) for both arms, and add them to `publication_v2/report.html`.

**Architecture:**
- `kcref/model.py` gains a column-subset argument (`blocks`) for the ladder.
- A new `kcref/explain.py` holds exact SHAP for the three blend members, a CV "explanation pass" that re-creates the
  final OOF scores while saving SHAP, and feature metadata (labels, families, sources, ABCD tags, sector geometry).
- `publication_v2/style.py` fixes one figure style.
- `publication_v2/figures.py` draws every committed figure from stored artefacts.
- `publication_v2/cases.py` draws individual-eye figures into gitignored `publication_v2/local/`.
- `build_report.py` embeds the figures and tables in new report sections.

**Tech Stack:**
- Python in `.venv-gpu`: numpy, pandas, scikit-learn, xgboost, lightgbm, matplotlib.
- No `shap`, no pytest. Tests are plain-assert scripts run with `.venv-gpu/bin/python -I`.

**Spec:** `docs/superpowers/specs/2026-11-13-explainability-evaluation-design.md`. Read it in full, including
"Decisions after builder review".

## Global Constraints

- Run from the repo root `/mnt5/home/hadar/kv2` with `.venv-gpu/bin/python -I`.
- CV only via `kcref/evaluate.py`: 5-fold StratifiedGroupKFold by patient × 5 repeats, seeds 2026–2030, 944 scans.
  Report pooled + first-visit AUC (523 eyes, 125 positive), with a 1000-resample patient-cluster bootstrap CI.
- Every new CV run appends to `kcref/results/experiments.csv`; never edit or delete rows. Pre-declare in
  `kcref/PROTOCOL.md` before running; log results in `kcref/RESULTS.md`.
- Both arms, noH and H, for every analysis, side by side. noH is the headline. H is labelled "visit + fellow-eye
  treatment history".
- Staging only via `kcref/staging.py` (`stage_A`, `stage_B`).
- Privacy:
  - no pid, name, id, date or filename in any committed figure, table or report;
  - individual-eye maps only in `publication_v2/local/` (gitignored);
  - maps are drawn only from `kcref/data/decoded_hr.npy`, which is already mirrored so that +x = nasal for both eyes.
- The default model path must stay bit-identical: `blocks='final'` reproduces `E28_{arm}_blend` OOF exactly.
- Git:
  - author `Hadar Cohen <hal.nls@gmail.com>`, no Co-Authored-By or session trailers;
  - commit on branch `clean-rethink`;
  - push only when the user asks.
- Sub-agents use `model: "sonnet"`.
- Figure style (spec "Figure style"):
  - widths 89 mm and 183 mm; PDF + PNG at 300 dpi;
  - font Liberation Sans (installed, Arial-metric), fallback DejaVu Sans, 7–8 pt, panel letters bold;
  - noH `#1F4E79`, H `#D98E04`, age-only `#9A9A9A`, L1 `#4D4D4D`;
  - families from Okabe–Ito;
  - sequential `cividis`, diverging `RdBu_r` centred at 0, corneal raw maps perceptually uniform (not rainbow).
- SHAP summaries use first-visit eyes. R-family SHAP is averaged only over eyes whose relevant image exists:
  `own_image == 1` for R own, `has_fellow_image == 1` for R fel / asym.

## Review Focus

1. **Scans without images.** R columns are median-imputed, so they get spurious attributions. Expected: excluded from
   R summaries (Task 4 test `test_r_mask`).
2. **Map orientation.** Nasal/temporal or superior/inferior flipped between the sector dartboards and the mean maps.
   Expected: dartboard +x = nasal, +y = superior, the same as radiomics. Task 6 test `test_sector_orientation`.
3. **Tiny strata.** Stage A4 has 5 scans, and some stages may have a single class in first visit. Expected: hatched bar,
   no CI, no crash. Task 5 test `test_stage_auc_small`.
4. **Privacy leak in committed outputs.** A pid or date column slips into `tables/*.csv`, or `local/` is not ignored.
   Task 9 test `test_no_identifiers`.
5. **Explanation pass drifts from the logged model.** For example, a non-deterministic thread order changes the OOF.
   Expected: hard failure, no silent new numbers. Task 3 asserts max |Δ| < 1e-12 vs `oof/E28_{arm}_blend.npy`.

---

### Task 1: Ladder support in `model.py`, pre-declaration, launch ladder runs

**Files:**
- Modify: `kcref/model.py` (`fit`, `main`, new `select_columns`)
- Modify: `kcref/PROTOCOL.md` (append E29 pre-declaration), `.gitignore`
- Test: `kcref/tests/test_ladder.py`

**Interfaces:**
- Produces:
  - `select_columns(cols: list[str], blocks: str) -> list[str]`, with blocks one of `'L1' | 'L2' | 'L2b' | 'final'`.
    H columns (`F.H_COLS`) are kept whenever present.
  - `fit(D, rows, use_h, seed, blocks='final') -> dict`; the `fz['cols']` subset is used by `matrix` unchanged.
  - CLI `python -I kcref/model.py cv {noH|H} [--blocks L1|L2|L2b]` logs `E29_ladder_{blocks}_{arm}`. Without
    `--blocks`, it logs `final_{arm}` as before.
- Rung definitions (column prefixes come from `features.block_N` / `block_R`):
  - L1: own columns only = `age` + `NUM` + 11 engineered. Excludes `fel_*`, `asym_*`, `worse_*`, `has_fellow`, flags
    and `R_*`.
  - L2: all of block N (205).
  - L2b: L1 + `R_own*` + `own_image`.
  - final: everything.

- [ ] **Step 1: Write the failing test** `kcref/tests/test_ladder.py`. Build a fake column list from the real names:
  `F.block_N` columns (rebuild them from the `NUM` list + `engineer` output names, with no data reads), H_COLS,
  `['has_fellow_image','own_image']`, and `R_own0..95`, `R_fel0..95`, `R_asym0..95`. Assert:
  - `len(select_columns(cols_noH,'L1')) == 66`;
  - `'L2' → 205`;
  - `'L2b' → 66+96+1 == 163`;
  - `'final' → 495`;
  - with H columns: each count + 5;
  - order is preserved (the output is a subsequence of the input).
- [ ] **Step 2:** Run `.venv-gpu/bin/python -I kcref/tests/test_ladder.py`. Expected: fails with ImportError on
  `select_columns`.
- [ ] **Step 3: Implement** `select_columns` and the `blocks` argument in `fit`. Apply the subset right after
  `F.assemble`, before the quantiles. Add the `--blocks` CLI. Add `kcref/results/shap/` and `publication_v2/local/` to
  `.gitignore`.
- [ ] **Step 4:** Run the test. Expected: prints `ok`.
- [ ] **Step 5: Bit-identity check.** Write `scratchpad/bitid.py` (not committed):
  - on fold 1 of seed 2026, run the old path (`git show HEAD:kcref/model.py` into a temp module) and the new
    `blocks='final'` path for noH and H;
  - expected: `max |diff| == 0.0` for both.
- [ ] **Step 6: Pre-declare in `kcref/PROTOCOL.md`** under the heading `## E29 (2026-11-13): ladder and explanation
  pass`. Cover:
  - the rungs above × both arms;
  - the paired Δ L1→L2, L2→final, L1→L2b, L2b→final (first visit, cluster bootstrap);
  - that the explanation pass re-creates the `E28_{arm}_blend` OOF and is not a new experiments row;
  - the S1 pairing rule copied from the spec;
  - that no result here changes the frozen model.
- [ ] **Step 7: Commit** `kcref/model.py kcref/tests/test_ladder.py kcref/PROTOCOL.md .gitignore`, with the message
  "E29: ladder column subsets in model.py; pre-declaration".
- [ ] **Step 8: Launch the six ladder runs in the background**, one process each: `cv noH --blocks L1`, `cv H --blocks
  L1`, and the same for L2 and L2b. Output goes to `kcref/results/E29_ladder_{blocks}_{arm}.log`. Each run takes
  ≈ 4–5.5 h (E28 blend: 15.7 ks noH, 19.7 ks H), on 40 cores with n_jobs=4 each. Do not wait; continue with Task 2.

### Task 2: Exact member SHAP and blend SHAP

**Files:**
- Create: `kcref/explain.py`
- Test: `kcref/tests/test_explain.py`

**Interfaces:**
- Consumes: `model.member_scores`, `model.matrix`, and the `fz` dict from `model.fit`.
- Produces:
  - `member_shap(fz, X) -> dict[str, np.ndarray]`: keys = `model.MEMBERS`; each value is (n, p) in that member's
    log-odds / margin.
    - xgb: `get_booster().predict(DMatrix(X), pred_contribs=True)[:, :-1]`;
    - lgbm_bag: mean over the 5 models of `predict(X, pred_contrib=True)[:, :-1]`;
    - lr_en: `coef * scaler.transform(X)`, the linear SHAP with the scaler mean as reference.
  - `member_margin(fz, X) -> dict[str, np.ndarray]`: each member's margin (xgb `output_margin=True`, lgbm
    `raw_score=True` bag-mean, lr `decision_function`).
  - `blend_shap(fz, X) -> np.ndarray` (n, p): mean over members of `member_shap[k] / member_margin[k].std()`.

- [ ] **Step 1: Write failing tests** with synthetic data (no patient data): 300 rows, 20 features, binary y. Fit a
  tiny `fz` by calling `model.learner` directly for each member with fixed hyper-parameters, and fake `fz['models']`.
  Assert:
  - `test_additivity`: per member `shap.sum(1) + bias == margin` (atol 1e-4); bias = last contrib column for xgb/lgbm, `intercept_` for lr.
  - `test_blend_shape`: `blend_shap` returns `(300, 20)`, all finite.
  - `test_member_agreement`: a planted strong feature 0 has the largest mean |SHAP| in all three members.
- [ ] **Step 2:** Run `.venv-gpu/bin/python -I kcref/tests/test_explain.py`. Expected: fails with ImportError.
- [ ] **Step 3: Implement** the three functions in `kcref/explain.py`.
- [ ] **Step 4:** Run the tests. Expected: `ok`.
- [ ] **Step 5: Commit**, message "E29: exact member SHAP and blend SHAP".

### Task 3: Explanation pass (OOF SHAP, both arms)

**Files:**
- Modify: `kcref/explain.py` (add `shap_pass`, CLI)
- Output (gitignored): `kcref/results/shap/{arm}_oof.npz`

**Interfaces:**
- Produces:
  - `shap_pass(arm: str) -> Path`. It runs `evaluate.cv_oof(coh, fp)` with a fit_predict that:
    - fits via `model.fit` (`blocks='final'`);
    - returns the same within-fold rank blend as `model.cv_fit_predict`;
    - stores, for that fit's test rows, `blend_shap`, each member's mean |SHAP| over the fit's first-visit test rows,
      and the fit id.
  - It asserts that the returned OOF equals `np.load(RES/'oof'/f'E28_{arm}_blend.npy')` (max |Δ| < 1e-12). It does
    not call `evaluate.evaluate` (no new experiments row).
  - npz keys:
    - `shap` (944, p): mean over the 5 repeats;
    - `fit_imp` (25, p): mean |blend SHAP| over first-visit test rows per fit;
    - `member_imp` (3, p);
    - `cols` (p,);
    - `raw` (944, p): un-standardised values for colouring, with N from `D['N']`, H from `D['H']`, flags, and R from
      `D['R'][own]`, `D['R'][felc]` and their difference, NaN where missing;
    - `oof` (944,).
  - CLI: `python -I kcref/explain.py shap {noH|H}`.

- [ ] **Step 1: Implement** `shap_pass` and the CLI.
- [ ] **Step 2: Smoke test** on one fold. Add an env switch `KC_SHAP_SMOKE=1` that limits the pass to repeat 0, fold
  0 and skips the full-OOF assert. Instead, assert equality with `E28_{arm}_blend.npy[0, test_rows]`. Run for noH.
  Expected: `smoke ok`, max |Δ| = 0.0.
- [ ] **Step 3: Launch** both arms in the background, writing logs to `kcref/results/E29_shap_{arm}.log`. Expected
  final line: `saved results/shap/{arm}_oof.npz; oof identical`.
- [ ] **Step 4: Commit** `kcref/explain.py`, message "E29: OOF SHAP explanation pass".

### Task 4: Feature metadata (labels, families, sources, ABCD, sectors)

**Files:**
- Modify: `kcref/explain.py`
- Test: `kcref/tests/test_explain.py`

**Interfaces:**
- Produces:
  - `family(col) -> str`, one of `own_num`, `fel_num`, `asym_num`, `R_own`, `R_fel`, `R_asym`, `flags`, `age`, `H`.
    `worse_*` and `has_fellow` go to `asym_num`.
  - `source(col) -> str`, one of `own_num`, `own_maps`, `fellow_num`, `fellow_maps`, `age`, `flags`, `H`, where:
    - fellow_num = `fel_num` + `asym_num`;
    - fellow_maps = `R_fel` + `R_asym`;
    - own_maps = `R_own`.
  - `r_name(col) -> str`: the radiomics feature name (`radiomics.names()[j]`) for `R_{own|fel|asym}{j}`.
  - `label(col) -> str`: a human-readable clinical label with units. Examples:
    - `asym_KMax Sagittal Front (D)` → "K max front (D), own − fellow";
    - `R_own48` → "Back elevation, 0–1 mm, superior (own)".
    - Map names: T thickness (µm), K front sagittal curvature (D), EF front elevation (µm), EB back elevation (µm).
  - `abcd(col) -> str`, one of `A|B|C|other`:
    - A: anterior curvature/radius (K map, `Rf F`/`Rm F`/`K* F`/`KMax` numerics);
    - B: posterior radius/elevation (EB map, `R* B`, `K* B`);
    - C: thickness (T map, `Pachy*`, `T_prog*`, `pachy_*`);
    - else `other`.
  - `sector(rname) -> (map, r0, r1, quad) | None` for ring × quadrant features, else None.
  - `r_mask(cols, raw_flags) -> np.ndarray[bool] (n, p)`: True where the column's image exists for that row.

- [ ] **Step 1: Write failing tests:**
  - `test_family_counts`: on the 495 noH columns, counts are own_num 65, age 1, fel_num 65, asym_num 74, R_* 96
    each, flags 2.
  - `test_source_partition`: each column maps to exactly one source.
  - `test_r_name`: `r_name('R_own0') == 'T_r01_S'` and `r_name('R_asym95') == 'T_ratio31'`.
  - `test_sector`: `sector('EB_r12_I') == ('EB', 1, 2, 'I')`.
  - `test_r_mask`: rows with `own_image=0` are False on `R_own*` and True on N columns.
  - `test_labels_unique`: all 500 H-arm labels are unique and non-empty.
  - `test_abcd_examples`: Pachy Min → C, `R_own` EB sector → B, `Km F (D):` → A, `age` → other.
- [ ] **Step 2:** Run. Expected: fails.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run. Expected: `ok`.
- [ ] **Step 5: Commit**, message "E29: feature metadata for explanations".

### Task 5: Figure style + evaluation figures (1a ROC, 1b stage AUC)

**Files:**
- Create: `publication_v2/style.py`, `publication_v2/figures.py`
- Modify: `publication_v2/build_report.py` (use them; replace current ROC/strata drawing)
- Test: `publication_v2/test_figures.py`

**Interfaces:**
- `style.py` produces:
  - `apply() -> None` (rcParams);
  - `C: dict` (arm/method colours from Global Constraints);
  - `FAM: dict[source|family, hex]`;
  - `MM1 = 89/25.4`, `MM2 = 183/25.4` (inches);
  - `save(fig, name, local=False) -> Path`: writes `figures/{name}.pdf` and `.png` at 300 dpi, or `local/…` when
    local, and returns the PNG path;
  - `panel_letter(ax, s)`.
- `figures.py` produces:
  - `stage_auc(y, scores: dict[str, array], stage: array, groups, min_class=10) -> pd.DataFrame` with columns
    `method, stage, n, pos, auc, lo, hi, small`. `small` = fewer than min_class positives or negatives. lo/hi are NaN
    when small, and auc is NaN when the stage has a single class.
  - `fig_roc(...)` and `fig_stage_auc(df, staging: 'A'|'B') -> Path`. The second is one row of 4 panels (age only,
    L1, final noH, final H), bars per stage with CI whiskers, n/pos under the bars, small stages hatched, and a 0.5
    line.
  - Until the L1 OOF exists, the L1 panel shows "pending".

- [ ] **Step 1: Write failing test** `test_stage_auc_small`. Use synthetic data with 4 stages, where stage 4 has 5
  rows of one class. Assert:
  - the row for stage 4 has `small == True` and `np.isnan(auc)`;
  - the other stages have finite CI with `lo <= auc <= hi`.
- [ ] **Step 2:** Run `.venv-gpu/bin/python -I publication_v2/test_figures.py`. Expected: fails.
- [ ] **Step 3: Implement** `style.py`, `stage_auc`, `fig_roc`, `fig_stage_auc`, and wire them into
  `build_report.py`. Load the `dataviz` skill first and check the palette with its validator.
- [ ] **Step 4:** Run the test and `python -I publication_v2/build_report.py`. Expected:
  - the test passes;
  - `figures/roc_final.{png,pdf}`, `figures/stage_auc_A.*` and `figures/stage_auc_B.*` exist.

  View the PNGs with the Read tool. Check: no overlapping text, legible at 100% width, and colours as in the spec.
- [ ] **Step 5: Commit.** Then show the user the two figures, for early style feedback, before Task 6.

### Task 6: Explainability figures E2 (sources/families, beeswarm, dependence) and E3 (cornea)

**Files:**
- Modify: `publication_v2/figures.py`, `publication_v2/test_figures.py`

**Interfaces:**
- Consumes: `kcref/results/shap/{arm}_oof.npz` (Task 3) and `explain.family/source/label/sector/r_mask/abcd`
  (Task 4).
- Produces:
  - `fig_sources(npz_noH, npz_H) -> Path` (E2a): mean |SHAP| per source, paired bars noH/H, first visit; secondary
    panel = the 9 families with the caveat in the caption.
  - `fig_beeswarm(npz, arm, top=20) -> Path` (E2b): ABCD chips on the labels.
  - `fig_dependence(npz, arm, top=4) -> Path` (E2c): coloured by stage A, LOWESS via a rolling median on sorted x
    (no statsmodels).
  - `fig_cornea(npz, arm, signed: bool) -> Path` (E3/E3b): 3 × 4 grid of polar sector dartboards; the non-sector
    features of each map as a mini bar beside the panel.
  - `fig_mean_maps(arm) -> Path` (E3c): mean `decoded_hr` of referred vs not first-visit eyes plus their difference,
    top 2 maps, with the top-3 sectors outlined. Uses only the eyes with `own >= 0`.
  - Each figure writes its numbers to `tables/{name}.csv`.

- [ ] **Step 1: Write failing test** `test_sector_orientation`. Make a synthetic dartboard input where only `EB_r23_N`
  has importance 1. Draw it and read back the patch centroid. Assert:
  - the centroid is at x > 0, |y| < x (the nasal side);
  - for `_S`, y > 0.

  Also assert that the mean-map helper puts a synthetic blob placed at +x columns (nasal, per `radiomics.grid`) on the
  same side.
- [ ] **Step 2:** Run. Expected: fails.
- [ ] **Step 3: Implement** after the SHAP npz files exist. Check with `ls kcref/results/shap/`; if they are missing,
  wait for the Task 3 notification.
- [ ] **Step 4:** Run the test and rebuild the report. View every new PNG and check that it is legible and the
  orientation labels S/I/N/T are present.
- [ ] **Step 5: Commit**, message "E29: SHAP and corneal importance figures".

### Task 7: Ladder figure E1 + RESULTS

**Files:**
- Modify: `publication_v2/figures.py`, `build_report.py`, `kcref/RESULTS.md`

**Interfaces:**
- `fig_ladder(rungs: list[str], arm_oofs) -> Path`:
  - horizontal dot-and-whisker of first-visit AUC [CI];
  - two tracks (noH, H); L0 age-only as a vertical reference line;
  - paired Δ [CI] annotated L1→L2→final and L1→L2b→final, via `evaluate.paired_delta` on first-visit rows with the
    repeat-mean OOF.
  - Also writes `tables/ladder.csv`.

- [ ] **Step 1:** After all six ladder logs end, check the experiments.csv rows `E29_ladder_*`. Expected: 6 rows.
- [ ] **Step 2: Implement** `fig_ladder`. Rebuild the report and view the figure.
- [ ] **Step 3: Log in RESULTS.md**: a table of the rungs × arms with the paired deltas. State L1 as found, with the
  previous paper's ≈0.62 as context only.
- [ ] **Step 4: Commit**, message "E29: ladder results".

### Task 8: S2 explanations by stage (E5) + S3 robustness/ABCD (E6)

**Files:**
- Modify: `publication_v2/figures.py`

**Interfaces:**
- `fig_stage_shap(npz_noH, npz_H, staging) -> Path`: a heatmap of sources × stages (cell = mean |SHAP| share,
  annotated). Arms side by side. Stages with fewer than 10 eyes greyed out.
- `fig_robustness(npz, arm) -> Path`: forest of the median rank with a 5–95% interval across the 25 fits for the top
  20, ABCD chips, plus a strip of source share per fit.
- `abcd_table(npz) -> pd.DataFrame` → `tables/abcd.csv`.

- [ ] **Step 1: Implement** both figures for both arms and both stagings.
- [ ] **Step 2:** Rebuild the report and view the PNGs.
- [ ] **Step 3: Commit**, message "E29: stage-wise explanations and robustness".

### Task 9: S1 matched pairs (stats committed, case figures local)

**Files:**
- Modify: `kcref/explain.py` (matching)
- Create: `publication_v2/cases.py`
- Test: `kcref/tests/test_explain.py`, `publication_v2/test_figures.py`

**Interfaces:**
- `match_pairs(df: DataFrame[pid, y, km, age, stage]) -> DataFrame[pos_idx, neg_idx, dist]`:
  - z-score km and age over df;
  - for each positive in order of increasing nearest distance, take its nearest unused negative of the same stage
    and a different pid (greedy).
- `pair_stats(pairs, s_final, s_L1) -> dict`: n_pairs, the fraction with pos > neg under each model, and a
  two-sided sign test with `scipy.stats.binomtest` on the pairs where the models disagree.
- `cases.py`:
  - ranks the pairs by `(s_final[pos]-s_final[neg]) - (s_L1[pos]-s_L1[neg])`, using noH;
  - draws the top 3 (main) and the next 5 into `publication_v2/local/cases_*.{png,pdf}` and `local/report_cases.html`;
  - labels "Pair k a/b" only, with Km F and age rounded to integers;
  - shows the H scores of the same pairs.

- [ ] **Step 1: Write failing tests:**
  - `test_match_rules` (synthetic): pairs share a stage, pids differ, and no index is used twice.
  - `test_no_identifiers`: after `build_report.py`, no `tables/*.csv` has a column named in
    `{'pid','pkey','date','scan_date','name','file'}`; `git check-ignore -q publication_v2/local/x.png` returns 0;
    and `report.html` contains no 16-hex-char pid-like strings that match any cohort pid.
- [ ] **Step 2:** Run. Expected: fails.
- [ ] **Step 3: Implement** `match_pairs` and `pair_stats` on first-visit eyes with both images. Write
  `tables/pairs_stats.csv` and a report section with the stats (no maps). Implement `cases.py`.
- [ ] **Step 4:** Run the tests. Run `python -I publication_v2/cases.py` and view the local figures. Run `git status
  --short publication_v2/local`. Expected: empty (ignored).
- [ ] **Step 5: Commit** the code and committed outputs only, message "E29: S1 matched pairs".

### Task 10: Report sections, final rebuild, log

**Files:**
- Modify: `publication_v2/report_template.html` (sections 18–24 per the spec's report list, each with captions:
  shown, n, CI method), `build_report.py`, `kcref/RESULTS.md`, `README.md` (one line: figures in
  `publication_v2/figures/`)

- [ ] **Step 1:** Add the sections and captions, and update the table of contents.
- [ ] **Step 2:** Run all tests and the report rebuild:

  ```bash
  for t in kcref/tests/test_ladder.py kcref/tests/test_explain.py publication_v2/test_figures.py; do .venv-gpu/bin/python -I $t || exit 1; done
  .venv-gpu/bin/python -I publication_v2/build_report.py
  ```

  Expected: every test prints `ok`, and the report builds.
- [ ] **Step 3:** Run `.venv-gpu/bin/python -m pyflakes` via system `python3 -m pyflakes kcref/*.py kcref/tests/*.py
  publication_v2/*.py`. Expected: no output.
- [ ] **Step 4:** Make a RESULTS.md entry "E29 explainability" with the key numbers per arm.
- [ ] **Step 5: Commit** "E29: report sections". Report to the user and ask about pushing.
