# Clean re-analysis protocol (branch `clean-rethink`)

> **Historical log.** This file records every experiment in the order it was run (2026-11-06 to 2026-11-13).
> Scripts and paths named here (`rethink/…`, `harness.py`, `expNN_*.py`) now live in `kcref/archive/`.
> The final model is `kcref/model.py`; its runs are `E28_noH_blend` and `E28_H_blend`.


Frozen 2026-11-06, before any new experiment was run. Changes to this file after the freeze are
logged at the bottom with a date and reason.

## Goal

Find the best legitimate predictor of CXL referral from Pentacam data, aiming for **ROC-AUC ≥ 0.70**,
and make the number that goes in the paper hold up to review. "By any means" covers models,
features, encoders and pipelines. It does not cover anything below that would inflate the estimate.

## Why the submitted 0.708 doesn't count

1. XGBoost early stopping used the test fold (`src/engine.py`, `src/evaluation.py`).
2. CV was split by eye, so the two eyes of one patient could land in different folds.
3. 53 of 220 positive scans were taken after the eye had already been cross-linked.
4. Units were scans, but the paper called them eyes.

## Cohort (frozen, `prepare_data.py`)

- Legacy inclusion: all four maps, no missing numerics, `Km F (D):` ≤ 55. This is kept for
  comparability. Note that the filter is on mean front K, not on KMax.
- Excluded: CXL-labelled scans taken on/after the CXL date, and scans taken on/after keratoplasty.
- Result: **944 scans, 523 eyes, 385 patients, 167 positive scans (17.7%), 125/523 positive eyes.**
- Names, national IDs and dates of birth are dropped. Patients are keyed by a salted hash (`pid`).

## Splits

- **Lockbox**: 25% of patients (96 patients, 134 eyes, 231 scans, 42 positive scans), drawn once
  with a fixed seed, stratified by patient label. sha256 of `lockbox_pids.csv`:
  `f875b78075576864c9b8acf25cbdd87d2700647dbfbd93105af37043dca04956`.
  **Touched once only**, for the final pre-declared model(s). Re-drawing it is forbidden.
- **Development set**: the remaining 289 patients (713 scans). All experimentation happens here.

## Evaluation (`harness.py`)

- 5-fold `StratifiedGroupKFold` by patient, repeated 5× (seeds 2026–2030).
- All preprocessing is fitted inside the training fold: scaling, feature selection, imputation,
  early stopping (inner patient-grouped split) and hyper-parameter tuning (nested).
- Primary metric: pooled out-of-fold AUC over all dev scans (predictions averaged over repeats),
  with a 95% CI from a patient-cluster bootstrap.
- Co-primary: AUC on **first-visit scans only** (one per eye). This is the "single visit at
  presentation" claim in the paper.
- Every run is appended to `results/experiments.csv`. Rows are never deleted, so the full
  number of configurations tried can be reported.

## Allowed inputs

Allowed: anything measurable at the index visit. That covers the index-scan numerics and maps,
age, and the **fellow eye's scan from the same visit**. It also covers prior scans of the same
patient. Models that use prior scans are reported as a separate "visit + history" model, not as
"single visit".

**Forbidden** (leakage or label proxies):
- scan year or calendar date
- number of scans or follow-up length
- any information dated after the index scan, including future fellow-eye CXL
- `months_to_cxl`, `visit_idx` and `legacy_row` used as model inputs
- free-text comments

## What "beating 0.70" means

A model is declared a success only if **all** of the following hold:

1. dev repeated-CV AUC ≥ 0.70;
2. a paired bootstrap shows it beats age-only logistic regression (ΔAUC CI excludes 0);
3. its lockbox AUC is reported as the headline, whatever it turns out to be.

At most **3** pre-declared candidates are scored on the lockbox.

## Reference points (E00, dev set)

| model | AUC [95% CI] | first-visit AUC |
|---|---|---|
| age only (LR) | 0.656 [0.584–0.724] | 0.680 |
| 55 numerics, L2-LR | 0.624 | 0.666 |
| 55 numerics, XGB | 0.609 | 0.618 |
| v15 multimodal, clean | 0.575 | 0.592 |
| v15 multimodal without age | 0.448 | 0.484 |
| frozen R18 images only | 0.468 | 0.497 |

## Change log
- 2026-11-06 frozen.
- 2026-11-06 (during E03): `harness.summarise` returns NaN for first-visit AUC when a subset has no
  first-visit scans. Needed for the history subset; does not change any existing number.
- 2026-11-06: added `rethink/compare.py` (paired cluster bootstrap of ΔAUC vs a reference run),
  which the protocol's success criterion 2 requires.
- 2026-11-06 **Lockbox candidates declared (3, max allowed)**, before any lockbox score:
  1. `E06b_fellow_mono_lgbm` (E02 fellow-eye features + age, monotone LightGBM) — primary;
  2. `E06a_fellow_dino_lr` (E02 features + E04 DINOv2 PCA10 + age, L2 LR);
  3. `E02_fellow_lr` (E02 features + age, L2 LR).
  Each is fitted once on the full dev set (713 scans) with seed 2026 and scored once on the lockbox
  (231 scans). The lockbox is spent after this step; any later experiment is judged on dev only.

- 2026-11-10 **E23 pre-declared (after E22 results were seen; reported as such).** User goal: first-visit AUC
  ≥ 0.80. Declared once, before any E23 run; no E23 variant is added after seeing E23 results.
  1. Weighting: first-visit cohort scans weight 2 in every learner's fit, all other rows weight 1.
  2. Stacking: in each outer training fold, non-negative logistic stacker on the 4-fold inner OOF predictions of
     the candidate pool, replacing single-winner selection.
  3. Fine-tuned encoder L: in each (outer and inner) training fold, L's last-epoch weights + linear head trained
     on referral, 20 epochs, AdamW lr 1e-4, wd 0.05, pretraining augmentations, last epoch kept; no early
     stopping, no AUC-based choice. Own and mirrored-fellow composites; its probability enters the stack.
  Runs: E23n_H (pool = 21 E22 candidates + fine-tuned L, with block H) and E23n_noH (same without H). Primary
  metric for this addendum: first-visit AUC; also pooled AUC, paired dAUC vs E22_ref_single_num and E22n_nested.

### 2026-11-11 — E24 pre-declared (after E23 results were seen)

**Reason.** The user wants the most training data in every model. In E22/E23, only the bn_aug candidates use the extra labelled exams.

**Change.** E24 = E23 exactly, except every one of the 21 candidates adds the extra exams (`extra_exams.parquet`) to its training rows. As before, they are used in training only, and a cohort patient's extras only when that patient is in the training fold. The extra exams have no composite image, so their image blocks are missing and flagged.

**Unchanged.** ftL stays as in E23, because it needs images. Same seeds, folds, stacker and first-visit weight 2.

**Runs.** E24n_H and E24n_noH, each run once. Whatever the result, it is reported. No further variants.

**History route.** A visit + history E25 was considered and not run: E03 and E11 already tested it and found nothing.

### 2026-11-11 — E25 pre-declared: more from the images (after E24 results were seen)

**Reason.** The user asked to try harder with the images.

**Pool.** The E24 pool plus 6 image candidates. All of them use the extra exams and first-visit weight 2, and each is fitted with lr and lgbm:
- **bi_R:** block R (`radiomics.py`), 96 label-free measurements of the decoded maps. These are ring × quadrant means, extremum value and location, I−S and N−T asymmetry, SD within 3 mm, and thickness progression around the thinnest point. Own, fellow and own − fellow; standardised, no PCA.
- **bi_probe:** for each of Z, D, L, X, T and R, a ridge-logistic probe (C = 0.01, fixed in advance) of the label on the own-composite embedding. It is fitted in fold. Training composites get 5-fold patient-grouped cross-fitted scores. The features are the own score, the fellow score and their difference.
- **bi_multi:** PCA(10) of Z, D, L, X and T side by side.

**Dropped before running.** Flip/rotation test-time averaging of the embeddings. Flips destroy the inferior–superior and nasal–temporal asymmetry, which is diagnostic in keratoconus.

**Runs.**
- Stack runs: E25n_H and E25n_noH, with the same stacker, folds and seeds as E23/E24.
- Each new candidate is also evaluated alone with block H (E25_bi_*), for the record.
- All are run once and reported whatever the result.

The number of configurations rises accordingly. The best model so far was selected among many, so the paper must report the count.

### 2026-11-11 — E26 and E27 pre-declared (user: an ML/deep-learning image arm; use all available information)

**E26, images alone.**
- Features: only one image representation, for the own eye, the fellow eye and own − fellow, plus has_fellow_image. No numerics, no age, no block H.
- Model: LR with inner CV over C.
- Rows: cohort scans only.
- Run once each for Z, D, L, X, T, R and probe, and for G0 and G once trained.
- Purpose: do the images carry referral signal of their own? Age alone ≈ 0.66 is the bar.

**E27, position-aware SSL** (`pretrain_geo.py`; fixed configuration in its docstring, written before training).
- **G0:** LeJEPA as L, with three changes:
  - coordinate channels warped with the maps;
  - gentle global views (scale 0.85–1, rotation ±5°, shift ±0.3 mm, no flips);
  - 11 input channels.
- **G:** G0 plus an auxiliary linear head that predicts the 96 label-free block R measurements (MU = 0.1).
- Training: 300 epochs; the last epoch is kept; labels are never read.
- Evaluation:
  - E26 image-only, for G0 and G;
  - standalone bi_G_lgbm and bi_G0_lgbm (block H, extra exams, first-visit weight 2);
  - stacks E27n_H and E27n_noH: the E25 pool plus bi_G0_{lr,lgbm} and bi_G_{lr,lgbm}, with the same stacker, folds and seeds.
- Every run is executed once and reported whatever the result.

### 2026-11-12 — lockbox dropped (user decision)
The user decided not to score the lockbox. All results are reported on all 944 scans with the repeated patient-grouped
CV (`harness.evaluate`). Consequence for reporting: the best of many configurations is selected on the same data it is
reported on, so the top AUCs carry selection optimism. Report the number of configurations tried next to them.

### 2026-11-12 — E28 pre-declaration: learner variants of the final non-stack model
Final model family (user decision): bi_R = numerics + fellow + age + R map measurements, no stacker. E28 tries 7
learner variants on the same input (`exp28_tricks.py`: xgb, lgbm_deep, lgbm_bag, et, lr_en, svm, blend = fixed equal-rank
mean of xgb / lgbm_bag / lr_en). All tuning by inner patient-grouped CV in the training fold; harness seeds fixed.
noH first (primary), H after. Selection rule: the highest noH first-visit AUC among E25_noH_bi_R_lgbm and the E28 variants
becomes the final model; report the number of variants tried (7 + 1) and the paired delta against E25_noH_bi_R_lgbm.

### 2026-11-13 — E28 addendum
Three input-side noH variants added to `exp28_tricks.py` (lgbm_nan: native NaN instead of median imputation;
lgbm_nofel: R without the fellow copy; pca_lr: PCA(k) + L2 logistic, k and C tuned in fold). The E28 count becomes
7 + 3 + 1 = 11 noH configurations. H arm: the 7 learner variants, run for the two-arm rule (CLAUDE.md).
