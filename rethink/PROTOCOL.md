# Clean re-analysis protocol (branch `clean-rethink`)

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
