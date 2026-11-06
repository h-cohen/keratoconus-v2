# Results log (branch `clean-rethink`)

Dev set only (713 scans, 289 patients, 125 positive eyes). Lockbox untouched. All numbers are
pooled OOF AUC with a patient-cluster bootstrap 95% CI; `rep` is the mean of the 5 repeats.
Paired ΔAUC vs age-only (LR) uses `rethink/compare.py` (same patient clusters resampled).

## Status vs. success criteria (PROTOCOL.md)

No model yet meets all three criteria. Best dev candidate so far: **E02b / E02** (see below),
which is borderline on criterion 1 and does not clear criterion 2.

## E00 (reproduced exactly from the handoff)
See `results/experiments.csv`. Age-only LR AUC 0.656 [0.584–0.724]; first-visit 0.680.

## E01 — tabular done right (`exp01_tabular.py`)

| run | AUC [CI] | rep | first-visit | ΔAUC vs age |
|---|---|---|---|---|
| E01_age_spline_lr | 0.628 | — | — | −0.028 [−0.069, +0.009] |
| E01_eng_lr (13 KC indices + age, L2 LR) | 0.622 [0.552–0.691] | 0.621 | 0.655 | −0.033 [−0.077, +0.012] |
| E01_eng_mono_lgbm (age monotone −) | 0.595 [0.525–0.662] | 0.604 | 0.611 | −0.062 [−0.129, +0.005] |

**Interpretation:** nonlinear age, engineered KC indices and monotone trees do not beat a linear
age term. This rules out "better tabular modelling" as the route to 0.70.

Note: the first E01 spline run was logged before a column-name crash in the engineered runs.
Column names fixed (`Pachy Min:` etc.); the crashed attempt wrote no row.

## E02 — fellow eye at the same visit (`exp02_fellow.py`)

275 of 713 dev scans have a same-date scan of the other eye.

| run | AUC [CI] | rep | first-visit | ΔAUC vs age |
|---|---|---|---|---|
| E02_fellow_lr (+ has_fellow flag) | 0.701 [0.632–0.775] | 0.693 | 0.739 | +0.047 [−0.019, +0.111] |
| E02b_fellow_noflag (ablation) | 0.704 [0.638–0.773] | 0.692 | 0.742 | +0.050 [−0.011, +0.111] |

Within-subset AUCs (diagnostic, from OOF):

| model | scans with same-visit fellow | scans without |
|---|---|---|
| age-only | 0.705 | 0.608 |
| E02 | 0.718 | 0.583 |
| E02b (no flag) | 0.732 | 0.593 |

**Interpretation / caveats:**
- The gain is concentrated in scans that have a same-visit fellow scan (+0.03 within that subset).
  Scans without one are worse than age.
- `has_fellow` alone has AUC 0.596 (positive rate 25% with a fellow scan vs 13% without). It is
  allowed under the protocol (it describes the index visit), but it may reflect clinic workflow
  rather than disease. The ablation without it gives the same result, so it is not the main driver.
- Criterion 1 is borderline: pooled AUC 0.70, but the repeat-mean is 0.692. Criterion 2 fails:
  the paired CI includes 0.
- Not a success. Candidate for the lockbox only if the dev result firms up.

## E03 — visit + history (`exp03_history.py`), separate model

History subset: scans with an earlier scan of the same eye: **324 scans, 162 eyes, 135 patients,
27 positive scans** (very few positives; CIs will be wide). History features use only prior scans.

| run | AUC [CI] | rep | ΔAUC vs age (same subset) |
|---|---|---|---|
| E03_age_lr_on_history_subset (reference) | 0.604 [0.498–0.727] | 0.605 | — |
| E03_hist_lr (≈40 features: current + prior-scan indices & rates) | 0.495 [0.374–0.614] | 0.490 | −0.111 [−0.225, −0.008] |
| E03_hist_small_lr (age + yearly ΔKMax, ΔPachyMin, Δt) | 0.578 [0.464–0.716] | 0.587 | −0.025 [−0.057, +0.002] |

**Interpretation:** the previous paper's ~0.75 on serial exams does not reproduce under these rules.
Positives with history are too few here to fit many features. The small model is still no better
than age. This is a negative result; it does not rule out a larger, better-powered history study.

Log note: re-running `exp03_history.py` re-logged the E03_age and E03_hist_lr rows (the script ignores
its arguments and runs all runs). The duplicates are kept, per the "never delete rows" rule. Count
them as one configuration each when reporting.

## Harness / protocol notes
- `harness.summarise` now returns NaN for the first-visit AUC when a subset has no first-visit scans
  (needed for E03). Earlier numbers are unchanged.
- E00 was rerun during setup, which appended a second copy of each E00 row to `experiments.csv`.
  Count them once each.
