# Results log (branch `clean-rethink`)

Dev set only (713 scans, 289 patients, 125 positive eyes). Lockbox untouched. All numbers are
pooled OOF AUC with a patient-cluster bootstrap 95% CI; `rep` is the mean of the 5 repeats.
Paired ΔAUC vs age-only (LR) uses `rethink/compare.py` (same patient clusters resampled).

## Status vs. success criteria (PROTOCOL.md)

No model meets all three criteria. Best dev candidate: **E02 / E02b** (same-visit fellow eye),
borderline on criterion 1 (pooled 0.701/0.704, but repeat-mean 0.693/0.692) and failing criterion 2
(paired ΔAUC CI includes 0). Lockbox not scored, because no candidate has passed.

Routes tested: E01 tabular, E02 fellow eye, E03 prior scans, E04 frozen DINOv2 maps, E05 raw-export
indices, E06 pre-declared combinations (E06b gives dev AUC 0.723, but its significance vs age is
borderline and seed-unstable; see the E06 section). Not tested: image fine-tuning, physical-scale map inversion (needs the Pentacam colour
LUT, which is not in the repo), and E06 stacking (not worth running when no component passes).

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

## E04 — frozen DINOv2-small image features (`extract_dino.py`, `exp04_dino.py`)

4 maps × (CLS + mean patch) × 384 → per-map standardise + PCA(10) fitted in-fold → L2 LR with age.
C tuned by inner CV; PCA refit inside each inner fold. No GPU; features extracted on CPU
(unsupervised, so lockbox images were processed but never scored).

| run | AUC [CI] | rep | first-visit | ΔAUC vs age |
|---|---|---|---|---|
| E04_dino_pca10_lr | 0.656 [0.585–0.723] | 0.647 | 0.669 | −0.001 [−0.072, +0.071] |

**Interpretation:** generic frozen encoders add nothing over age (same result as frozen ResNet-18
in E00). Routes that depend on frozen ImageNet-style features look exhausted. Fine-tuning and
physical-scale map inversion remain untested.

## E05 — extra indices from the raw Pentacam export (`prepare_raw.py`, `exp05_raw.py`)

The raw export (`Raw_Combined_IdsMesh.xlsx`, 11,760 rows) matches all 944 cohort scans on patient
id, eye and exam time (not date: the export has day/month swaps). Its shared numerics agree with
the cohort. Extra columns used: TKC grade (ordinal; 74 free-text annotations → NaN), pupil and
pachymetric-minimum positions, RSagMin, axis sin/cos. `KC diagnosis`, CXL, keratoplasty and
notes are never used.

| run | AUC [CI] | rep | first-visit | ΔAUC vs age |
|---|---|---|---|---|
| E05_raw_extra_lr | 0.603 [0.535–0.669] | 0.604 | 0.633 | −0.052 [−0.096, −0.006] |

TKC alone: AUC 0.516 (no signal). The extra indices add nothing over age.

## E06 — pre-declared combinations (`exp06_combo.py`)

| run | AUC [CI] | rep | first-visit | ΔAUC vs age [CI] |
|---|---|---|---|---|
| E06a_fellow_dino_lr (E02 features + E04 PCA10 + age, L2 LR) | 0.712 [0.638–0.779] | 0.705 | 0.739 | +0.057 [−0.014, +0.126] |
| E06b_fellow_mono_lgbm (E02 features + age, monotone LightGBM) | 0.723 [0.661–0.785] | 0.726 | 0.748 | +0.068 [+0.000, +0.138] |

Bootstrap-seed check on the paired ΔAUC lower bound (seeds 0, 1, 2):
- E06b: +0.0003, −0.0034, −0.0003. **Sign flips, so criterion 2 is not reliably met.**
- E06a: −0.0145, −0.0154, −0.0146. Fails.

Both pass criterion 1 on the pooled dev AUC (≥ 0.70). Neither reliably passes criterion 2. The
configurations were chosen on dev results, so the pooled AUC is optimistic. Total configurations
logged so far, counting repeats: 18 distinct runs (E00 ×7, E01 ×3, E02 ×2, E03 ×3, E04 ×1, E05 ×1,
E06 ×2). The lockbox is **not scored**, because no candidate has passed criterion 2 robustly.

## E07 — all-data baselines (nested run stopped by the user)

Two valid runs on all 944 scans were logged (`E07_age_all`, `E07_E06b_all`); their AUCs are in
`results/experiments.csv`. The nested-selection run (`E07_nested_select_all`) was stopped by the user
before it finished and is NOT logged. The user then asked to restrict work to image-based directions.

## E08 — IMAGE-ONLY (DINOv2 on the four maps, no age, no numerics), all 944 scans

| run | AUC [CI] | rep | first-visit |
|---|---|---|---|
| E08_dino_imageonly_pca5 | 0.563 [0.500–0.623] | 0.560 | 0.572 |
| E08_dino_imageonly_pca20 | 0.559 [0.505–0.614] | 0.552 | 0.578 |

**Interpretation:** frozen generic image features carry little signal on their own (lower CI bound
≈ 0.50). The age-free image signal is weak.

## E09 — IMAGE + RAW NUMERIC fusion, all 944 scans (`exp09_fusion.py`)

Image: DINOv2-small, 4 maps, per-map PCA(5) fitted in-fold. Raw: the 55 Pentacam numerics incl. age.

| run | AUC [CI] | rep | first-visit | ΔAUC vs age, all 944 [CI] |
|---|---|---|---|---|
| Age only, all 944 (reference) | 0.661 [0.598–0.723] | — | — | — |
| E09a concat → LR | 0.662 [0.607–0.716] | 0.654 | 0.690 | +0.002 [−0.058, +0.062] |
| E09b concat → LightGBM | 0.664 [0.602–0.724] | 0.656 | 0.681 | +0.003 [−0.069, +0.076] |
| E09c late fusion (stack) | 0.635 [0.577–0.696] | 0.624 | 0.674 | −0.025 [−0.075, +0.024] |

**Interpretation:** adding the DINOv2 map features to the Pentacam numerics does not improve on age
alone. Concatenation and late fusion both come out level with or below age. The images add no
signal beyond what the numerics and age already give.

## LOCKBOX (one-shot, 2026-11-06) — spent

Fitted on all 713 dev scans (seed 2026), scored once on the 231 lockbox scans (96 patients, 42
positive scans). Candidates were declared in PROTOCOL.md before scoring (commit `2cc8d6a`).

| candidate | lockbox AUC [CI] | ΔAUC vs age [CI] |
|---|---|---|
| E02_fellow_lr | **0.693** [0.591–0.785] | +0.018 [−0.064, +0.108] |
| E06b_fellow_mono_lgbm (primary) | 0.686 [0.589–0.785] | +0.010 [−0.094, +0.111] |
| E06a_fellow_dino_lr | 0.641 [0.529–0.746] | −0.036 [−0.143, +0.061] |
| age-only LR (reference) | 0.673 | — |

**Result: no candidate reaches 0.70 on the lockbox, and none beats age significantly.** The headline
lockbox AUC is 0.693 (E02, best of the three). The dev-set 0.72 did not hold up. The lockbox is now
spent; further claims need a new, untouched cohort (the protocol forbids re-drawing it).

## E11 — visit + history from the full raw exam record (separate claim)

`prepare_history.py` rebuilds the raw exam record: 11,704 Pentacam exams for 2,624 patients. The
record is two stacked exports: rows < 10125 have month-first text dates and day/month-swapped Excel
datetimes. Index dates match the cohort for 944/944 scans. Subset H holds scans with a prior exam of
the same eye ≥ 91 days earlier, within a 3-year look-back: 481 scans, 252 eyes, 203 patients and only
52 positives. Because having a history is itself a proxy for follow-up (positive rate 10.8% with a
history vs 24.8% without), the history models are evaluated only within H and are never pooled with
scans that lack one.

| run (subset H, n=481) | AUC [CI] | ΔAUC vs age on H |
|---|---|---|
| E11_age_lr_H | 0.604 [0.517–0.690] | — |
| E11a history LR (Theil–Sen slopes of KMax, K2F, K2B, Pachy min; time since last) | 0.599 | −0.004 [−0.092, +0.089] |
| E11b same, monotone LightGBM | 0.626 | +0.023 [−0.071, +0.107] |
| E11c + fellow-eye slopes | 0.589 | −0.013 [−0.096, +0.076] |

**Interpretation:** prior-exam slopes do not predict referral beyond age, even in this subset. With
52 positives the CI is ±0.09, so this is underpowered, not proof of no effect. The earlier paper's
0.75 on serial exams is not reproduced under patient-grouped CV.

## E10 — spatial image features (pre-declared closing image test, all 944)

DINOv2 on the full uncropped map: 16×16 patches pooled to a 4×4 grid, 4 maps, PCA(10) per map
fitted in-fold. Plus a colour (HSV) 4×4 grid, joined with the 55 numerics.

| run | AUC [CI] | vs age 0.661 |
|---|---|---|
| E10a spatial DINOv2 + numerics, LR | 0.663 | ≈ 0 |
| E10b spatial + colour + numerics, LightGBM | 0.664 | ≈ 0 |

| E10c spatial + numerics + fellow-eye image difference, LR | **0.697** [0.644–0.746], first 0.710 | +0.036 [−0.026, +0.101] |
| E10d late-fusion stack | 0.631 | below age |

The fellow-eye image difference is the first image signal: E10c vs E10a gives +0.033 [−0.004, +0.074].
The maps carry nothing on their own, but the difference between the two eyes' maps does.

## E12 / E13 — ensembles on all 944 scans (outer CV, same folds)

| run | AUC [CI] | rep mean | first-visit | ΔAUC vs age (3 bootstrap seeds) |
|---|---|---|---|---|
| E07_age_all (reference) | 0.661 [0.598–0.723] | 0.661 | 0.680 | — |
| E07_E06b_all | 0.697 | 0.705 | 0.727 | +0.037 [−0.024, +0.093] |
| E12a E02 fellow LR | 0.684 | 0.676 | 0.716 | +0.023 [−0.029, +0.073] |
| E12b rank-avg E02 + E06b, Platt-recalibrated in-fold | **0.701** [0.642–0.759] | 0.697 | 0.732 | +0.040 [−0.010, +0.089], P=0.94 |
| **E13 rank-avg E12b + E10c** | **0.719** [0.667–0.768] | 0.711 | **0.743** | **+0.058 [+0.009, +0.109]**, stable across seeds, P=0.99 |

E12b is calibrated: mean prediction 0.181 vs prevalence 0.177, slope 1.15, Brier 0.135 (age 0.140).
Its net benefit is at or above age at every threshold, and above it from 0.25 up.

**E13 is the first run that meets both dev criteria: AUC ≥ 0.70 and ΔAUC vs age with a CI that
excludes 0.** Caveats, all of which must be reported:
- E13 was chosen after seeing E12b and E10c, so it is post-hoc. About 45 configurations have now
  been logged in total.
- The lockbox is spent, so this is a CV estimate on all data, not a held-out score. The user's
  external validation set is the confirmatory test.

**Where the signal lives (has_fellow strata).** The flag alone gives AUC 0.58 (positive rate 0.24
with a same-day fellow scan vs 0.14 without), a workflow confound. Within strata:

| stratum | n / pos | E13 | age | ΔAUC |
|---|---|---|---|---|
| same-day fellow scan | 360 / 86 | **0.785** | 0.680 | **+0.106 [+0.029, +0.185]** |
| no fellow scan | 584 / 81 | 0.620 | 0.641 | −0.021 [−0.077, +0.031] |

The gain is real inter-eye signal, not the flag. When both eyes are imaged at the visit, comparing
them (numerics and maps) predicts referral well beyond age. Without the fellow eye, nothing beats age.
Clinical framing: the model is a **bilateral-visit** tool. For single-eye visits, use age (or say
that no model helps).

## E14 — bilateral-visit specialist with mirror-aligned map differences

Left-eye 4×4 cell grids are flipped into the right-eye layout before own − fellow differences are
taken; E10c had subtracted unaligned PCA scores. The models are trained and evaluated only on the
360 scans (131 patients, 86 positive) with a same-day fellow scan.

| run (bilateral subset) | AUC [CI] | first-visit | ΔAUC vs age (0.671) |
|---|---|---|---|
| E14a fellow numerics, no flag | 0.724 [0.644–0.803] | 0.737 | +0.053 [−0.031, +0.138] |
| E14b + own DINOv2 + mirrored differences | 0.737 [0.663–0.813] | 0.755 | +0.068 [−0.019, +0.155] |
| E14c mean(E14a, E14b) | 0.747 [0.669–0.824] | 0.765 | +0.077 [−0.008, +0.160] |
| routed: E14c for bilateral scans, age otherwise (all 944) | 0.712 [0.653–0.768] | — | +0.051 [+0.007, +0.094] |

The mirrored maps add +0.015 [−0.034, +0.067] over the numerics. The specialist loses to E13
(trained on all 944 scans), which scores 0.785 on the same scans. Training on all scans gives better
estimates of the age and own-eye effects. Conclusion: keep all-data training and add the mirrored
differences to it (E15).

## E15 — mirrored map differences in the all-data model

| run (all 944) | AUC [CI] | first-visit | ΔAUC vs age | bilateral AUC | single-eye AUC |
|---|---|---|---|---|---|
| E15a E02 numerics + own DINOv2 + mirrored diffs, LR | 0.685 [0.626–0.740] | 0.710 | +0.024 [−0.039, +0.085] | 0.729 | 0.598 |
| E15b rank-avg E12b + E15a | 0.705 | 0.736 | +0.044 [−0.007, +0.096] | 0.755 | 0.609 |
| E15c rank-avg E12b + E10c + E15a | 0.712 | 0.739 | +0.051 [−0.001, +0.107] | 0.776 | 0.615 |
| E13 rank-avg E12b + E10c (best) | **0.719** | **0.743** | **+0.058 [+0.009, +0.109]** | **0.785** | 0.620 |

In the all-data model, mirror alignment does not help: E15a 0.685 vs E10c 0.697, and adding E15a to
E13 lowers it. The small mirror gain seen in E14 does not carry over. **E13 stays the final model.**
Variants of the inter-eye approach now sit at 0.70–0.72; further variants would mostly add to the
configuration count. Weak spot shared by every model: single-eye scans (≈ 0.60–0.62 vs age 0.641).

## E16 — fully nested classifier selection on all 944 scans (the clean headline)

The outer folds are the harness folds (5×5, patient-grouped), the same as every other all-data run.
Inside each outer training fold, a 4-fold patient-grouped inner CV scores every candidate in a
pre-declared pool. The best candidate is refit on the outer training fold and scored on the outer
test fold. Nothing is chosen by looking at outer results, so this removes the post-hoc choice of
E13. Choices are in `results/nested16_choices.csv`.

| | pooled OOF AUC [CI] | first-visit | mean within-fold AUC (25 folds) | Δ vs age, pooled | folds beating age |
|---|---|---|---|---|---|
| age only (E07) | 0.661 [0.598–0.723] | 0.680 | 0.669 | — | — |
| **E16a pool A, single eye** | 0.644 [0.585–0.700] | 0.660 | 0.647 | −0.017 [−0.061, +0.030] | 1/25 |
| **E16b pool B, + same-day fellow** | **0.695** [0.640–0.750] | **0.723** | **0.707** | +0.035 [−0.019, +0.086], P=0.89 | **19/25** |
| fixed E13 (for comparison, post-hoc) | 0.719 | 0.743 | 0.715 | +0.058 | 19/25 |

Pools:
- A: age LR, numerics LR, numerics LightGBM, images+numerics LR, images+numerics LightGBM.
- B: A plus E02, E06b, E12b, E10c_fast and E13.

Choices:
- Pool A picked age only in 15 of 25 folds, and an image model in 9.
- Pool B never picked a single-eye model: E06b 14, E13 6, E12b 5.

E16b's per-repeat AUC (0.654) is far below its pooled AUC. The fold-specific winners have different
score scales: E06b is uncalibrated (mean ≈ 0.42), while E12b and E13 are calibrated. Pooling the
mixed scales penalises the per-repeat AUC. The mean within-fold AUC is the fair figure for a
selection procedure: 0.707.

**Interpretation:**
- Under fully nested selection, single-eye information does not beat age: 0.644–0.647 vs 0.661–0.669.
- With the fellow eye, the selection procedure reaches 0.695 pooled and 0.707 within fold. It beats
  age in 19 of 25 folds, but the paired-bootstrap CI includes 0.
- The post-hoc E13 (0.719) is about 0.01–0.02 optimistic relative to the honest nested 0.707.

## Frozen models for external validation

`freeze_models.py` fits every candidate on all 944 scans and writes `rethink/frozen/models.pkl`
(gitignored; trained on patient data):
- age only;
- E02 (C=0.03);
- E06b (100 trees, 2 leaves);
- E12b (E02 + E06b quantile average, Platt-calibrated on inner OOF);
- E10c (C=0.3, DINOv2 PCA fitted on all 944);
- **E13**, the final model: mean quantile of E12b and E10c. It is a ranking score; use E12b for
  calibrated risk.

`score_external.py FILE [--mapping cols.csv] [--images DIR]` prints aggregates only. The input
format is in its docstring. Output:
- AUC with a patient-cluster CI and paired ΔAUC vs age for every model;
- first-visit AUC;
- AUC on bilateral vs single-eye visits;
- E12b calibration.

E10c and E13 need the four map images (`DIR/<image_stem>_<map>.jpg`, same crop as
`cropped_output_v2`) and the `facebook/dinov2-small` weights.

Checks:
- Tested in an isolated copy holding only the code, `models.pkl` and `numeric_features.txt`, with
  no cohort data: 143 scans, 60 patients, about 1 minute including image extraction.
- Image features re-extracted by the scorer match the training features exactly (max |Δ| = 0.0).
- Apparent AUCs on training scans are inflated by design: E13 about 0.90, E10c 0.91 vs its CV
  0.697. On a new cohort, expect the CV values: E13 about 0.72, E12b about 0.70.

Supporting changes, behaviour-preserving:
- the data loads in `exp04_dino`/`exp10_spatial` are lazy;
- `harness.numeric_features()` falls back to `numeric_features.txt` (names only) when the cohort is
  absent.

## Reviewer analyses on the all-data OOF predictions (`analysis_e07.py` → `results/analysis_e07.txt`)

E07_E06b_all vs E07_age_all, all 944 scans.
- **Age strata:** <25 E06b 0.617 vs age 0.476; 25–29 0.480 vs 0.513; ≥30 0.664 vs 0.600 (20 pos).
  Below 25 the fellow-eye model carries real within-stratum signal that age cannot.
- **TKC stage:** none/possible 0.667 vs 0.567; KC1 0.727 vs 0.628; KC2 0.713 vs 0.700; KC3–4 0.675
  vs 0.786. The model helps most in early disease. In advanced eyes, age dominates.
- **Operating point** (threshold chosen on the other half of patients): E06b sens 0.83, spec 0.49,
  PPV 0.26, NPV 0.93. Age: 0.83 / 0.47 / 0.25 / 0.93.
- **Calibration:** E06b is mis-calibrated (mean prediction 0.42 vs prevalence 0.18, from
  scale_pos_weight; Brier 0.200 vs 0.140 for age). As a result, its net benefit falls below age at
  thresholds ≥ 0.15. Recalibration is needed before any clinical use; see E12b.
- **Time to CXL:** positives 0–6 months before CXL, E06b 0.716; 6–12 months 0.653; 12–24 months
  0.718; > 24 months 0.653 (11 positives).
- **Secondary endpoint**, KMax rise ≥ 1 D at an exam 6–15 months later and before CXL: 344 scans,
  40 progressed, only 8 of them also positive for CXL. E06b 0.574, age 0.522. Single-visit data
  barely predicts objective progression.

## Harness / protocol notes
- `harness.summarise` now returns NaN for the first-visit AUC when a subset has no first-visit scans
  (needed for E03). Earlier numbers are unchanged.
- E00 was rerun during setup, which appended a second copy of each E00 row to `experiments.csv`.
  Count them once each.
