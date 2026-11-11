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

## E17 — single-visit raw data squeezed from the composites (single eye)

New information extracted from the full "4 Maps Selectable" composites (`data/image_scans/`, one per
scan, all 944). The tabular raw exports hold nothing beyond the 55 numerics plus the E05 extras.

1. **Panel OCR** (`ocr_panel.py`): template-matching OCR of the printed values. It is 100% exact
   on held-out known fields (pachymetry at pupil / apex / thinnest point, thinnest y, KMax, pupil
   x/y; 472 images each). New fields: KMax location, cornea volume, KPD, chamber volume / angle /
   depth, pupil diameter, IOP correction. Finding along the way: the export flips the x sign for
   right eyes relative to the screen.
2. **Physical map decoding** (`decode_maps.py`): each image's own 31-step colour bars invert the
   four maps to µm / D. Checks:
   - the decoded thinnest band matches the printed thinnest pachymetry to the 20 µm quantum (mean
     difference 0.1 µm);
   - the scale is 28.9 / 27.3 px/mm, calibrated on the printed thinnest location (r = 0.99 / 0.95);
   - under 1% of central pixels are undecodable.
   Clinical indices derived from the decoded maps: back and front elevation at the thinnest point,
   max elevation in the central 4 mm, I-S curvature and thickness, cone area, area of back elevation
   ≥ +20 µm, plus ring/sector means.

| run (all 944, single eye) | AUC [CI] | first-visit | ΔAUC vs age |
|---|---|---|---|
| E17a 55 numerics + panel + map indices, LR | 0.612 [0.553–0.673] | 0.646 | −0.048 [−0.099, +0.003] |
| E17b same, LightGBM | 0.630 [0.571–0.691] | 0.660 | −0.030 [−0.098, +0.039] |
| E17c + ring/sector PCA, LR | 0.591 [0.532–0.652] | 0.619 | −0.070 [−0.130, −0.012] |
| E17d nested selection, pool A + the three | 0.644 [0.585–0.700] | 0.660 | −0.017 (identical to E16a) |

The 19 new features have univariate AUCs of 0.50–0.56. The best are anterior-chamber depth, volume
and angle (0.54–0.56), which correlate with age (r ≈ −0.2). The nested selection never picked a
raw-data candidate. Mean inner AUC: age 0.665 > images + numerics LightGBM 0.647 > raw LightGBM
0.626.

**Interpretation:** the single-visit, single-eye information is exhausted. The data used are:
- the 55 numerics, the 10 export extras and 11 printed panel values;
- the physically decoded maps and their clinical indices;
- DINOv2, spatial and colour image features.
None of it predicts referral beyond age. A soon-to-be-treated eye looks like any other keratoconic
eye of the same age at one visit. The referral decision depends on progression (seen over time)
and on the patient as a whole (both eyes), not on a single-eye snapshot.

## E18 — single eye trained on more labelled exams (`prepare_extra.py`, `exp18_augment.py`)

The raw export carries the 55 numerics and the CXL fields for exams without images. Checks against
the cohort: the label agrees for 944/944 scans; the numerics are identical, apart from the sign of
`Pachy Min Pos X:` for right eyes, which is harmonised; age is exactly scan year − birth year.

Extra training exams: 1,487 pre-CXL exams of CXL-clinic patients that are not cohort scans, 299 of
them positive.
- 587 are from 269 patients outside the cohort; they are always used for training.
- 900 are other exams of cohort patients; they are used only when that patient is in the training
  fold.

Patients outside the clinic sheet are excluded: all of them carry label 0, so their label does not
mean the same thing. Evaluation is unchanged (cohort scans, same folds).

| run (all 944, single eye) | AUC [CI] | first-visit | ΔAUC vs age |
|---|---|---|---|
| E18a numerics LR, status-OK extras (558) | 0.652 [0.595–0.709] | 0.677 | −0.009 [−0.051, +0.032] |
| E18b numerics LightGBM, status-OK extras | 0.613 [0.553–0.671] | 0.645 | −0.047 [−0.103, +0.010] |
| E18c numerics LR, all extras (1,487) | 0.650 [0.589–0.711] | 0.670 | −0.011 [−0.045, +0.022] |
| E18d numerics LightGBM, all extras | 0.626 [0.566–0.684] | 0.652 | −0.035 [−0.082, +0.012] |
| E18_age_aug age only, all extras (control) | 0.663 [0.600–0.724] | 0.680 | +0.002 [−0.002, +0.005] |
| E18e nested selection, pool A + the five | 0.635 [0.576–0.692] | 0.659 | −0.026 [−0.067, +0.017] |

E18e chose age 15 times out of 25; it chose an augmented candidate once.

**Interpretation:** the extra exams nearly double the positives seen in training (167 → 466 at most),
but the numeric models stay at or below age.

Side finding: the `.DCM` files in `image_scans/` are 5-page Pentacam PDFs (4 Maps Refractive, Cataract
Pre-OP, Holladay EKR, Belin/Ambrósio with BAD-D). They exist for only 380 cohort scans, and 98% of those
are negatives. They were exported for non-referred patients, so their availability almost reveals the
label. They are not used as features.

## E19 — self-supervised image representation (`decode_all.py`, `exp19_ssl.py`)

1. **Decoding:** every "4 Maps Selectable" composite in `image_scans/` (2,259 exams: 944 cohort, 650
   other visits of cohort patients, 665 of 199 patients outside the cohort) is decoded to physical maps.
   The cohort scans' decoded maps match `decode_maps.py` exactly.
2. **Pretraining:** a small CNN masked autoencoder is pretrained on the four maps.
   - Inputs: the normalised maps plus visibility channels.
   - Pretext: 50% of 8×8 patches are hidden independently per map, so the net must infer one map from
     the others.
   - Settings: 40 epochs, shifts of up to 3 px as augmentation.
   - A fresh encoder is trained for every outer fold, on images of non-cohort patients and outer-training
     patients only (~1,900 images); test-fold patients' images are never seen.
   - Reconstruction loss fell from ~1.0 to 0.19–0.31 in 24 of 25 encoders; one ended at 0.63.
3. **Downstream:** the encoder is used frozen as a 64-d embedding in an L2 LR with nested C.

| run (all 944, single eye) | AUC [CI] | first-visit | ΔAUC vs age |
|---|---|---|---|
| E19a age + SSL embedding, LR | 0.614 [0.549–0.671] | 0.646 | −0.048 [−0.092, −0.009] |
| E19b SSL embedding + 55 numerics, LR | 0.628 [0.569–0.681] | 0.671 | −0.033 [−0.083, +0.014] |
| E19c nested selection, pool A + both | 0.644 [0.585–0.700] | 0.660 | −0.017 [−0.061, +0.030] (= E16a) |

E19c never chose an SSL candidate (age 15, images + numerics LightGBM 5, images + numerics LR 4,
numerics LightGBM 1).

**Interpretation:** a representation learned from the maps themselves does no better than ImageNet
ResNet, DINOv2, colour grids or decoded indices. Adding it to age makes things significantly worse
(more noise dimensions). The map content does not carry referral information beyond age.

## E20 / E21 — bilateral single visit with fellows recovered from the raw export (`exp20_bilateral.py`, `prepare_raw_all.py`, `exp21_bilateral_clean.py`)

Three changes over E02/E13:
1. **Recovered fellows.** The other eye's same-day exam is taken from the raw export when it has no
   image.
2. **Full features.** All 54 numerics + 12 indices for own eye, fellow and own − fellow, plus
   worse-eye flags. E02 used 12 indices only.
3. **Bilateral training augmentation.** The extra labelled exams, each with its own fellow, are added
   to training rows only.

Candidates are LR and age-monotone LightGBM, each on the engineered and full feature sets, with and
without augmentation. Nested selection works as in E16. Strata are by E20 fellow source:
- image: 360 scans, 23.9% positive;
- raw: 268 scans, 19.0% positive;
- none: 316 scans, 9.5% positive.

**E20 fellow definition.** E20 takes fellows from `extra_exams.parquet`, which keeps only pre-CXL
exams. Of the 316 scans without a fellow, 273 have a fellow eye cross-linked before or at the visit,
and 1 has an undated CXL. So `has_fellow` effectively means "the other eye is not yet treated".
- This is past information, known at the visit (no future leak).
- It comes from the clinical sheet, not the scan.
- The image fellows of E02/E13 behave the same way, because the cohort excludes post-CXL scans.
- **User decision 2026-11-10:** fellow treatment status as of the visit is a valid input.

**E21** uses a label-free definition: any same-day exam of the other eye in the raw export, preferring
status OK. This gives 941/944 scans a fellow, and the flag alone has AUC 0.502. The old fellow
candidates are not in its pool.

| run (all 944) | AUC [CI] | first-visit | ΔAUC vs age [CI] | image / raw / none (age) |
|---|---|---|---|---|
| E20_b_eng_lgbm | 0.774 [0.724–0.822] | 0.777 | +0.114 [+0.055, +0.174] | 0.757 / 0.801 / 0.580 (0.680 / 0.719 / 0.592) |
| E20_b_full_lgbm | 0.772 [0.720–0.820] | 0.777 | +0.112 [+0.049, +0.175] | 0.766 / 0.781 / 0.546 |
| E20_b_full_aug_lgbm | 0.771 [0.717–0.820] | 0.781 | +0.110 [+0.049, +0.174] | 0.757 / 0.793 / 0.577 |
| **E20n nested (pool B + 9)** | **0.761** [0.710–0.812] | 0.775 | **+0.101 [+0.042, +0.160]** | 0.753 / 0.777 / 0.559 |
| E21_b_full_lgbm | 0.743 [0.692–0.793] | 0.753 | +0.082 [+0.015, +0.151] | 0.764 / 0.784 / 0.671 |
| E21_b_full_aug_lr | 0.741 [0.681–0.795] | 0.765 | +0.080 [+0.018, +0.145] | 0.778 / 0.755 / 0.691 |
| E21_b_full_aug_lgbm | 0.744 [0.689–0.797] | 0.759 | +0.083 [+0.015, +0.152] | 0.752 / 0.779 / 0.709 |
| **E21n nested (pool A + 8)** | **0.725** [0.668–0.778] | 0.749 | **+0.064 [+0.003, +0.130]** | 0.733 / 0.775 / 0.674 |

All 17 E20/E21 runs are logged. The other E20 rows are 0.730–0.765 and the other E21 rows are
0.681–0.740.

Nested choices:
- E20n: eng_aug_lgbm 10, full_lgbm 7, eng_lgbm 4, eng_aug_lr 2, b_E13 1, full_aug_lgbm 1.
- E21n: full_aug_lgbm 10, full_aug_lr 6, full_lgbm 4, eng_aug_lgbm 3, eng_lgbm 2.

**Interpretation:** both eyes at one visit predict referral well beyond age.
- Clean selection beats age with the CI excluding 0 under either fellow definition.
- With label-free fellows, the gain holds in every stratum, including scans whose fellow was already
  treated (0.67–0.71 vs age 0.59).
- E20's extra ~0.035 comes from knowing the fellow's treatment status.
- The project continues on this path only (bilateral single visit); single-eye and history scripts
  are archived.

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

- 2026-11-10: the last 7 rows of `experiments.csv` (E00_*, git 07455a2) are an accidental re-run of
  E00 caused by importing `exp00_baselines.py` (it then had no `__main__` guard). Their numbers equal
  the original E00 rows exactly. They are kept, per the never-delete rule, but are not new
  configurations: 95 rows, 88 distinct configurations.
- `harness.summarise` now returns NaN for the first-visit AUC when a subset has no first-visit scans
  (needed for E03). Earlier numbers are unchanged.
- E00 was rerun during setup, which appended a second copy of each E00 row to `experiments.csv`.
  Count them once each.

## E22 — bilateral single-visit programme on the GPU machine (`gpu_run/DESIGN.md`)

### Set-up and environment (2026-11-10)

- Branch `clean-rethink` from `a5a36c3`. Hardware: 7 × NVIDIA GeForce RTX 2080 Ti (11 GB, sm_75; an eighth card,
  GPU 3, reports "Unknown Error" and is not used), driver 560.35.03 (CUDA ≤ 12.6), 40 CPU cores, 251 GB RAM.
- The Python 3.12 environment named in the brief did not exist on the machine (system Python 3.10 without packages;
  the repo's `.venv` has no interpreter). A new venv `.venv-gpu/` (untracked, in `.git/info/exclude`) was built with
  uv: Python 3.12.13, torch **2.14.1+cu126** (the newest build for this driver; the brief said 2.15), lightgbm 4.7.0,
  xgboost 3.4.1, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4, pyarrow 26.0.1, scipy 1.18.1, pillow 13.0.0,
  transformers 5.22.2, plus pydicom and pymupdf (to read the DCM-wrapped PDFs). All scripts run as
  `.venv-gpu/bin/python -I rethink/<script>.py` from the repo root.
- **E00 smoke test (reproduction, not new configurations):** `exp00_baselines.py` appended 7 rows (git `a5a36c3`)
  whose numbers equal the E00 table exactly (age 0.656 [0.584–0.724], first visit 0.680; clin_lr 0.624; clin_xgb
  0.609; noage_xgb 0.490; mm_legacy 0.575; mm_noage 0.448; img_only 0.468). `experiments.csv`: 102 rows.

### E22 data (`prepare_fellow_tx.py`, `decode_hr.py`, `decode_dcm.py`, `zernike.py`)

1. **Fellow treatment block H** (`prepare_fellow_tx.py` → `rethink/data/eye_tx.parquet`). CXL / keratoplasty
   dates per patient-eye from the clinic sheet (the sheet that defines the label). Features use the other eye's
   dates strictly before the visit (`<`): `fellow_cxl_before_visit`, `years_since_fellow_cxl`,
   `fellow_kp_before_visit`, `fellow_cxl_unknown` (undated CXL; C1), `fellow_tx_missing` (patient not in sheet).
   All 944 cohort patients are in the sheet. 1,516 patient-eyes: 643 dated CXL, 5 undated, 79 dated KP, 1 undated.
   - Cohort: fellow treated before the visit for 244/944 scans; their positive rate is 5.7% (vs 17.7% overall).
     `fellow_cxl_before_visit` alone: AUC 0.394 (i.e. 0.606 inverted), as expected from E20. 31 scans have a fellow
     CXL on the visit day; they count as not yet treated. 1 cohort scan has an undated fellow CXL.
2. **High-resolution decoding** (`decode_hr.py` → `decoded_hr.npy`, 2,260 × 4 × 145 × 145 float16). Same
   decoder and per-image 31-step palettes as `decode_maps.py`, 2-px block medians (was 5 px / 58 × 58). Palette
   check passes on 2,260/2,260 composites for all 4 maps; ~72% of each 290 × 290 box is decodable (the circular
   map minus overlays). Row-aligned with `decoded_all_meta.parquet` (asserted).
3. **DCM pages** (`decode_dcm.py`, pretraining only, C4). The 1,590 `4 Maps Select.DCM` files are 5-page PDFs of
   1200 × 910 screenshots; every one has a JPG composite of the same exam. The 188 `c0000*.DCM` files are unnamed
   PDFs that cannot be keyed to an exam; they are not used. Page geometry comes from the printed axis labels
   (fixed across files); each map is resampled onto the composite grid.

   | page | maps | palette check fail | twin check pass | decision |
   |---|---|---|---|---|
   | 0 4 Maps Selectable | — | — | — | skipped: duplicate of the JPG composite |
   | 1 4 Maps Refractive | curvature, elev. front, thickness, elev. back | 0% | 100% / 96.4% / 100% / 87.0% | kept |
   | 2 Cataract Pre-OP | curvature front, thickness (TCRP not used) | 0% | 100% / 99.9% | kept |
   | 3 Holladay EKR | EKR power | **100%** (relative 15-step scale) | — | **dropped** |
   | 4 Belin/Ambrósio | elev. front/back, thickness | **100%** (continuous elevation colour bar) | — | **dropped** |

   Twin check: the decoded map must agree with the same exam's composite (central 4 mm median |Δ| ≤ one palette
   step); maps failing it are left empty. Page 1 elevation uses one colour bar for ±37.5 µm (2.5 µm steps) and
   ±75 µm (5 µm) scales; both are tried. Result: 3,180 extra page samples (1,590 × 2), each a partial or full
   4-channel map of an exam already in the corpus (different rendering, finer elevation steps, wider thickness
   zone on page 2).
4. **Encoder Z, Zernike** (`zernike.py` → `emb_Z.npy`, 2,260 × 135): OSA Zernike up to order 8 (45 terms) of
   front elevation, back elevation and thickness on the central 6 mm (28.9 px/mm). All 2,260 fitted; residual RMS
   1.3 µm (elevations) and 5.4 µm (thickness), i.e. at the colour-quantisation noise level.

### E22 encoders: compute choices, fixed before training (`pretrain_jepa.py`, `CFG`)

Written and committed before any full pretraining run. Labels are never read; the last-epoch checkpoint is used;
nothing is selected on referral AUC.
- **Budget check:** the smoke runs measured ≈ 0.25 s per L step (batch 128, 2 global 128 px + 6 local 64 px views)
  on one 2080 Ti in fp16. Full runs: L 300 epochs × 42 steps ≈ 1 h; X 300 × 12 steps and T 300 × ~58 steps are
  smaller. All three fit at the design's settings, each on its own GPU (0, 1, 2), so **X and T do not share a
  backbone** and no epochs were cut.
- **Backbone:** ResNet-18 layout (2-2-2-2 basic blocks), widths 48/96/192/384, 3 × 3 stride-2 stem, no max-pool,
  global average pool → **384-d embedding** (design cap). ViT-S/8 was not chosen: ~5.4k training maps are too few
  for a ViT from scratch. Projector 384–1024–1024–128 (BN, ReLU); X/T predictor 128(+1 for T's gap)–512–512–128.
- **Input:** 4 decoded maps (145 × 145, 2-px cells, left eyes mirrored) normalised with `exp19_ssl.NORM`, plus 4
  validity masks (8 channels). Corpus 5,440 samples = 2,260 composites + 3,180 DCM page renderings.
- **Views / augmentation:** global crops 128 px (area 0.4–1.0), local crops 64 px (0.1–0.35), aspect 3/4–4/3,
  rotation ±10°, Gaussian noise σ = 0.05 (normalised units) on decoded pixels, channel dropout p = 0.15 (≥ 1 map
  kept), two 20% patch drop-outs at p = 0.5. No flips, no colour jitter.
- **Optimisation (all encoders):** AdamW, weight decay 0.05, 10 warm-up epochs then cosine to 0, **300 epochs**,
  λ(SIGReg) = 0.05, 256 slices, 17 knots on [−5, 5], fp16 autocast, seed 2026.
  - L: batch 128, lr 1e-3, 2 global + 6 local views (LeJEPA defaults).
  - X: batch 64, lr 5e-4, 2 global views per eye; units = the 786 same-day bilateral composite visits.
  - T: batch 64, lr 5e-4, 2 global views per visit; units = every ordered same-eye visit pair with a strictly later
    date; the gap in years / 3 is a predictor input.
  - X and T show each exam as its composite (p = 0.5) or one of its DCM page renderings.
- **Embedding:** full map resized to 128 px, no augmentation, pooled backbone output (384-d), saved per composite
  as `rethink/data/emb_{L,X,T}.npy`. Checkpoints every 10 epochs in `rethink/results/ssl/<enc>/` (resumable);
  training curves in `curve.csv` there, with label-free embedding diagnostics (effective rank) in `embed_stats.csv`.

### E22 encoders: training record (label-free)

All three ran the full 300 epochs at the fixed configuration on separate 2080 Ti cards (L 54 min, X 9 min,
T 46 min); last-epoch checkpoints embedded every composite. Loss (prediction / SIGReg) epoch 1 → 300:
L 1.25 (0.16 / 21.9) → 0.12 (0.043 / 1.50); X 1.17 (0.31 / 17.5) → 0.26 (0.13 / 2.62);
T 0.91 (0.28 / 12.8) → 0.08 (0.017 / 1.29). Embedding effective rank (of 384): L 13.5, X 11.0, T 34.9 — no
collapse. Curves: `rethink/results/ssl/<enc>/curve.csv` (gitignored with the checkpoints; copied in the bundle).

Supporting tables for E22 blocks: `panel_all.py` (P and M for all 2,260 composites; equal to E17's
`rawmap_features` on all 944 cohort composites, 19/19 features 100%) and `dino_all.py` (encoder D: frozen
DINOv2-small mean-pooled patch tokens per map, 4 × 384 = 1,536 dims — above the 384 cap before in-fold PCA;
left eyes flipped; 2,259/2,260 complete, one composite has no crops). Cohort scans with a same-day fellow
composite (any composite, cohort or not): 563 / 944; with same-day fellow numerics: 941 / 944.

### E22 results (`exp22_bilateral.py`, `analysis_e22.py` → `results/analysis_e22.txt` and `analysis_e22_*.csv`)

All 944 scans, harness folds (5 × 5, seeds 2026–2030), paired cluster bootstrap. Full tables in
`results/analysis_e22.txt`.

**Headline — `E22n_nested`** (inner-CV selection over the 21 candidates per outer fold):
**AUC 0.780 [0.727–0.829]**, first visit 0.784, per-repeat 0.760 ± 0.013, mean within-fold 0.779.
- vs `E22_ref_single_num` (0.650 [0.593–0.708], first visit 0.683; primary): ΔAUC +0.131, CI over bootstrap seeds
  0/1/2 [+0.069, +0.190] / [+0.068, +0.195] / [+0.065, +0.201]; E22n beats ref in 25/25 outer folds.
- vs `E07_age_all` (0.661): +0.120, [+0.052, +0.185] / [+0.052, +0.185] / [+0.054, +0.188].
- Choices (25 folds): ens_D 7, bi_D_lgbm 5, bi_L_lgbm 4, ens_L 2, bnpm_lgbm 2, bi_D_lr, bi_Z_lgbm, bi_T_lgbm,
  ens_Z, bn_aug_lgbm 1 each. The ref chose num_lgbm 19, num_lr 6.

| candidate | AUC | first | Δ vs ref | | candidate | AUC | first | Δ vs ref |
|---|---|---|---|---|---|---|---|---|
| bn_lr | 0.751 | 0.757 | +0.101 | | bi_L_lr | 0.759 | 0.767 | +0.109 |
| bn_lgbm | 0.780 | 0.770 | +0.130 | | bi_L_lgbm | 0.790 | 0.786 | +0.140 |
| bn_aug_lr | 0.768 | 0.770 | +0.118 | | bi_X_lr | 0.745 | 0.750 | +0.095 |
| bn_aug_lgbm | 0.780 | 0.778 | +0.130 | | bi_X_lgbm | 0.781 | 0.773 | +0.131 |
| bnpm_lr | 0.745 | 0.760 | +0.095 | | bi_T_lr | 0.749 | 0.758 | +0.099 |
| bnpm_lgbm | 0.786 | 0.784 | +0.137 | | bi_T_lgbm | 0.798 | 0.788 | +0.149 |
| bi_Z_lr | 0.751 | 0.765 | +0.102 | | ens_Z | 0.780 | 0.789 | +0.131 |
| bi_Z_lgbm | 0.787 | 0.779 | +0.138 | | ens_D | 0.789 | 0.795 | +0.139 |
| bi_D_lr | 0.768 | 0.780 | +0.119 | | ens_L | 0.783 | 0.789 | +0.133 |
| bi_D_lgbm | 0.792 | 0.784 | +0.143 | | ens_X | 0.776 | 0.783 | +0.126 |
| | | | | | ens_T | 0.779 | 0.785 | +0.129 |

Every Δ vs ref has a CI excluding 0 (per-candidate CIs in `analysis_e22_candidates.csv`). Images add little over
numerics: best image candidate 0.798 (bi_T_lgbm, post-hoc) vs bn_lgbm 0.780; the nested headline (0.780) equals
the numeric-only bn_lgbm. The SSL encoders are not better than Zernike or frozen DINOv2.

**Label-free sensitivity / negative control.**
- `E22n_noH` 0.731 [0.676–0.784], first 0.758; vs ref +0.081 [+0.023, +0.138]; the treatment block H adds
  +0.049 [+0.020, +0.076].
- `E22_negctrl` (whole fellow from a random patient ±2 y of age, nested over 4 bn) 0.633 [0.573–0.688]; vs ref
  −0.018 [−0.072, +0.036]. The gain is the patient's own fellow eye, not population structure.

**Severity (KMax of the scanned eye; original four bins exact, plus > 55 D).**

| stage | n / pos | E22n | ref | age | Δ vs ref |
|---|---|---|---|---|---|
| ≤ 46.5 | 222 / 44 | 0.821 | 0.651 | 0.584 | +0.170 [+0.071, +0.269] |
| 46.5–48 | 101 / 13 | 0.843 | 0.580 | 0.552 | +0.257 [+0.053, +0.475] |
| 48–53 | 232 / 38 | 0.800 | 0.622 | 0.633 | +0.184 [+0.047, +0.317] |
| 53–55 | 102 / 20 | 0.607 | 0.634 | 0.607 | −0.026 [−0.168, +0.112] |
| > 55 | 287 / 52 | 0.753 | 0.706 | 0.795 | +0.048 [−0.067, +0.159] |

Interaction (gain at KMax ≤ 48 minus gain > 48): +0.125 [−0.007, +0.249]. The bilateral gain is in early and
moderate disease; above 53 D it is not shown, and above 55 D age alone does as well.

**Fellow strata** (mutually exclusive): fellow composite 552 / 94: E22n 0.759 vs ref 0.719 vs age 0.721;
fellow numerics only 147 / 59: 0.710 vs 0.586 vs 0.640; fellow treated before the visit 244 / 14: 0.611 vs 0.552
vs 0.516; no fellow 1 / 0. `has_fellow_image` alone 0.480.

**Age strata / first visit.** < 25 (380 / 95): 0.722 vs ref 0.521; 25–29 (196 / 52): 0.662 vs 0.429; ≥ 30
(368 / 20): 0.822 vs 0.529; first visit (523 / 125): 0.784 vs 0.683, Δ +0.101 [+0.054, +0.148]. The model
is not "age": it beats age-only within every age band.

**Operating point, calibration, DCA** (Youden threshold and Platt map fitted on the other folds): E22n sens
0.751, spec 0.632, PPV 0.306, NPV 0.922 (ref 0.665 / 0.516 / 0.228 / 0.878; age 0.857 / 0.443 / 0.249 / 0.935).
Raw E22n is uncalibrated (mean prediction 0.34 vs prevalence 0.177, from `scale_pos_weight`); after in-fold Platt:
slope 1.20, intercept 0.24, Brier 0.127 (ref 0.141, age 0.140). Net benefit is above age, ref and treat-all at
every threshold 0.10–0.40 (e.g. 0.20: 0.058 vs age 0.026, ref 0.029); ≈ 0 from 0.45.

**Leakage checks.** Block H alone 0.39 (= 0.61 inverted), as expected for "fellow already treated → less
referral". Encoder dimensions alone: max univariate |AUC − 0.5| ≤ 0.12 (medians 0.02–0.04). `has_fellow_image`
0.48. Negative control at chance vs ref. One flag to report, not a leak: the "fellow numerics only" stratum has a
40% positive rate (59 / 147) vs 17% with a fellow composite — whether the other eye's composite was exported is a
workflow difference correlated with referral. No candidate uses an image-availability flag except via
`has_fellow_image` (block I / PM), which alone is at chance; the numeric-only bn models reach the same AUC.

**Interpretation (SHAP, numeric bilateral LGBM on all 944, descriptive):** mean |SHAP| by family: fellow-eye
numerics 0.62, age 0.34, fellow treatment H 0.31, own eye 0.25, own − fellow 0.04. Top: age, years since fellow
CXL, Pachy Min / age, fellow D4mm Prog, fellow IHD, fellow KMax − Km. The signal is mostly the **fellow eye's
severity** and whether/when it was already treated, i.e. a patient-level picture at the visit.

**Configuration count:** `experiments.csv` now 127 rows; E22 added 25 (ref, 21 candidates, E22n, noH, negctrl)
→ 88 + 25 = **113 distinct configurations**.

### E22 freeze (`freeze_bilateral.py` → `rethink/frozen/bilateral.pkl`, `enc_L.pt`; `score_external.py`)

Inner selection once on all 944 scans (4-fold patient-grouped, seed 2026), winner refitted on all 944 (+ aug rows
if used): headline **bi_L_lgbm** (inner 0.799); fallback (a) without H **noH_bi_L_lgbm** (0.756); fallback (b)
numerics N only **noH_bn_lgbm** (0.740); ref num_lgbm (0.627); age LR. Encoder L weights saved.
`score_external.py` builds every block from raw inputs (fellow rows with empty y, `--tx` dates table, `--composites`
dir), scores whichever models the input supports, prints aggregates only. `test_score_external.py` ran it from a
code-only copy (no `data/`, no `rethink/data/`) on 60 cohort patients (159 scored scans, 157 fellow-only rows,
187 composites under neutral names): its AUCs equal the internal frozen path exactly (headline 0.858, (a) 0.871,
(b) 0.851, ref 0.917 — apparent, on training patients, inflated by design), and it correctly declines the headline
without `--tx` and (a) without composites. The frozen LGBM outputs are ranking scores (mean ≈ 0.42); recalibrate
(Platt) on the external set before reading them as risks.

## E23 — first-visit addendum (pre-declared in PROTOCOL.md 2026-11-10, run 2026-11-11)

Pool = the 21 E22 candidates refitted with first-visit scans weighted 2, plus `ftL` (encoder L fine-tuned nested, 125 splits, `ft_L.py`).
In-fold non-negative logistic stack on the inner 4-fold OOF predictions. One run with block H, one without. Nothing was tuned after the results were seen.

| run | AUC [95% CI] | rep mean ± SD | first-visit AUC |
|---|---|---|---|
| E23n_H | 0.795 [0.745–0.840] | 0.788 ± 0.012 | 0.793 |
| E23n_noH | 0.752 [0.701–0.800] | 0.746 ± 0.011 | 0.769 |

Paired cluster bootstrap, ΔAUC [95% CI]:
- E23n_H − E22_ref_single_num: all +0.145 [+0.085, +0.207]; first visit +0.110 [+0.064, +0.157].
- E23n_H − E22n_nested: all +0.014 [+0.004, +0.027]; first visit +0.009 [−0.003, +0.021].
- E23n_noH − E22_ref_single_num: all +0.102 [+0.045, +0.156]; first visit +0.085 [+0.038, +0.132].
- E23n_noH − E22n_noH: all +0.021 [+0.006, +0.038]; first visit +0.011 [−0.002, +0.025].

Mean stacker weights:
- With H, the top weights are bn_aug_lr 0.22, bi_L_lgbm 0.21, bnpm_lgbm 0.19 and bi_D_lr 0.17.
- Without H, the top weights are bn_aug_lr 0.18, bn_aug_lgbm 0.17, bi_D_lr 0.17 and bi_L_lgbm 0.15.
- The fine-tuned L gets almost no weight (mean 0.04 with H, 0.05 without).

Interpretation:
- Stacking and weighting give a small, real pooled gain over E22 (about +0.01 to +0.02).
- The first-visit gain is not significant.
- First-visit AUC stays just below 0.80 (0.793 with H, 0.769 without).
- Supervised fine-tuning of L adds nothing beyond the frozen embedding.
- 115 distinct configurations have been evaluated through E23.
