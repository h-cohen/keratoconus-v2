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

## Harness / protocol notes
- `harness.summarise` now returns NaN for the first-visit AUC when a subset has no first-visit scans
  (needed for E03). Earlier numbers are unchanged.
- E00 was rerun during setup, which appended a second copy of each E00 row to `experiments.csv`.
  Count them once each.
