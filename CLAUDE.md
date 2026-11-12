# CLAUDE.md — keratoconus-v2, branch `clean-rethink`

## Mission

Predict physician referral for corneal cross-linking (CXL) in keratoconus from Pentacam data.
Target: **ROC-AUC ≥ 0.70 under clean evaluation.** Any model, feature or pipeline is fair game,
provided the protocol below is followed. Work in small steps. Log every run. After each
experiment, report the result to the user before starting the next one.

Background: the paper was rejected by TVST. Reviewers said the model learns that young patients
get referred, not anything about disease, and that the AUC is moderate. The submitted
AUC of 0.708 was inflated:

- XGBoost early stopping used the test fold (`algo-pipeline/src/engine.py` and `src/evaluation.py`
  pass `eval_set=[(X_test, y_test)]`);
- CV was split by eye, not by patient;
- 53 of the 220 positive scans were taken after the eye had already been cross-linked.

Once these are fixed, **age alone is the model to beat**, and the old pipeline does worse than it.
The full write-up is in `publication/age_subanalysis_draft.md` in the main checkout.

The rules are in `rethink/PROTOCOL.md`; read it before any experiment. The short version:

- **Lockbox**: `rethink/lockbox_pids.csv` holds 25% of patients
  (sha256 `f875b78075576864c9b8acf25cbdd87d2700647dbfbd93105af37043dca04956`).
  - **Never** evaluate, tune or look at lockbox performance during development.
  - It is scored **once**, at the end, for ≤ 3 candidates declared in advance.
  - Do not re-draw it.
- **Development**: `harness.evaluate(name, fit_predict, desc, params)` runs a 5-fold
  StratifiedGroupKFold by patient, repeated 5 times, on the dev set (713 scans, 289 patients).
  - All preprocessing goes inside `fit_predict`: scaling, selection, imputation, early stopping
    and tuning. Tuning must be nested, i.e. inner CV inside the training fold.
- **Logging**: every run is appended to `rethink/results/experiments.csv`, which is git-tracked.
  Never delete rows. The number of configurations tried will be reported in the paper.
- **Inputs**:
  - Allowed: anything known at the index visit — the scan's numerics and maps, age, and the
    fellow eye's scan from the same visit. Prior scans are also allowed, but only in a model
    reported separately as **"visit + history"**.
  - Forbidden as features: scan date or year, number of scans, follow-up length, `months_to_cxl`,
    `visit_idx`, `legacy_row`, and any information dated after the index scan (including a future
    CXL in the fellow eye).
- **Success** requires all three:
  1. dev AUC ≥ 0.70;
  2. a paired cluster bootstrap shows the model beats `E00_age_lr` (ΔAUC CI excludes 0);
  3. the lockbox AUC is reported as the headline, whatever it is.
- If a result looks too good, hunt for leakage before celebrating.

## Setup (single folder)

Everything lives in one folder, `keratoconus-v2/`, checked out on branch `clean-rethink` (2026-11-10;
the old `keratoconus-v2-clean` worktree is gone). `data/` and `results/` (sources) and `rethink/data/`,
`rethink/results/` and `rethink/frozen/` (built outputs) are gitignored and sit in the same folder.
All scripts resolve paths from the repo root; set `KC_SRC` only if the data are elsewhere.

```bash
cd keratoconus-v2
pip install xgboost scikit-learn pandas pyarrow openpyxl lightgbm matplotlib torch transformers
python -I rethink/exp00_baselines.py   # sanity check: must reproduce the E00 table below
                                         # (appends 7 rows to experiments.csv; note them as a reproduction)
```

`rethink/data/` is already built. To rebuild it from scratch: `prepare_data.py`, `prepare_raw.py`,
`prepare_history.py`, `prepare_extra.py`, `prepare_raw_all.py`, `decode_maps.py`, `decode_all.py`,
`ocr_panel.py`, `extract_dino_grid.py`, `extract_colorgrid.py`. The current programme is in
`gpu_run/DESIGN.md` (untracked); the agent brief is `gpu_run/AGENT_PROMPT.md`.

## Code map (`rethink/`)

| file | role |
|---|---|
| `prepare_data.py` | Builds the frozen cohort. Inputs: `data/multimodal_dataset.csv`, `data/multimodal_dataset_features.txt`, `data/221204_v2/Combined_excel_KC.xlsx` (CXL / keratoplasty dates per eye), `results/age_subanalysis/cnn_features.npy` (frozen ResNet-18 features, 997×2048, row-aligned with the legacy cohort). Writes `rethink/data/cohort.parquet` (pseudonymous `pid`, `eid` = pid-eye, `scan_date`, `visit_idx`, `first_visit`, `y`, `months_to_cxl`, `legacy_row`, 55 numerics) and `cnn_frozen.npy` (aligned). `private_keys.csv` maps pid to the original id and image filename (needed to join images). |
| `harness.py` | `load(split)`, `numeric_features()`, `evaluate()`, `cluster_boot_auc()`. Reports pooled OOF AUC with a patient-cluster bootstrap CI, the per-repeat mean ± SD, and the **first-visit AUC** (one scan per eye; this is the paper's "single visit" claim). |
| `models.py` | Fold-safe builders: `logreg(cols, C)`, `xgb_tab(cols, **xgb)`, `legacy_multimodal(cols, k)`. `_xgb_fit` does early stopping on an inner patient-grouped split. |
| `exp00_baselines.py` | E00 reference runs. |
| `make_lockbox.py` | Already run; it refuses to run again. |
| `PROTOCOL.md` | Frozen rules plus a change log. Changing the rules needs a dated entry and a reason. |

Name new experiments `rethink/expNN_<topic>.py`, with run names `ENN_<variant>`.

## Data facts

- Cohort: 944 scans, 523 eyes, 385 patients, 167 positive scans (17.7%), 125/523 positive eyes.
  Scans per eye: 1 for 333 eyes, 2 for 120, 3 or more for the rest.
  Scan dates run from 2017-09 to 2022-02.
- The label `y` is per eye: 1 if the eye was ever cross-linked (date in the clinical sheet).
  The positive scans fall 0–6 months before CXL (91), 6–12 (41), 12–24 (24) and > 24 months (11).
- Legacy inclusion filter: `Km F (D):` ≤ 55. This is mean front K, not KMax — a known quirk,
  kept for comparability. Do not change the cohort to chase AUC.
- `Exam Date:` has day/month swaps in some rows. Use `scan_date` (parsed from the filename).
- The 55 numerics are listed in `data/multimodal_dataset_features.txt` (the paper wrongly says 59).
  `age` is the only demographic.
- Within the 12 months before CXL, Kmax (AUC 0.49) and thinnest pachymetry (0.50) do not separate
  the classes. At a single visit, a soon-to-be-treated eye looks like any other KC eye.
- Images: `data/cropped_output_v2/<filename stem>_<type>.jpg`, where type is
  corneal_thickness, curvature_front, elevation_front or elevation_back. These are colour-coded
  Pentacam maps on absolute scales. Join via `rethink/data/private_keys.csv` (`filename`).
- Other sources not yet used:
  - `data/221204_v2/KC_filtered_by_date_labeled.csv` — longitudinal raw exams; there were about
    11,760 Pentacam exams before filtering;
  - `data/221204_v2/Raw_Combined_IdsMesh.xlsx` — raw mesh / elevation data;
  - `data/image_scan_metadata_labeled.csv`;
  - `data/Yamit New/`, `data/raw/`.
  Inspect them before using them.

## E00 reference results (dev set, 2026-11-06)

| run | AUC [95% CI] | first-visit AUC |
|---|---|---|
| E00_age_lr (age only) | **0.656** [0.584–0.724] | **0.680** |
| E00_clin_lr (55 numerics, L2 C=0.1) | 0.624 | 0.666 |
| E00_clin_xgb | 0.609 | 0.618 |
| E00_clin_noage_xgb | 0.490 | 0.503 |
| E00_mm_legacy (v15 clean) | 0.575 | 0.592 |
| E00_mm_noage | 0.448 | 0.484 |
| E00_img_only (frozen R18, k=30) | 0.468 | 0.497 |

Takeaway: the 54 non-age numerics and the frozen ImageNet features carry roughly no signal on top
of age. A different classifier will not fix that; gains must come from **new information**.

## Roadmap (in priority order; one experiment at a time; report after each)

1. **E01 — tabular done right** (cheap, sets the ceiling for single-visit numerics):
   - model age non-linearly (splines);
   - try age × severity interactions, monotone constraints, LightGBM/CatBoost with nested tuning;
   - engineer known KC indices: KMax − Km, ART-like pachymetric progression ratios from the
     D*mm Prog / Pachy columns, front/back elevation ratios, a BAD-D-like composite, distance
     from apex to the thinnest point.
   - Expected gain: small. Run it to rule this route out.
2. **E02 — fellow eye at the same visit**:
   - join the other eye's scan from the same patient and date;
   - features: inter-eye asymmetry (ΔKMax, ΔPachy Min, ΔKI/ISV/IHD) and fellow-eye severity;
   - check first how many scans have a same-day fellow scan;
   - this still counts as single-visit and is clinically well motivated.
3. **E03 — visit + history** (separate model, never mixed into the single-visit claim):
   - features from prior scans of the same eye: ΔKMax per year, ΔPachy min per year,
     time since previous scan, previous values;
   - the previous paper reported 0.75 AUC on serial exams;
   - check whether `KC_filtered_by_date_labeled.csv` adds prior visits that lack images
     (prior numerics alone are enough here).
4. **E04 — better image representation**. Frozen ImageNet R18 is at chance. Options:
   - (a) invert the Pentacam colour scale to physical values (µm or D) on a polar grid, then
     ring/sector statistics or PCA;
   - (b) a stronger frozen encoder (DINOv2 / CLIP ViT) plus a linear probe with nested
     regularisation;
   - (c) light fine-tuning only with heavy regularisation and nested early stopping; earlier
     fine-tuning runs overfit.
   - Feature selection must stay inside the fold.
5. **E05 — raw mesh data** (`Raw_Combined_IdsMesh.xlsx`), if it maps onto the cohort scans.
6. **E06 — ensembling/stacking** of the best E01–E05 models, using nested OOF stacking only.
7. **Final**:
   - declare ≤ 3 candidates in `PROTOCOL.md`, then score them on the lockbox once;
   - also report sensitivity, specificity, PPV and NPV at a threshold chosen within each training
     fold, calibration, and DCA;
   - report AUC within age strata (< 25, 25–29, ≥ 30) and by Belin stage with n per stage.
     Reviewers asked for these.

Secondary endpoint worth adding later: objective progression (e.g. a KMax increase ≥ 1 D within
12 months) for eyes with follow-up. Reviewers want a non-referral endpoint. Report it as secondary;
it does not replace the primary label.

## Staging for stage analyses (final, user decision 2026-11-12)

Use `rethink/staging.py` for every stage-stratified result and report both approaches side by side:
- **A:** `Km F (D):` (front mean K, not KMax): 1 < 48, 2 48–53, 3 53–55, 4 ≥ 55 D. Stage 4 holds only 5 scans, because the inclusion filter is Km F ≤ 55. Report that; don't work around it.
- **B:** `Rm B (mm):`: 1 ≥ 5.70, 2 5.15–5.70, 3 4.95–5.15, 4 < 4.95 mm.

Do not substitute KMax sagittal or Amsler–Krumeich (`ak_stage.py`); those were earlier variants. The cohort filter is unchanged.

## Working conventions

- Commit after each experiment: script, the `experiments.csv` rows, and a short entry in
  `rethink/RESULTS.md` (create it) with the result and its interpretation, negative results included.
- Default compute: CPU is fine for tabular work. The image work (E04) benefits from a GPU.
- Do not edit `paper/` (manuscript drafts). It is gitignored.
- Use `python -I` or run from the repo root. Do not write outputs into `data/`.

## Sub-agents

All sub-agents (Agent tool, workflows) run on Sonnet 5.5: pass `model: "sonnet"` (`claude-sonnet-5-5`) on every
spawn. Do not use other models for sub-agents.
