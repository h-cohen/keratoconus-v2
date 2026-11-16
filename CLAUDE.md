# CLAUDE.md — keratoconus referral model (branch `main`)

## Project

Predict physician referral for corneal cross-linking (CXL) in keratoconus from a single Pentacam visit.
The paper goes to a new journal; the earlier TVST submission is superseded. The final model is fixed (2026-11-13;
feature set updated by E31 on 2026-11-16): the rank blend of XGBoost, a 5-seed LightGBM bag and an elastic-net LR on
blocks N + R (+ H) without the 11 derived indices (`FINAL_DROP = 'noeng'`), in `kcref/model.py`. Its CV run is
`E31_noeng_{arm}`; `E28_{arm}_blend` (all columns) is the comparison.
The methods/results note `publication_v2/report.html` is the source for the algorithmic part of the paper.
Rebuild it with `python -I publication_v2/build_report.py` after every new result.

## Rules

- **Two arms (user decision 2026-11-12).** noH (single visit) is the primary result. H (adds fellow-eye treatment
  strictly before the visit) is auxiliary, labelled "visit + fellow-eye treatment history". Every analysis is run and
  reported for both arms, side by side. H never replaces noH as the headline.
- **Evaluation.** Use `kcref/evaluate.py`: 5-fold StratifiedGroupKFold by patient × 5 repeats (seeds 2026–2030),
  on all 944 scans. Report pooled AUC and first-visit AUC, each with a patient-cluster bootstrap CI. Fit all preprocessing
  and tuning inside the training fold (nested inner CV). The lockbox was dropped by user decision on 2026-11-12;
  do not score `archive/lockbox_pids.csv`.
- **Logging.** Every run appends to `kcref/results/experiments.csv`. Never delete or edit rows; the number of
  configurations tried is reported in the paper. Pre-declare new experiments in `kcref/PROTOCOL.md`; log results in `kcref/RESULTS.md`.
- **Inputs.** Allowed: anything known at the visit (both eyes' numerics and maps from that day, age). Forbidden:
  scan date or year, visit counts, follow-up length, `months_to_cxl`, `visit_idx`, `legacy_row`, and anything dated
  after the visit.
- **Staging (user decision 2026-11-12).** Use `kcref/staging.py` and report both approaches side by side:
  - A: `Km F (D):` stages < 48, 48–53, 53–55, ≥ 55. Stage 4 holds 5 scans, because the inclusion filter is Km F ≤ 55.
  - B: `Rm B (mm):` stages ≥ 5.70, 5.15–5.70, 4.95–5.15, < 4.95.
  - Do not substitute KMax or Amsler–Krumeich.
- **Privacy.** Identifiable files are read on this machine only: raw exports, image filenames, `kcref/data/private_keys.csv`.
  No output, log, commit or report may contain names, ids, dates of birth or original filenames. Key everything by the
  pseudonymous `pid`. Never send data to an external service.
- **Git.** Commits are authored as `Hadar Cohen <hal.nls@gmail.com>`, with no co-author or session trailers.
  Do not edit `paper/` (gitignored).

## Layout and data

- Code: `kcref/`. See README.md for the data-flow diagram and the run order.
- Built data: `kcref/data/`, gitignored. Fitted models: `kcref/frozen/`, gitignored. SHAP files (`kcref/results/shap/`)
  hold raw clinical values and are gitignored. Everything else a result needs (experiments.csv, OOF scores in
  `kcref/results/oof/`, run logs, tables, figures, report) is committed and pushed after every run: the working copy
  was once wiped by an outside script (2026-11-16) and only what was on GitHub survived.
- Sources: `data/` (or `KC_SRC`). These are `multimodal_dataset.csv`, `221204_v2/Raw_Combined_IdsMesh.xlsx` (raw export,
  11,760 exams), `221204_v2/Combined_excel_KC.xlsx` (CXL / keratoplasty dates, which define the label) and
  `image_scans/` (four-map composites).
- Cohort: 944 scans, 523 eyes, 385 patients, 167 positive scans. First visit: 523 eyes, 125 positive.
- Label: the eye was ever cross-linked. Scans after CXL are excluded.
- `Exam Date:` in the exports has day/month swaps. Use `scan_date` (from the filename) and `prepare_data.parse_date`.
- Superseded experiments live in `kcref/archive/`. Their scripts import modules that no longer exist, so they are a
  record, not runnable code.

## Sub-agents

All sub-agents (Agent tool, workflows) run on Sonnet 5.5: pass `model: "sonnet"` on every spawn.
