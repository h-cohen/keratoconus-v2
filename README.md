# Keratoconus: predicting referral for corneal cross-linking from a single Pentacam visit

Final model (2026-11-13): **E28 blend**. It is an equal-weight rank average of XGBoost, a 5-seed LightGBM bag and an
elastic-net logistic regression. The input is one visit: both eyes' Pentacam numerics (block N), 96 measurements per eye
from the decoded thickness, curvature and elevation maps (block R; own, fellow and own − fellow), and age.

| arm | input | AUC, all 944 scans | AUC, first visit (523 eyes) |
|---|---|---|---|
| **noH (primary)** | single visit | 0.768 [0.714–0.817] | **0.790** [0.744–0.835] |
| H (auxiliary) | + fellow-eye treatment before the visit | 0.813 [0.767–0.857] | **0.813** [0.770–0.854] |
| reference | age only | 0.661 | 0.680 |

Evaluation: 5-fold patient-grouped CV repeated 5 times (seeds 2026–2030), all tuning in-fold, and a patient-cluster
bootstrap for the 95% CI.

**Full method and results:** `publication_v2/report.html`. Rebuild it with `python -I publication_v2/build_report.py`.

## Layout

```
rethink/                 final pipeline (data prep, map decoding, features, model, freeze)
  exp28_tricks.py        the model and its CV run:  python -I rethink/exp28_tricks.py {noH|H} blend
  freeze_final.py        refit on all scans -> rethink/frozen/final.pkl (not shared)
  radiomics.py           block R (map measurements)
  harness.py             evaluation (folds, AUC, CIs, logging to results/experiments.csv)
  RESULTS.md, PROTOCOL.md  experiment log and pre-declarations
  archive/               superseded experiments (their runs remain in experiments.csv)
publication_v2/          methods and results report, figures, tables
preprocessing/           cropping of the Pentacam composites
```

Data (`data/`, `rethink/data/`) and fitted models are not in the repository.
