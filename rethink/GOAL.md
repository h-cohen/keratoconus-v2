# Goal (for /goal)

Get a legitimate predictor of CXL referral with **dev repeated-CV AUC ≥ 0.70**, and show it beats
age-only LR with a paired cluster bootstrap whose ΔAUC CI excludes 0. Then score at most 3
pre-declared candidates on the lockbox once, and report that AUC as the headline.

Work only under `rethink/PROTOCOL.md` and `CLAUDE.md`:
- Lockbox: never touch it during development. Score it once, at the end.
- Use `harness.evaluate()`, with all preprocessing and tuning nested inside each training fold.
- Log every run to `rethink/results/experiments.csv`. Never delete rows.
- After each experiment, add a result entry to `rethink/RESULTS.md`, run `compare.py` against
  `E00_age_lr`, commit on `clean-rethink`, and report the result before starting the next one.
- Route order: E01 tabular (done, negative) → E02 fellow eye (borderline) → E03 history (done,
  negative) → E04 image representation → E05 raw mesh → E06 stacking → final lockbox.

Stop when one candidate passes all three success criteria and its lockbox AUC is reported, or
when the remaining routes are exhausted and the negative result is written up honestly.
