# Experiments

This directory holds the registered query library, dataset commit pins, and (post-empirical) the per-run logs.

## Files generated on first run

- `query-library-v1.json` — the 100-query workload (40 range scans, 30 aggregates, 20 windowed regressions, 10 anomaly scans). Generated deterministically from the registered seed `20260514`. SHA-256 archived to `query-library-v1.sha256` for the deviations log.
- `d2.commit` — the NAB upstream commit hash pinned at acquisition. Once written, this file must not be advanced between pre-registration and the empirical run.
- `run-YYYY-MM-DDThhmmssZ.log` — per-run audit log. One per execution of `replicate.py`.

## What never lives in this directory

- Raw data files (those live under `data/` and are gitignored).
- Estimator hyperparameters outside the registered grid (those would be a §14 deviation and would live in `prereg/deviations-log.md`).
- Cherry-picked subsets of results.

## Deviations log

If anything in the registered protocol is changed between v0.1.0-prereg and the empirical execution, the deviation is logged at `../prereg/deviations-log.md` with a date, a rationale, and the exploratory-vs-confirmatory label.
