# Amendment 1 to the H2 Pre-Registration

**Parent document:** [`h2-prereg-v1.md`](./h2-prereg-v1.md) — archived on Zenodo at [10.5281/zenodo.20188014](https://doi.org/10.5281/zenodo.20188014).
**Amendment posted:** 2026-05-14
**Type:** Additive (new secondary outcome). The primary outcome and falsification rule of v1 are **unchanged**.
**Status:** Pre-registered. No D1 or D2 analyses have been performed prior to this amendment being archived.

---

## Why this amendment exists

The v0.2.0 D3 pilot (synthetic fGn corpus, 16 persistent series, length 4096; results at [`experiments/d3-pilot-results.json`](../experiments/d3-pilot-results.json) on Git tag `v0.2.0`) surfaced a methodological problem with the v1 primary outcome metric **M-PRIMARY** (total bytes read per query, normalized to FIXED-DAILY).

On constant-H synthetic fGn — where there are no within-series regime breaks by construction — the candidate **HURST-CI** policy correctly produces a single chunk per series (no false fires). The **VARIANCE-CUSUM** baseline, in contrast, fires liberally on Gaussian noise, producing many small chunks. M-PRIMARY rewards smaller chunks regardless of whether the boundaries are statistically meaningful, so VARIANCE-CUSUM "beats" HURST-CI by Hodges-Lehmann shift +26,184 bytes/query (`p_holm ≈ 5e-235`). This is *expected and correct* on D3 (no real regime breaks to recover), but it identifies a vulnerability that will follow the protocol to D1 and D2 unless addressed: **any partitioner that subdivides liberally on noise can win on bytes-read against a conservative partitioner whose boundaries are real**.

The amendment adds one secondary outcome that **operationalizes "boundary quality"**, plus one secondary hypothesis tied to it. M-PRIMARY remains the primary outcome and the falsification rule §12 is not changed.

## What this amendment adds

### New secondary outcome metric: M-S6, Boundary Quality Score

**M-S6** is the F1 score of partitioner boundaries against ground-truth regime breaks on a new D4 synthetic corpus (defined below). It is a **per-series** scalar in `[0, 1]`, computed against:

- Ground-truth boundaries on D4: positions where the generating H switches.
- Predicted boundaries: positions returned by the partitioner's `fit()`.
- Tolerance: a predicted boundary counts as true-positive iff it lies within `tol = max(window // 4, 50)` rows of a ground-truth boundary. The tolerance is **fixed** in this amendment.

M-S6 is reported per dataset × partitioner, averaged across series, with 95% block-bootstrap CIs.

### New dataset D4 (synthetic regime-switching fGn)

D4 is generated alongside D3 and serves the same code-correctness function for boundary quality that D3 serves for I/O accounting:

- 256 series, each composed of 4 segments of length 4096 concatenated.
- Each segment has its own H drawn deterministically from {0.30, 0.55, 0.70, 0.85}.
- Adjacent segments are forced to differ in H by at least 0.15. The 3 boundaries per series (at positions 4096, 8192, 12288) constitute the ground truth.
- Generator seed: `20260514` (matches D3). Per-series generation deterministic from the global seed via `SeedSequence.spawn`.
- Implementation: extends `src/hurst_partitioning/io.py::make_fgn_corpus` to a new function `make_regime_switching_corpus`.

### New secondary hypothesis: H2-S4

**H2-S4 (boundary quality on D4):** On D4 the candidate **HURST-CI** policy has a strictly higher mean **M-S6** than each of the FIXED-DAILY, FIXED-MONTHLY, VARIANCE-CUSUM, and EQUAL-ROWS baselines. Predicted effect: HURST-CI F1 ≥ 0.5; each baseline's F1 ≤ 0.3.

H2-S4 is **disconfirmed** if HURST-CI's F1 falls below 0.5 on D4 OR any baseline's F1 exceeds 0.4. Either disconfirmation is exploratory — it does not falsify H2-P from v1, but it does make the case for H2 weaker because the primary win cannot be attributed to boundary quality.

### Relationship to M-PRIMARY

M-S6 is **not** combined with M-PRIMARY into a composite score. M-PRIMARY stays primary; M-S6 is reported alongside. Reviewers can then judge whether a primary win on bytes-read is supported by a secondary win on boundary quality (the credible case) or undermined by a secondary loss (the "spurious chunks beat real boundaries" case).

### Update to §11 family

The pre-registered ranking of comparisons (§11 of v1) is extended by one block at the bottom, ranked lower than every v1 comparison:

49. D4 × FIXED-DAILY × DFA (boundary quality)
50. D4 × FIXED-MONTHLY × DFA
51. D4 × VARIANCE-CUSUM × DFA
52. D4 × EQUAL-ROWS × DFA

Holm-Bonferroni is re-applied at the family level. Family size grows from 48 to 52; α stays at 0.05.

## What this amendment does NOT change

- **Primary outcome (§8 v1):** M-PRIMARY remains primary. Bytes-read accounting is unchanged.
- **Falsification rule (§12 v1):** unchanged. M-S6 wins or losses are exploratory with respect to the H2-P falsification check.
- **Datasets D1, D2, D3 (§4 v1):** unchanged.
- **Estimator battery (§5 v1):** unchanged.
- **Baselines (§6 v1):** unchanged.
- **Workload (§7 v1):** unchanged.
- **Stopping rules (§13 v1):** unchanged.
- **Software environment (§17 v1):** unchanged.

## Authorship and conflicts

Sole author. No conflicts.

## Code & data

- D4 generator: `src/hurst_partitioning/io.py::make_regime_switching_corpus` (lands at v0.2.1; not in the v0.2.0 tarball that v1 was archived alongside).
- M-S6 implementation: `src/hurst_partitioning/benchmark.py::boundary_f1` (v0.2.1+).
- This amendment will be re-archived on Zenodo as a new version on the existing concept DOI (`10.5281/zenodo.20188013`) once the v0.2.1 implementation lands.

## What "pre-registered" means here

This amendment is being archived *before* M-S6 or D4 have been evaluated on any series. The first M-S6 computation will be on D4 only; D1 and D2 boundary-quality evaluations require a separate amendment (D1/D2 have no ground-truth regime breaks, so M-S6 is not directly applicable to them without proxy regime detection, which is itself a research question).

---

*This file does not replace `h2-prereg-v1.md`. The v1 document remains authoritative for everything it covers; this amendment is additive only.*
