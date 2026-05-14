# Pre-Registration: Hurst-Aware Adaptive Partitioning of Persistent Time Series

**Version:** 1.0
**Status:** Pre-registered. No analyses on the registered datasets have been performed prior to the public archival of this document.
**Author:** Dineshkumar Malempati Hari ([ORCID 0009-0003-1036-9477](https://orcid.org/0009-0003-1036-9477)), Independent Researcher
**Date posted:** 2026-05-14
**Archived at:** Zenodo (DOI minted on first release; see CITATION.cff)
**Connects to:** [fractal-pv-coupling](https://github.com/mhdk1602/fractal-pv-coupling) and the companion [research plan](https://github.com/mhdk1602/fractal-indexing) (H2)

---

## 1. Title

Hurst-Aware Adaptive Partitioning of Persistent Time Series: A Pre-Registered Benchmark Against Fixed-Interval and Variance-Aware Baselines.

## 2. Background

Production time-series databases (TimescaleDB, QuestDB, InfluxDB) draw chunk boundaries on fixed time or row intervals, indifferent to the statistical structure of the series. On series with long-range dependence (Hurst exponent H above 0.5), persistent regimes mean that adjacent time intervals carry correlated information; a fixed-interval chunk boundary may fall in the middle of a regime, fragmenting a coherent block of correlated data across two storage chunks. The hypothesis is that drawing chunk boundaries at statistically detected regime breaks in the rolling Hurst estimate reduces the total bytes read per query relative to fixed-interval partitioning.

This question has, to the best of the author's literature search (May 2026), no published empirical answer at the storage layer. Adjacent algorithmic work exists ([Zhang et al. 2023, arXiv 2310.19051](https://arxiv.org/abs/2310.19051) on optimal sequence partitioning for Hurst estimation; [Cerqueti et al. 2022](https://www.sciencedirect.com/science/article/abs/pii/S0888613X22002067) on fuzzy clustering with time-varying memory), but none address storage-layer chunking decisions, and none compare against the deployed time-series database defaults.

## 3. Pre-Registered Hypotheses

### Primary hypothesis (H2-P)

On streaming univariate time series whose effective long-range dependence statistic (rolling Hurst exponent, DFA(2), window W=500, step 20) exceeds 0.6 over at least 30% of the series length, a partitioning policy that draws chunk boundaries at statistically detected regime breaks in the rolling Hurst estimate produces fewer total bytes read for the registered query workload than a fixed-daily-interval partitioning policy. The expected effect size is a mean reduction of 20-40% in bytes read, with the 95% confidence interval on the Hodges-Lehmann shift estimator strictly above zero.

### Secondary hypotheses

- **H2-S1 (adaptive competitor):** The primary win replicates against a variance-aware baseline that draws chunk boundaries via CUSUM on |returns|. Predicted reduction: 10-25%.
- **H2-S2 (anti-persistent control):** On series with rolling Hurst below 0.5, the Hurst-aware policy does *not* outperform fixed-daily. A win here would be artifactual and would falsify the primary hypothesis.
- **H2-S3 (estimator robustness):** Conclusions do not depend on a single Hurst estimator. The primary result must replicate across DFA(2), R/S, and generalized Hurst exponent (GHE) at q=2; the wavelet estimator is reported but treated as informational.

## 4. Datasets

| ID | Source | Series | Period | Frequency | Acquisition | Provenance |
|---|---|---|---|---|---|---|
| **D1** | Yahoo Finance via `yfinance ≥ 0.2.37` | 50 S&P 500 tickers, all 11 GICS sectors, continuous listing | 2015-01-02 to 2026-04-30 | Daily OHLCV | Cached on first run to `data/raw/d1/*.parquet` | SHA-256 manifest in `data/d1.sha256` |
| **D2** | Numenta Anomaly Benchmark (NAB) | 10 NAB series labeled `realKnownCause/*` and `realTraffic/*` | Per-series | Per-series | Submodule pinned at NAB commit `<pinned at acquisition>` | NAB upstream license respected |
| **D3** | Synthetic fGn corpus | 1024 series, lengths 4096 / 16384 / 65536, H ∈ {0.3, 0.5, 0.7, 0.85} uniformly distributed across the corpus | n/a | n/a | Generated via `fbm ≥ 0.3`, seed `20260514`, regenerated deterministically | Code in `src/hurst_partitioning/io.py::make_fgn_corpus` |

D3 is used for code validation and for the anti-persistent control (H2-S2) only. The variance-aware baseline's CUSUM sensitivity is tuned on D3-train (half of D3, by series id parity) only; D3-test is held out.

## 5. Estimators (Registered)

The Hurst estimator battery, in order of primacy:

1. **DFA(2)** on cumulative log returns. Window W ∈ {250, 500, 1000}, step 20. Primary.
2. **R/S** on raw returns, same windowing. Robustness.
3. **GHE** at q=2, same windowing. Robustness.
4. **Wavelet** (Daubechies db4, level 6). Informational; not part of the formal robustness battery because of known finite-sample bias.

Every Hurst estimate is reported with a 95% block-bootstrap confidence interval ([Politis & Romano 1994](https://www.jstor.org/stable/2290993)) at 1000 bootstrap iterations. Block length is set to `floor(W ** (1/3))`. Partitioning decisions use the CI, not the point estimate (see §6).

## 6. Baselines (Registered)

Five baseline partitioning policies, each fully specified:

- **B1 FIXED-DAILY** — 1-day chunks (TimescaleDB default for daily-frequency data).
- **B2 FIXED-MONTHLY** — 30-day chunks (a common default for medium-frequency time series).
- **B3 VARIANCE-CUSUM** — chunk boundaries placed at CUSUM-detected change points on |returns|. CUSUM threshold tuned on D3-train.
- **B4 EQUAL-ROWS** — 10,000-row chunks regardless of time.
- **B5 ORACLE (synthetic only)** — on D3, where the ground-truth Hurst is known per series, partitions at the true regime change points. Upper bound, reported but not part of the primary comparison.

**CANDIDATE: HURST-CI** — chunk boundary fires when the rolling Hurst estimate's 95% CI on window `t` does not overlap with the CI on window `t-1`, AND the point estimates differ by at least 0.1. This is a conservative trigger; false-positive boundaries are penalized via the chunk-count secondary metric.

## 7. Query Workload (Registered)

A pre-specified library of **100 queries** per dataset:

- 40 time-range scans (start, end) where start and end are sampled from the date range, biased toward intervals of length 30, 90, or 365 days
- 30 aggregate-over-range queries (mean, std, p95, max) on a range
- 20 windowed-regression queries (slope of price on lagged price over a 60-day window starting at random dates)
- 10 anomaly-detection queries (range scan + threshold trigger)

The 100 queries are generated once on D3, archived to `experiments/query-library-v1.json`, hashed (`SHA-256`), and lifted unchanged to D1 and D2 by date-range alignment. The partitioner has no read access to the query library at partition time.

## 8. Primary Outcome Metric

**M-PRIMARY** — total bytes read from storage per query, averaged over the 100-query workload, normalized to FIXED-DAILY on the same dataset.

A "byte read" is counted at the chunk level: any chunk overlapping the query's time range counts its full byte size. This intentionally favors policies that align chunk boundaries with regime structure. Wall-clock latency is measured but is secondary.

## 9. Secondary Outcome Metrics

- **M-S1** Wall-clock query latency (mean over workload, with cold and warm cache reported separately)
- **M-S2** Chunk count (a lower chunk count with comparable I/O wins on metadata overhead)
- **M-S3** Build time (partitioner construction time as a function of series length)
- **M-S4** Memory footprint of the partitioner during construction
- **M-S5** Sensitivity to W (Hurst window length) — does the answer change qualitatively as W moves across {250, 500, 1000}?

## 10. Statistical Analysis (Registered)

For each dataset × baseline pair (D × B), the per-query bytes read is paired between HURST-CI and B. The test is a **paired Wilcoxon signed-rank** on the 100 paired observations. Effect size is reported as the **Hodges-Lehmann shift estimator** with 95% CI from block-bootstrap (1000 iterations).

Hypotheses are corrected with **Holm-Bonferroni** across the registered family. The family is enumerated explicitly in §11.

## 11. Multiple-Testing Family (Pre-Registered Ranking)

The total family size is **48** comparisons: 3 datasets (D1, D2, D3-persistent-only) × 4 baselines (B1, B2, B3, B4) × 4 estimators (DFA, R/S, GHE, wavelet). The wavelet estimator is included but not load-bearing.

Pre-registered ranking from most to least important:

1. D1 × B1 × DFA (primary primary)
2. D1 × B3 × DFA
3. D1 × B1 × R/S
4. D1 × B1 × GHE
5. D2 × B1 × DFA
6. D2 × B3 × DFA
7. D3-persistent × B1 × DFA (sanity check on synthetic)
8. D1 × B2 × DFA
9. D1 × B4 × DFA
10. through 48: remaining comparisons in dataset-then-baseline-then-estimator order

Holm-Bonferroni is applied at family-wise α = 0.05 across all 48 hypotheses simultaneously.

## 12. Falsification Criteria

The **primary hypothesis (H2-P) is rejected** if all three of the following hold:

- On D1, the Hurst-aware policy shows < 10% mean I/O reduction versus FIXED-DAILY OR the 95% CI on the Hodges-Lehmann shift crosses zero. AND
- The same holds on D2. AND
- H2-S2 fires: on D3 anti-persistent series (H < 0.5), the Hurst-aware policy shows > 10% reduction versus FIXED-DAILY — that is, the win replicates where it should not.

Any single one of the three is a concern; all three jointly constitute rejection.

## 13. Stopping Rules

All 48 paired comparisons are computed in a single batch, in a single run of `replicate.py`, before any results are inspected. No iterative tuning of the HURST-CI threshold parameters (CI-overlap rule, point-estimate gap of 0.1) on the test data. The variance-aware baseline's CUSUM sensitivity is the only parameter tuned, and it is tuned on D3-train only, before D1 and D2 are touched.

## 14. Deviations Protocol

Any analytical decision not specified in this document is logged in `prereg/deviations-log.md`, dated, and labeled "exploratory." Exploratory results may be reported as descriptive findings but cannot be claimed as pre-registered.

If the protocol must be revised post-archival, the revision is published as `prereg/h2-prereg-v2.md` with explicit diff against v1 and a new Zenodo version. Replication of the v1 protocol remains possible at the v0.1.0-prereg tag forever.

## 15. Negative Controls

Pre-specified workloads on which the Hurst-aware policy should *not* win:

- D3 anti-persistent (H<0.5) series with the registered workload (this is the H2-S2 control)
- D3 white-noise series (H=0.5) — chunking choice should be irrelevant
- D1 with a single-asset workload (no cross-sectional structure) — partitioning is per-series, so this should match Hurst-aware = fixed-daily on a per-series basis

A win on any of these workloads requires explanation in the analysis report.

## 16. Code & Data Availability

- **Code:** [github.com/mhdk1602/hurst-aware-partitioning](https://github.com/mhdk1602/hurst-aware-partitioning), tagged `v0.1.0-prereg` and archived on Zenodo.
- **D1 data:** cached on first run from `yfinance`; SHA-256 manifest at `data/d1.sha256` after acquisition. Manifest is committed; data files are gitignored.
- **D2 data:** NAB submodule, commit hash pinned in `experiments/d2.commit`.
- **D3 data:** regenerated deterministically from `src/hurst_partitioning/io.py::make_fgn_corpus(seed=20260514)`.

## 17. Software Environment

- Python 3.10 or higher
- Pinned dependency versions in `pyproject.toml`
- `pip install -e .` from a clean virtualenv
- `python replicate.py` runs the full registered protocol once data are acquired

## 18. Conflicts of Interest

None to declare.

## 19. Authorship

Sole author: Dineshkumar Malempati Hari, [ORCID 0009-0003-1036-9477](https://orcid.org/0009-0003-1036-9477).

## 20. Timeline (Informational, Not Part of the Pre-Registration)

| Phase | Window | Action |
|---|---|---|
| Implementation | 2026-05 to 2026-07 | Code on D3 (synthetic) only; baselines validated against known properties. |
| Pre-registration lock | 2026-08 | This document declared final v1.0; any later change becomes v2 with explicit diff. |
| Execution | 2026-08 to 2026-10 | D1 and D2 analyses run end-to-end. |
| Submission | 2026-11 | VLDB Industrial 2027 (deadline typically March, this is a buffer cycle). |

---

*This pre-registration is archived on Zenodo. The DOI in `CITATION.cff` always resolves to the latest version; the `v0.1.0-prereg` Git tag points to this exact text.*
