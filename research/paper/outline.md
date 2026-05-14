# Paper Outline (Post-Empirical)

**Status:** Outline only. The paper is drafted after the registered protocol is executed end-to-end.

**Working title:** Hurst-Aware Adaptive Partitioning of Persistent Time Series: A Pre-Registered Storage-Layer Benchmark.

**Target venue:** VLDB Industrial 2027 (deadline typically March).

## Section plan

1. **Introduction.** The fixed-interval default in production time-series databases ignores the statistical structure of the series. We test a single, falsifiable claim about whether long-range dependence in the series is exploitable at the storage layer.
2. **Background.** Hurst exponent estimation (DFA, R/S, GHE, wavelet) and storage-layer chunking in TimescaleDB / QuestDB / InfluxDB. Brief literature on adjacent algorithmic work (Zhang 2023, Cerqueti 2022).
3. **Pre-registered protocol.** Reproduces the public pre-registration at `prereg/h2-prereg-v1.md`. Hypotheses, datasets, estimator battery, baselines, query workload, statistical analysis, falsification criteria.
4. **Implementation.** The reference partitioner, the I/O-accounting harness, the registered query workload generator, the block-bootstrap pipeline.
5. **Results.** D1, D2, D3 tables of paired Wilcoxon outcomes with Holm-Bonferroni-adjusted p-values and Hodges-Lehmann shift estimators with 95% CI. The §13 falsification check is reported as a binary outcome.
6. **Discussion.** Threats to validity, sensitivity to W, behavior under non-stationarity, what generalizes to other long-memory series (network traffic, IoT sensor streams).
7. **Limitations.** Single-table, single-key partitioning only. Production database integration is future work.
8. **Reproducibility statement.** Pre-registration on Zenodo. Code on GitHub. Data acquired via pinned upstream sources with SHA-256 manifests. Deviations log committed.

## Figures (planned)

- F1. Rolling Hurst on three exemplar D1 tickers, with candidate chunk boundaries from HURST-CI overlaid against FIXED-DAILY and VARIANCE-CUSUM. (Sourced from the visualization spike v2.)
- F2. Per-query bytes-read distribution under each policy on D1, violin plot, paired.
- F3. Hodges-Lehmann shift CIs on D1 across all four baselines × four estimators.
- F4. Sensitivity of the primary result to W ∈ {250, 500, 1000} (the M-S5 secondary metric).
- F5. The §13 falsification dashboard.

## What I will NOT write before the empirical run

- A predictive narrative about why H2 will succeed.
- A discussion section calibrated to expected results.
- Pre-rendered tables with placeholder values.

The §13 falsification rule decides the framing of every later section.
