# Hurst-Aware Adaptive Partitioning of Persistent Time Series

[![DOI](https://zenodo.org/badge/DOI/PENDING.svg)](https://doi.org/PENDING)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Pre-registered](https://img.shields.io/badge/status-pre--registration-orange.svg)](./prereg/h2-prereg-v1.md)

**Pre-registration and reproducibility package** for a benchmark study of Hurst-aware adaptive chunk-boundary partitioning against fixed-interval (TimescaleDB-style) and variance-aware (CUSUM) baselines.

> **Status.** This release archives the pre-registered protocol and reference scaffolding only. **No empirical results on the registered datasets are included.** All analyses on D1 (S&P 500) and D2 (NAB) will be run end-to-end after the protocol is locked at v1.0 final.

**Author:** [Dineshkumar Malempati Hari](https://orcid.org/0009-0003-1036-9477), Independent Researcher.

**Companion paper:** [Static and Temporal Fractal Coupling Between Volatility and Trading Volume](https://doi.org/10.5281/zenodo.19611543) — the Hurst estimation infrastructure used here originates from that paper.

---

## What this repository is

A pre-registered, falsifiable empirical study of one specific engineering claim:

> On time series with long-range dependence (H > 0.6), drawing chunk boundaries at statistically detected regime breaks in the rolling Hurst estimate reduces bytes read per query relative to fixed-interval partitioning, by 20-40%, with a 95% CI strictly above zero.

The pre-registration in [`prereg/h2-prereg-v1.md`](./prereg/h2-prereg-v1.md) specifies the hypotheses, datasets, estimator battery, baselines, query workload, primary and secondary outcomes, statistical tests, **falsification criteria**, multiple-testing protocol, stopping rules, and deviations protocol. It is archived on Zenodo at the DOI in `CITATION.cff`, with a permanent Git tag at `v0.1.0-prereg`.

## What this repository is *not*

- Not a paper. The paper follows after empirical execution against the registered protocol.
- Not a system. The implementation is reference scaffolding; it is not a production-ready time-series database engine.
- Not yet a complete dataset acquisition. Data are acquired on first run from the pinned upstream sources; checksums are recorded after acquisition.

## Why pre-registration

Modern systems research has a credibility problem: results are often reported after analytical decisions are made, with no record of what was decided in advance. Pre-registration before analysis fixes the protocol publicly. If the registered protocol falsifies the hypothesis, the null result is publishable on its own merits.

The pre-registration here mirrors the [OSF pre-registration template](https://osf.io/zab38/) adapted for systems benchmarks. It is comparable in spirit to a Type I clinical trial pre-registration, narrower in scope and tailored to a single hypothesis family.

## Repository structure

```
hurst-aware-partitioning/
├── prereg/
│   └── h2-prereg-v1.md             # The pre-registration itself (load-bearing)
├── src/hurst_partitioning/
│   ├── estimators.py               # DFA, R/S, GHE, wavelet wrappers + block-bootstrap CIs
│   ├── partitioner.py              # Hurst-aware chunk-boundary policy
│   ├── baselines.py                # FIXED-DAILY, FIXED-MONTHLY, VARIANCE-CUSUM, EQUAL-ROWS, ORACLE
│   ├── benchmark.py                # Query workload harness, I/O accounting
│   └── io.py                       # D1 / D2 / D3 data loaders, fGn corpus generator
├── experiments/
│   ├── query-library-v1.json       # Generated on first run, hashed, archived
│   └── README.md
├── research/paper/
│   └── outline.md                  # Paper outline (post-empirical, not yet drafted)
├── tests/
│   └── test_smoke.py               # Quick checks that the scaffold loads and the estimators agree on white noise
├── data/                           # Gitignored; populated on first run
├── replicate.py                    # Master script — runs the registered protocol end-to-end
├── pyproject.toml
├── CITATION.cff
├── .zenodo.json
└── LICENSE                         # MIT
```

## How to cite

Until the paper exists, cite the pre-registration via the Zenodo concept DOI in [`CITATION.cff`](./CITATION.cff). The concept DOI always resolves to the latest version; the `v0.1.0-prereg` Git tag points to the protocol exactly as registered.

## Reproducibility quick-start

```bash
git clone https://github.com/mhdk1602/hurst-aware-partitioning.git
cd hurst-aware-partitioning
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest tests/  # smoke test, ~10 seconds
```

Once data are acquired:

```bash
python replicate.py
```

`replicate.py` executes the registered protocol end-to-end: data acquisition (~5 min on first run for D1), estimator battery, partitioner construction for each policy, query workload, paired Wilcoxon with Holm-Bonferroni correction, summary tables, and the §13 falsification check.

## Connection to the wider research program

| Node | Status | DOI |
|---|---|---|
| Fractal-PV coupling paper (Hurst infrastructure originates here) | Preprint, May 2026 | [10.5281/zenodo.19611543](https://doi.org/10.5281/zenodo.19611543) |
| H2 pre-registration (this) | Pre-registration v1.0 | See `CITATION.cff` |
| H1 (learned-fractal hybrid OLAP) | Plan-only | — |
| H3 pivot (multifractal HNSW or fractal ANN diagnostics) | Plan-only | — |
| Visualization spike | Plan-only | — |

The research plan that situates this paper within the broader program is private (in the author's `non-git-files/`). The plan structure is summarized in §2 of the pre-registration.

## License

[MIT](./LICENSE). Use freely with attribution.

## Acknowledgments

Hurst estimation primitives and the block-bootstrap pattern are adapted from [fractal-pv-coupling](https://github.com/mhdk1602/fractal-pv-coupling). Numenta Anomaly Benchmark is courtesy of Numenta. The Apache Iceberg, Delta Lake, and TimescaleDB teams shipped the production systems whose default behavior this study evaluates against.
