"""Smoke and unit tests for v0.2.0.

Covers:
    * Package version pin (still load-bearing).
    * Trivial partitioners (FIXED-DAILY, FIXED-MONTHLY, EQUAL-ROWS, ORACLE).
    * CUSUM baseline (implemented at v0.2.0).
    * Synthetic fGn corpus determinism (prereg §4).
    * DFA(2) recovers H within tolerance on a known H=0.7 sample.
    * block_bootstrap_ci shape contract.
    * paired Wilcoxon + Holm-Bonferroni table contract.
    * Workload generator counts and indices.
    * Chunk-level I/O accounting in ``measure``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hurst_partitioning import __version__
from hurst_partitioning.baselines import (
    EqualRowsPartitioner,
    FixedDailyPartitioner,
    FixedMonthlyPartitioner,
    OraclePartitioner,
    VarianceCusumPartitioner,
)
from hurst_partitioning.benchmark import (
    Query,
    boundary_f1,
    generate_workload,
    measure,
    paired_wilcoxon_with_holm,
)
from hurst_partitioning.estimators import HurstEstimate, _dfa2, block_bootstrap_ci
from hurst_partitioning.io import make_fgn_corpus, make_regime_switching_corpus
from hurst_partitioning.partitioner import Partition


def test_package_version_pinned() -> None:
    assert __version__ == "0.2.1"


def test_hurst_estimate_dataclass_constructs() -> None:
    est = HurstEstimate(
        estimator="dfa",
        point=0.65,
        ci_low=0.58,
        ci_high=0.72,
        level=0.95,
        n_bootstrap=1000,
        window=500,
    )
    assert est.estimator == "dfa"
    assert 0 <= est.ci_low <= est.point <= est.ci_high <= 1


def test_fixed_daily_boundaries_on_daily_index() -> None:
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    series = pd.Series(np.arange(10), index=idx)
    part = FixedDailyPartitioner().fit(series)
    assert part.boundaries[0] == 0
    assert len(part.boundaries) == 10


def test_fixed_monthly_boundaries_across_month_change() -> None:
    idx = pd.date_range("2026-01-28", periods=10, freq="D")
    series = pd.Series(np.arange(10), index=idx)
    part = FixedMonthlyPartitioner().fit(series)
    assert part.boundaries[0] == 0
    assert len(part.boundaries) >= 2


def test_equal_rows_partitioner_count() -> None:
    series = pd.Series(np.arange(25_000))
    part = EqualRowsPartitioner(chunk_size=10_000).fit(series)
    assert part.boundaries.tolist() == [0, 10_000, 20_000]


def test_oracle_partitioner_uses_known_breaks() -> None:
    series = pd.Series(np.arange(1000))
    breaks = np.array([100, 300, 600])
    part = OraclePartitioner(known_breaks=breaks).fit(series)
    assert part.boundaries.tolist() == [0, 100, 300, 600]


def test_variance_cusum_returns_partition_with_first_boundary_zero() -> None:
    # 1000 white-noise points then 1000 with 5x larger variance: CUSUM
    # on |first differences| should fire near the regime change.
    rng = np.random.default_rng(0)
    pre = rng.standard_normal(1000)
    post = 5.0 * rng.standard_normal(1000)
    series = pd.Series(np.concatenate([pre, post]))
    part = VarianceCusumPartitioner(h=5.0).fit(series)
    assert part.boundaries[0] == 0
    # At least one regime-break boundary should fire inside the post window.
    interior = part.boundaries[1:]
    assert (interior >= 900).any() and (interior <= 1500).any()


def test_make_fgn_corpus_deterministic_from_seed() -> None:
    a = make_fgn_corpus(seed=20260514, n_series=8)
    b = make_fgn_corpus(seed=20260514, n_series=8)
    assert len(a) == len(b) == 8
    for sa, sb in zip(a, b):
        assert sa.attrs["true_H"] == sb.attrs["true_H"]
        assert sa.attrs["length"] == sb.attrs["length"]
        assert sa.attrs["seed"] == sb.attrs["seed"]
        assert np.array_equal(sa.values, sb.values)
        assert "true_H" in sa.attrs
        assert isinstance(sa.index, pd.DatetimeIndex)


def test_make_fgn_corpus_series_id_invariant_to_n_series() -> None:
    """Per spec: ``(length, H)`` per series id is deterministic from seed,
    independent of how many series are requested."""
    short = make_fgn_corpus(seed=20260514, n_series=4)
    long = make_fgn_corpus(seed=20260514, n_series=12)
    for i in range(4):
        assert short[i].attrs["true_H"] == long[i].attrs["true_H"]
        assert short[i].attrs["length"] == long[i].attrs["length"]
        assert np.array_equal(short[i].values, long[i].values)


def test_dfa2_recovers_hurst_on_synth_fgn_h07() -> None:
    """DFA(2) on a 16384-point fGn with true H=0.7 returns approximately 0.7."""
    # series_id=2 has H=0.70 len=65536 per the registered seed.
    corpus = make_fgn_corpus(seed=20260514, n_series=4)
    persistent = [s for s in corpus if abs(s.attrs["true_H"] - 0.7) < 1e-9]
    assert persistent, "expected at least one H=0.7 series in n=4 corpus"
    s = persistent[0]
    H_est = _dfa2(s.values)
    assert abs(H_est - 0.7) < 0.1, f"DFA(2) recovered {H_est}, expected ~0.7"


def test_block_bootstrap_ci_returns_finite_quantiles() -> None:
    rng = np.random.default_rng(0)
    series = rng.standard_normal(1024)
    lo, hi = block_bootstrap_ci(
        series, lambda x: float(np.std(x)), n_bootstrap=100, level=0.95
    )
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


def test_generate_workload_counts() -> None:
    qs = generate_workload(series_len=4096, seed=20260514)
    assert len(qs) == 100
    kinds = [q.kind for q in qs]
    assert kinds.count("range-scan") == 40
    assert kinds.count("aggregate") == 30
    assert kinds.count("windowed-regression") == 20
    assert kinds.count("anomaly-scan") == 10
    for q in qs:
        assert 0 <= q.start_idx < q.end_idx <= 4096


def test_generate_workload_deterministic() -> None:
    a = generate_workload(series_len=4096, seed=20260514)
    b = generate_workload(series_len=4096, seed=20260514)
    assert [(q.kind, q.start_idx, q.end_idx) for q in a] == [
        (q.kind, q.start_idx, q.end_idx) for q in b
    ]


def test_measure_counts_touched_chunks_and_bytes() -> None:
    # Equal-rows partition with 5 chunks: [0,100), [100,200), [200,300),
    # [300,400), [400,+inf). Chunks 0..3 each have size 100 (next-start
    # minus start); the last chunk's size depends on the query's reach.
    part = Partition(
        boundaries=np.array([0, 100, 200, 300, 400]),
        metadata={"policy": "equal-rows", "chunk_size": 100},
    )
    # Query [150, 320) overlaps chunks 1, 2, 3 (each size 100).
    q = Query(kind="range-scan", start_idx=150, end_idx=320, payload={})
    res = measure(part, q, row_byte_size=16)
    assert res.chunks_touched == 3
    assert res.bytes_read == 3 * 100 * 16

    # Query [350, 600) overlaps chunks 3 and 4. Chunk 3 has size 100 (start
    # 300, next 400). Chunk 4 is the LAST chunk and its size is capped at
    # max(end_idx - start_4, 1) = max(600 - 400, 1) = 200. Total bytes
    # = (100 + 200) * 16 = 4800.
    q2 = Query(kind="range-scan", start_idx=350, end_idx=600, payload={})
    res2 = measure(part, q2, row_byte_size=16)
    assert res2.chunks_touched == 2
    assert res2.bytes_read == (100 + 200) * 16


def test_make_regime_switching_corpus_deterministic() -> None:
    """D4 generator must be deterministic and n_series-invariant for the
    first k series (Amendment 1)."""
    a = make_regime_switching_corpus(seed=20260514, n_series=4)
    b = make_regime_switching_corpus(seed=20260514, n_series=4)
    assert len(a) == len(b) == 4
    for sa, sb in zip(a, b):
        assert np.array_equal(sa.attrs["true_Hs"], sb.attrs["true_Hs"])
        assert sa.attrs["seed"] == sb.attrs["seed"]
        assert np.array_equal(sa.values, sb.values)
        assert sa.attrs["dataset_id"] == "D4"
        assert len(sa) == 16384
        assert np.array_equal(
            sa.attrs["true_boundaries"], np.array([4096, 8192, 12288])
        )
        assert isinstance(sa.index, pd.DatetimeIndex)
    # n_series-invariance: a longer call returns the same first k series.
    longer = make_regime_switching_corpus(seed=20260514, n_series=12)
    for i in range(4):
        assert np.array_equal(a[i].attrs["true_Hs"], longer[i].attrs["true_Hs"])
        assert a[i].attrs["seed"] == longer[i].attrs["seed"]
        assert np.array_equal(a[i].values, longer[i].values)


def test_make_regime_switching_corpus_adjacent_H_gap() -> None:
    """Adjacent segments must differ in H by at least 0.15 (Amendment 1)."""
    corpus = make_regime_switching_corpus(seed=20260514, n_series=16)
    for s in corpus:
        Hs = s.attrs["true_Hs"]
        gaps = np.abs(np.diff(Hs))
        assert (gaps >= 0.15 - 1e-12).all(), (
            f"series {s.attrs['series_id']} has Hs={Hs.tolist()} with adjacency "
            f"gaps {gaps.tolist()}"
        )


def test_boundary_f1_perfect_recovery() -> None:
    gt = np.array([4096, 8192, 12288])
    assert boundary_f1(gt, gt, tolerance=50) == 1.0


def test_boundary_f1_within_tolerance() -> None:
    gt = np.array([4096, 8192, 12288])
    within = np.array([4100, 8200, 12300])  # max |diff| = 12; tol=50 -> all TP
    assert boundary_f1(within, gt, tolerance=50) == 1.0
    outside = np.array([5000, 9000, 13000])  # |diff| > 50 each -> all FP, all FN
    assert boundary_f1(outside, gt, tolerance=50) < 1.0
    # Each ground-truth boundary may match at most one predicted boundary:
    # two predictions clustered around one gt should yield TP=1, FP=1.
    cluster = np.array([4096, 4097])
    f1 = boundary_f1(cluster, np.array([4096, 9000]), tolerance=50)
    # tp=1, fp=1, fn=1 -> precision=0.5, recall=0.5, f1=0.5
    assert abs(f1 - 0.5) < 1e-12


def test_boundary_f1_no_predicted_returns_zero() -> None:
    gt = np.array([4096, 8192, 12288])
    assert boundary_f1(np.array([], dtype=int), gt, tolerance=50) == 0.0


def test_boundary_f1_empty_ground_truth_raises() -> None:
    with pytest.raises(ValueError):
        boundary_f1(np.array([4096]), np.array([], dtype=int))


def test_paired_wilcoxon_with_holm_table_shape() -> None:
    rng = np.random.default_rng(0)
    # Candidate consistently smaller by ~3 on 100 paired observations.
    obs = {
        "lab-A": np.column_stack(
            (rng.standard_normal(100), rng.standard_normal(100) + 3.0)
        ),
        "lab-B": np.column_stack(
            (rng.standard_normal(100), rng.standard_normal(100) + 0.05)
        ),
    }
    df = paired_wilcoxon_with_holm(obs, ranking=["lab-A", "lab-B"], alpha=0.05)
    assert list(df.columns) == [
        "label", "p", "p_holm", "reject", "hl_shift", "hl_ci_low", "hl_ci_high"
    ]
    assert set(df["label"]) == {"lab-A", "lab-B"}
    # lab-A has a strong negative shift, lab-B does not.
    row_a = df.set_index("label").loc["lab-A"]
    assert row_a["hl_shift"] < 0
    assert bool(row_a["reject"]) is True
