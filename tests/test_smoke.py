"""Smoke tests for the v0.1.0-prereg scaffold.

The scaffold is intentionally NotImplementedError for the analysis-bearing
functions; the smoke test confirms imports, dataclass construction, and
the trivial partitioners (FIXED-DAILY, FIXED-MONTHLY, EQUAL-ROWS, ORACLE)
that do not depend on the unwritten estimator code.
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
from hurst_partitioning.estimators import HurstEstimate


def test_package_version_pinned() -> None:
    assert __version__ == "0.1.0"


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


def test_variance_cusum_unimplemented_in_prereg() -> None:
    series = pd.Series(np.random.default_rng(0).standard_normal(100))
    with pytest.raises(NotImplementedError):
        VarianceCusumPartitioner().fit(series)
