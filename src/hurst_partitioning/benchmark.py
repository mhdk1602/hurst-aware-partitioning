"""Benchmark harness — query workload, I/O accounting, paired Wilcoxon.

Implements the §7-§11 protocol of the pre-registration. Workload generation
is deterministic from the registered seed and is archived to
experiments/query-library-v1.json on first run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .partitioner import Partition

QueryKind = Literal["range-scan", "aggregate", "windowed-regression", "anomaly-scan"]


@dataclass(frozen=True)
class Query:
    """A single registered query against a time-series dataset."""

    kind: QueryKind
    start_idx: int
    end_idx: int
    payload: dict


@dataclass(frozen=True)
class WorkloadResult:
    """Per-query measurement under one partitioning policy."""

    policy: str
    query_idx: int
    bytes_read: int
    chunks_touched: int
    latency_us: int | None


def generate_workload(
    series_len: int, seed: int = 20260514
) -> list[Query]:
    """Deterministic 100-query workload (prereg §7).

    Counts: 40 range scans, 30 aggregates, 20 windowed regressions, 10 anomaly scans.
    The seed is registered and must NOT be changed across runs.
    """
    raise NotImplementedError(
        "generate_workload is unimplemented in v0.1.0-prereg. "
        "See prereg/h2-prereg-v1.md §7 for the registered specification."
    )


def archive_workload(queries: list[Query], path: Path) -> str:
    """Write the workload to disk, return SHA-256 hash for the deviations log."""
    raise NotImplementedError("archive_workload unimplemented in v0.1.0-prereg.")


def measure(partition: Partition, query: Query, row_byte_size: int = 16) -> WorkloadResult:
    """Count bytes touched by ``query`` under ``partition``.

    A chunk is "touched" if it overlaps the query's index range. Bytes-read
    counts the full chunk byte size (chunk-level accounting). This is the
    primary outcome metric M-PRIMARY.
    """
    raise NotImplementedError("measure unimplemented in v0.1.0-prereg.")


def paired_wilcoxon_with_holm(
    paired_observations: dict[str, np.ndarray],
    ranking: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Holm-Bonferroni over the registered pairwise comparisons.

    Parameters
    ----------
    paired_observations : dict[label, ndarray]
        For each (dataset, baseline, estimator) triple, the per-query
        bytes-read for the CANDIDATE minus the bytes-read for the baseline.
    ranking : list[str]
        Pre-registered importance ranking of the labels (prereg §11).
    alpha : float
        Family-wise error rate.

    Returns
    -------
    pandas.DataFrame
        One row per comparison: p-value, Holm-adjusted p, reject decision,
        Hodges-Lehmann shift estimator with 95% block-bootstrap CI.
    """
    raise NotImplementedError("paired_wilcoxon_with_holm unimplemented in v0.1.0-prereg.")
