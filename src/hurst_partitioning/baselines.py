"""Pre-registered baseline partitioning policies (prereg §6).

B1 FIXED-DAILY     — 1-day chunks, TimescaleDB default for daily series.
B2 FIXED-MONTHLY   — 30-day chunks.
B3 VARIANCE-CUSUM  — CUSUM-detected change points on |returns|. CUSUM
                     threshold tuned on D3-train ONLY.
B4 EQUAL-ROWS      — 10,000-row chunks regardless of time.
B5 ORACLE          — known regime breaks on D3 only. Upper bound, not
                     part of the primary comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .partitioner import Partition, Partitioner


class FixedDailyPartitioner(Partitioner):
    """B1. One chunk per calendar day."""

    name = "fixed-daily"

    def fit(self, series: pd.Series) -> Partition:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("FixedDailyPartitioner requires a DatetimeIndex.")
        day_change = series.index.normalize().diff().fillna(pd.Timedelta(0))
        boundaries = np.flatnonzero(day_change.values != pd.Timedelta(0))
        boundaries = np.insert(boundaries, 0, 0)
        return Partition(boundaries=boundaries, metadata={"policy": self.name})


class FixedMonthlyPartitioner(Partitioner):
    """B2. Calendar-month chunks."""

    name = "fixed-monthly"

    def fit(self, series: pd.Series) -> Partition:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("FixedMonthlyPartitioner requires a DatetimeIndex.")
        ym = series.index.year * 12 + series.index.month
        change = np.diff(ym, prepend=ym[0] - 1)
        boundaries = np.flatnonzero(change != 0)
        return Partition(boundaries=boundaries, metadata={"policy": self.name})


class VarianceCusumPartitioner(Partitioner):
    """B3. CUSUM change-point detection on |returns|.

    The CUSUM threshold ``h`` is the only tunable parameter in the registered
    family. It MUST be tuned on D3-train (synthetic, half-by-parity); never
    on D1 or D2.
    """

    name = "variance-cusum"

    def __init__(self, h: float = 5.0) -> None:
        self.h = h

    def fit(self, series: pd.Series) -> Partition:
        raise NotImplementedError("VarianceCusumPartitioner unimplemented in v0.1.0-prereg.")


class EqualRowsPartitioner(Partitioner):
    """B4. Equal-row chunks. Default 10,000."""

    name = "equal-rows"

    def __init__(self, chunk_size: int = 10_000) -> None:
        self.chunk_size = chunk_size

    def fit(self, series: pd.Series) -> Partition:
        n = len(series)
        boundaries = np.arange(0, n, self.chunk_size)
        return Partition(
            boundaries=boundaries,
            metadata={"policy": self.name, "chunk_size": self.chunk_size},
        )


class OraclePartitioner(Partitioner):
    """B5. Known regime breaks. Only applicable to D3 (synthetic)."""

    name = "oracle"

    def __init__(self, known_breaks: np.ndarray) -> None:
        self.known_breaks = known_breaks

    def fit(self, series: pd.Series) -> Partition:
        return Partition(
            boundaries=np.unique(np.concatenate([[0], self.known_breaks])),
            metadata={"policy": self.name},
        )
