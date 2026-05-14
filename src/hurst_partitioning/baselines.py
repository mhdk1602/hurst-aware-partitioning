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
    """B3. CUSUM change-point detection on |first differences|.

    Implements Page (1954) recursive CUSUM on standardized |returns|:

        diffs = |x[t+1] - x[t]|
        z[t]  = (diffs[t] - mean(diffs)) / std(diffs)
        S[t]  = max(0, S[t-1] + z[t] - k)
        if S[t] > h:  boundary at t,  reset S[t] = 0

    A boundary fires when the cumulative standardized positive drift exceeds
    ``h``; the accumulator is reset on each fire. The drift parameter ``k``
    is the conventional 0.5 (half a standard deviation of slack).

    The CUSUM threshold ``h`` is the only tunable parameter in the registered
    family. It MUST be tuned on D3-train (synthetic, half-by-parity); never
    on D1 or D2.
    """

    name = "variance-cusum"

    def __init__(self, h: float = 5.0, k: float = 0.5) -> None:
        self.h = h
        self.k = k

    def fit(self, series: pd.Series) -> Partition:
        values = np.asarray(series.values, dtype=float).ravel()
        n = values.size
        if n < 3:
            return Partition(
                boundaries=np.asarray([0], dtype=np.int64),
                metadata={"policy": self.name, "h": self.h, "k": self.k},
            )
        diffs = np.abs(np.diff(values))
        mu = float(diffs.mean())
        sigma = float(diffs.std())
        if not np.isfinite(sigma) or sigma <= 1e-12:
            # Degenerate (constant) series: no boundaries.
            return Partition(
                boundaries=np.asarray([0], dtype=np.int64),
                metadata={
                    "policy": self.name,
                    "h": self.h,
                    "k": self.k,
                    "note": "degenerate-zero-variance",
                },
            )
        z = (diffs - mu) / sigma

        boundaries: list[int] = [0]
        S = 0.0
        for t in range(z.size):
            S = max(0.0, S + z[t] - self.k)
            if S > self.h:
                # diffs[t] corresponds to the increment from index t to t+1
                # in the original series. The boundary lands at t+1.
                pos = t + 1
                if pos > boundaries[-1]:
                    boundaries.append(pos)
                S = 0.0

        return Partition(
            boundaries=np.asarray(boundaries, dtype=np.int64),
            metadata={
                "policy": self.name,
                "h": self.h,
                "k": self.k,
                "mu_abs_diff": mu,
                "sigma_abs_diff": sigma,
            },
        )


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
