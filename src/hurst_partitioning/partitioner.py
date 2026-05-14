"""Chunk-boundary partitioning policies.

Defines the abstract Partitioner protocol and the registered candidate
policy (HurstCI). Baselines live in baselines.py; see prereg/h2-prereg-v1.md
§6 for the registered specification.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import __version__
from .estimators import EstimatorName, _dfa2, _ghe, _rs, _wavelet, block_bootstrap_ci


@dataclass(frozen=True)
class Partition:
    """A single chunk-boundary plan over a time-indexed series.

    Attributes
    ----------
    boundaries : np.ndarray
        Integer positions in the series at which chunks start. The first
        element is always 0. The implicit last boundary is len(series).
    metadata : dict
        Free-form audit fields (estimator, threshold, version) preserved for
        the deviations log.
    """

    boundaries: np.ndarray
    metadata: dict


class Partitioner(ABC):
    """Abstract chunk-boundary policy. Every baseline implements this."""

    name: str

    @abstractmethod
    def fit(self, series: pd.Series) -> Partition:
        """Compute boundaries for a single series."""
        raise NotImplementedError


_ESTIMATOR_DISPATCH = {
    "dfa": _dfa2,
    "rs": _rs,
    "ghe": _ghe,
    "wavelet": _wavelet,
}


class HurstCIPartitioner(Partitioner):
    """The registered CANDIDATE partitioner (prereg §6).

    A boundary is placed at position t iff:
        1. The rolling-Hurst CI on the window ending at t does NOT overlap
           the CI on the window ending at t - step, AND
        2. The two point estimates differ by at least ``min_point_gap``.

    Both conditions are required, so the policy is conservative against
    false-positive boundaries. The gap-of-0.1 default is registered; do NOT
    tune it on test data.
    """

    name = "hurst-ci"

    def __init__(
        self,
        estimator: EstimatorName = "dfa",
        window: int = 500,
        step: int = 20,
        min_point_gap: float = 0.1,
        ci_level: float = 0.95,
        n_bootstrap: int = 1000,
        rng_seed: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.window = window
        self.step = step
        self.min_point_gap = min_point_gap
        self.ci_level = ci_level
        self.n_bootstrap = n_bootstrap
        self.rng_seed = rng_seed

    def fit(self, series: pd.Series) -> Partition:
        """Roll the estimator across the series and locate regime breaks.

        For each integer position ``t`` that is a multiple of ``step`` and
        satisfies ``t >= window``, compute the Hurst estimate and its
        block-bootstrap CI on ``series[t - window : t]``. A boundary fires
        at ``t`` iff the CI at ``t`` does not overlap the CI at ``t - step``
        AND the absolute difference between the two point estimates is at
        least ``min_point_gap``.
        """
        values = np.asarray(series.values, dtype=float).ravel()
        n = values.size
        estimator_fn = _ESTIMATOR_DISPATCH[self.estimator]
        rng = np.random.default_rng(self.rng_seed)

        # Roll through the series.
        positions: list[int] = []
        points: list[float] = []
        ci_lows: list[float] = []
        ci_highs: list[float] = []
        t = self.window
        while t <= n:
            win = values[t - self.window : t]
            try:
                point = float(estimator_fn(win))
                lo, hi = block_bootstrap_ci(
                    win,
                    estimator_fn,
                    n_bootstrap=self.n_bootstrap,
                    level=self.ci_level,
                    rng=rng,
                )
            except Exception:
                # Skip degenerate windows rather than aborting the partition.
                t += self.step
                continue
            positions.append(t)
            points.append(point)
            ci_lows.append(lo)
            ci_highs.append(hi)
            t += self.step

        # Boundary detection: compare each (t) against the previous (t-step).
        boundaries: list[int] = [0]
        for i in range(1, len(positions)):
            ci_disjoint = (ci_lows[i] > ci_highs[i - 1]) or (
                ci_highs[i] < ci_lows[i - 1]
            )
            point_gap_ok = abs(points[i] - points[i - 1]) >= self.min_point_gap
            if ci_disjoint and point_gap_ok:
                boundaries.append(positions[i])

        meta = {
            "policy": self.name,
            "estimator": self.estimator,
            "window": self.window,
            "step": self.step,
            "min_point_gap": self.min_point_gap,
            "ci_level": self.ci_level,
            "n_bootstrap": self.n_bootstrap,
            "package_version": __version__,
            "n_windows": len(positions),
        }
        return Partition(
            boundaries=np.asarray(boundaries, dtype=np.int64), metadata=meta
        )
