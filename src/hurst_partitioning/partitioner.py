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

from .estimators import EstimatorName


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


class HurstCIPartitioner(Partitioner):
    """The registered CANDIDATE partitioner (prereg §6).

    A boundary is placed at position t iff:
        1. The rolling-Hurst CI on window (t-W, t) does NOT overlap
           the CI on window (t-2W, t-W), AND
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
    ) -> None:
        self.estimator = estimator
        self.window = window
        self.step = step
        self.min_point_gap = min_point_gap
        self.ci_level = ci_level
        self.n_bootstrap = n_bootstrap

    def fit(self, series: pd.Series) -> Partition:
        raise NotImplementedError(
            "HurstCIPartitioner.fit is unimplemented in the v0.1.0-prereg scaffold. "
            "See prereg/h2-prereg-v1.md §6 for the registered specification."
        )
