"""Hurst exponent estimators with block-bootstrap confidence intervals.

Pre-registered estimator battery (see prereg/h2-prereg-v1.md §5):
    - DFA(2)  — primary
    - R/S     — robustness
    - GHE q=2 — robustness
    - Wavelet — informational only

All estimators return (point_estimate, ci_low, ci_high) at the requested
bootstrap level. Block length follows Politis-Romano 1994 with l = W^(1/3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

EstimatorName = Literal["dfa", "rs", "ghe", "wavelet"]


@dataclass(frozen=True)
class HurstEstimate:
    """A single Hurst estimate with its block-bootstrap CI.

    The ``ci_low`` and ``ci_high`` fields are the empirical quantiles at
    (1 - level) / 2 and 1 - (1 - level) / 2 of the bootstrap distribution.
    """

    estimator: EstimatorName
    point: float
    ci_low: float
    ci_high: float
    level: float
    n_bootstrap: int
    window: int


class Estimator(Protocol):
    """Common signature for every registered Hurst estimator."""

    name: EstimatorName

    def estimate(self, series: np.ndarray) -> float: ...


def _dfa2(series: np.ndarray) -> float:
    """DFA(2) point estimate. Implementation deferred to nolds in execution.

    The protocol requires nolds.dfa with order=2; the wrapper here is a
    placeholder that raises in the scaffold release so callers cannot
    accidentally treat the scaffold as a working analysis pipeline.
    """
    raise NotImplementedError(
        "DFA(2) point estimate is unimplemented in the v0.1.0-prereg scaffold. "
        "Implementation will land at v0.2.0 along with the D3 pilot. See "
        "prereg/h2-prereg-v1.md §5 for the registered specification."
    )


def _rs(series: np.ndarray) -> float:
    """Rescaled-range estimate. Same scaffold contract as _dfa2."""
    raise NotImplementedError("R/S point estimate is unimplemented in v0.1.0-prereg.")


def _ghe(series: np.ndarray, q: float = 2.0) -> float:
    """Generalized Hurst exponent at order q. Same scaffold contract."""
    raise NotImplementedError("GHE point estimate is unimplemented in v0.1.0-prereg.")


def _wavelet(series: np.ndarray) -> float:
    """Wavelet (Daubechies db4, level 6) Hurst estimate. Same scaffold contract."""
    raise NotImplementedError("Wavelet point estimate is unimplemented in v0.1.0-prereg.")


def block_bootstrap_ci(
    series: np.ndarray,
    estimator_fn,
    n_bootstrap: int = 1000,
    level: float = 0.95,
    block_length: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Politis-Romano stationary block bootstrap CI for a Hurst estimator.

    Parameters
    ----------
    series : 1-D array
        The series on which the estimator operates.
    estimator_fn : callable
        Accepts a 1-D array and returns a scalar Hurst estimate.
    n_bootstrap : int
        Number of bootstrap replicates. The pre-registration fixes this at 1000.
    level : float
        Confidence level. The pre-registration fixes this at 0.95.
    block_length : int, optional
        Expected block length. If None, set to ``int(round(len(series) ** (1/3)))``
        per Politis-Romano 1994.
    rng : numpy.random.Generator, optional
        For reproducibility under the registered protocol seeds.

    Returns
    -------
    (ci_low, ci_high) : tuple[float, float]
        Empirical quantiles of the bootstrap distribution.

    Notes
    -----
    Unimplemented in the v0.1.0-prereg scaffold. See estimators._dfa2 docstring.
    """
    raise NotImplementedError(
        "block_bootstrap_ci is unimplemented in v0.1.0-prereg. "
        "The pre-registration fixes the contract; the implementation lands at v0.2.0."
    )


def estimate(
    series: np.ndarray,
    estimator: EstimatorName,
    window: int,
    n_bootstrap: int = 1000,
    level: float = 0.95,
) -> HurstEstimate:
    """Top-level entry point. Returns a HurstEstimate with CI."""
    dispatch = {"dfa": _dfa2, "rs": _rs, "ghe": _ghe, "wavelet": _wavelet}
    fn = dispatch[estimator]
    point = fn(series)
    ci_low, ci_high = block_bootstrap_ci(
        series, fn, n_bootstrap=n_bootstrap, level=level
    )
    return HurstEstimate(
        estimator=estimator,
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        level=level,
        n_bootstrap=n_bootstrap,
        window=window,
    )
