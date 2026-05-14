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
    """DFA(2) point estimate via ``nolds.dfa`` with ``order=2``.

    Per prereg §5, DFA(2) is the primary estimator. We delegate to nolds
    which implements the standard Peng et al. detrended-fluctuation
    procedure; the order=2 argument fits a quadratic to each segment
    before computing the residual variance.
    """
    import nolds

    arr = np.asarray(series, dtype=float).ravel()
    if arr.size < 8:
        raise ValueError(f"DFA(2) needs at least 8 points; got {arr.size}")
    return float(nolds.dfa(arr, order=2))


def _rs(series: np.ndarray) -> float:
    """Rescaled-range estimate via the ``hurst`` package.

    Uses ``kind='change'`` and ``simplified=True`` per spec, which treats
    the input as the increments and returns the slope of log(R/S) on log n.
    Returns only the Hurst exponent (the package returns a tuple).
    """
    import hurst as hurst_lib

    arr = np.asarray(series, dtype=float).ravel()
    if arr.size < 100:
        # The hurst library asks for at least ~100 points; below that the
        # estimator is unreliable. We raise to make this loud rather than
        # silently returning a near-meaningless number.
        raise ValueError(f"R/S needs at least 100 points; got {arr.size}")
    H, _c, _data = hurst_lib.compute_Hc(arr, kind="change", simplified=True)
    return float(H)


def _ghe(series: np.ndarray, q: float = 2.0) -> float:
    """Generalized Hurst exponent at order ``q`` via MFDFA.

    Per prereg §5 (robustness battery), we use MFDFA at the requested
    order ``q``. MFDFA integrates the series internally (treats input as
    increments) and reports the scaling exponent h(q); for q=2, h(2) is
    the Hurst exponent. Lags are log-spaced from 8 to N/4.
    """
    from MFDFA import MFDFA

    arr = np.asarray(series, dtype=float).ravel()
    n = arr.size
    if n < 64:
        raise ValueError(f"GHE needs at least 64 points; got {n}")
    max_lag = max(8, n // 4)
    lags = np.unique(np.logspace(np.log10(8), np.log10(max_lag), num=20).astype(int))
    lags = lags[lags >= 4]
    if lags.size < 4:
        raise ValueError("Not enough valid lags for GHE.")
    _lag_out, dfa = MFDFA(arr, lag=lags, q=q, order=1)
    valid = dfa[:, 0] > 0
    if valid.sum() < 4:
        raise ValueError("MFDFA returned too few positive fluctuations to fit.")
    slope, _intercept = np.polyfit(
        np.log(_lag_out[valid]), np.log(dfa[valid, 0]), 1
    )
    return float(slope)


def _wavelet(series: np.ndarray) -> float:
    """Wavelet (Daubechies db4, level 6) Hurst estimate.

    Per prereg §5, this is informational only. Implementation is deferred
    to v0.2.1+ pending PyWavelets being part of the install.
    """
    raise NotImplementedError(
        "Wavelet estimator is deferred to a later release. "
        "It is informational per prereg/h2-prereg-v1.md §5 and does not "
        "load-bear on the primary or robustness comparisons."
    )


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
        Empirical quantiles of the bootstrap distribution at
        (1 - level) / 2 and 1 - (1 - level) / 2.
    """
    arr = np.asarray(series, dtype=float).ravel()
    n = arr.size
    if n < 4:
        raise ValueError(f"block_bootstrap_ci needs at least 4 points; got {n}")
    if block_length is None:
        block_length = max(1, int(round(n ** (1 / 3))))
    p = 1.0 / float(block_length)
    if rng is None:
        rng = np.random.default_rng()

    estimates = np.empty(n_bootstrap, dtype=float)
    n_kept = 0
    for b in range(n_bootstrap):
        # Politis-Romano stationary bootstrap: at each step, with probability
        # p start a new block at a fresh uniform index; otherwise continue
        # the current block by one position (mod n).
        idx = np.empty(n, dtype=np.int64)
        cur = int(rng.integers(0, n))
        idx[0] = cur
        if n > 1:
            new_block = rng.random(n - 1) < p
            jumps = rng.integers(0, n, size=n - 1)
            for i in range(1, n):
                if new_block[i - 1]:
                    cur = int(jumps[i - 1])
                else:
                    cur = (cur + 1) % n
                idx[i] = cur
        sample = arr[idx]
        try:
            est = float(estimator_fn(sample))
        except Exception:
            # Bootstrap resamples can hit edge cases; skip failures rather
            # than failing the whole CI. We rebalance n_bootstrap downward.
            continue
        if not np.isfinite(est):
            continue
        estimates[n_kept] = est
        n_kept += 1

    if n_kept < max(10, n_bootstrap // 10):
        raise RuntimeError(
            f"Bootstrap CI failed: only {n_kept}/{n_bootstrap} replicates returned a finite estimate."
        )
    kept = estimates[:n_kept]
    alpha = 1.0 - level
    lo = float(np.quantile(kept, alpha / 2.0))
    hi = float(np.quantile(kept, 1.0 - alpha / 2.0))
    return lo, hi


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
