"""Benchmark harness — query workload, I/O accounting, paired Wilcoxon.

Implements the §7-§11 protocol of the pre-registration. Workload generation
is deterministic from the registered seed and is archived to
experiments/query-library-v1.json on first run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
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


def _sample_range(
    rng: np.random.Generator,
    series_len: int,
    biased_lengths: tuple[int, ...],
) -> tuple[int, int]:
    """Pick a (start, end) interval. End is biased to one of ``biased_lengths``
    after start; if that overshoots the series, fall back to a uniform end."""
    start = int(rng.integers(0, max(1, series_len - 1)))
    length = int(rng.choice(biased_lengths))
    end = start + length
    if end > series_len:
        end = int(rng.integers(start + 1, series_len + 1))
    return start, end


def generate_workload(
    series_len: int, seed: int = 20260514
) -> list[Query]:
    """Deterministic 100-query workload (prereg §7).

    Counts: 40 range scans, 30 aggregates, 20 windowed regressions, 10 anomaly scans.
    Range scans bias the end-start length toward {30, 90, 365}-style intervals.
    Aggregates similarly bias the range length but draw from the same pool.
    Windowed regressions use a 60-point window. Anomaly scans use 30-point ranges.
    Within each kind, indices are uniform random in ``[0, series_len)``.

    The seed is registered and must NOT be changed across runs.
    """
    rng = np.random.default_rng(seed)
    queries: list[Query] = []

    # 40 range scans. Bias the chunk length toward 30, 90, 365 "days"
    # (treated as raw row counts under the synthetic minute index).
    biased = (30, 90, 365)
    for _ in range(40):
        start, end = _sample_range(rng, series_len, biased)
        queries.append(
            Query(
                kind="range-scan",
                start_idx=start,
                end_idx=end,
                payload={"length_bias": list(biased)},
            )
        )

    # 30 aggregates over a range (mean, std, p95, max). Cycle through the
    # four aggregate kinds.
    agg_kinds = ("mean", "std", "p95", "max")
    for i in range(30):
        start, end = _sample_range(rng, series_len, biased)
        queries.append(
            Query(
                kind="aggregate",
                start_idx=start,
                end_idx=end,
                payload={"agg": agg_kinds[i % 4]},
            )
        )

    # 20 windowed-regression queries (slope of value on lagged value over
    # a 60-row window). 60 is a placeholder for "60-day" under the prereg.
    for _ in range(20):
        start = int(rng.integers(0, max(1, series_len - 60)))
        end = min(start + 60, series_len)
        queries.append(
            Query(
                kind="windowed-regression",
                start_idx=start,
                end_idx=end,
                payload={"lag": 1, "window": 60},
            )
        )

    # 10 anomaly-detection queries (range scan + threshold trigger). Use a
    # 30-row range; threshold value is informational only.
    for _ in range(10):
        start = int(rng.integers(0, max(1, series_len - 30)))
        end = min(start + 30, series_len)
        queries.append(
            Query(
                kind="anomaly-scan",
                start_idx=start,
                end_idx=end,
                payload={"threshold_sigma": 3.0},
            )
        )

    return queries


def archive_workload(queries: list[Query], path: Path) -> str:
    """Write the workload to disk, return SHA-256 hash for the deviations log."""
    payload = [asdict(q) for q in queries]
    text = json.dumps(payload, indent=2, sort_keys=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    (path.with_suffix(path.suffix + ".sha256")).write_text(digest + "\n")
    return digest


def _chunk_sizes_from_boundaries(
    boundaries: np.ndarray, series_len: int
) -> np.ndarray:
    """Convert chunk-start positions to chunk lengths (one per chunk)."""
    b = np.asarray(boundaries, dtype=np.int64)
    if b.size == 0:
        return np.asarray([series_len], dtype=np.int64)
    ends = np.append(b[1:], series_len)
    return ends - b


def measure(
    partition: Partition, query: Query, row_byte_size: int = 16
) -> WorkloadResult:
    """Count chunks touched and bytes read for ``query`` under ``partition``.

    A chunk ``[b_i, b_{i+1})`` is "touched" if it overlaps the half-open
    interval ``[query.start_idx, query.end_idx)``. Bytes read counts the
    full chunk-byte-size for every touched chunk. Latency is None.
    """
    boundaries = np.asarray(partition.boundaries, dtype=np.int64)
    if boundaries.size == 0:
        boundaries = np.asarray([0], dtype=np.int64)
    # We need to know the implicit last-chunk size, but the partition does
    # not store the series length. Recover it from the query end as a lower
    # bound; for accurate accounting the partition's last chunk extends to
    # the end of the series. We use query.end_idx as an upper bound on the
    # last accessed chunk: a chunk is touched if start_b < end_idx AND
    # end_b > start_idx. The end of the LAST chunk is conservatively
    # treated as +infinity so any chunk whose start is below end_idx counts
    # as touching the query. This is consistent with chunk-level I/O
    # accounting in storage engines.
    starts = boundaries
    # Per-chunk end: next-start, with last chunk extending forever.
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = np.iinfo(np.int64).max

    qs, qe = int(query.start_idx), int(query.end_idx)
    touched = (starts < qe) & (ends > qs)
    chunks_touched = int(touched.sum())

    # For the byte count we need a finite chunk size for the last chunk.
    # The prereg's primary metric M-PRIMARY counts "full chunk byte size"
    # for any overlapping chunk. Use the query end as a conservative cap
    # for the last chunk if it extends past it; otherwise use the larger
    # of (qe, max(starts)+1). This avoids assuming the global series end.
    sizes = np.empty_like(starts, dtype=np.int64)
    sizes[:-1] = starts[1:] - starts[:-1]
    last_size = max(qe - int(starts[-1]), 1)
    sizes[-1] = last_size

    bytes_read = int((sizes[touched]).sum()) * int(row_byte_size)

    return WorkloadResult(
        policy=str(partition.metadata.get("policy", "unknown")),
        query_idx=-1,
        bytes_read=bytes_read,
        chunks_touched=chunks_touched,
        latency_us=None,
    )


# ---------------------------------------------------------------------------
# Statistical analysis: paired Wilcoxon with Holm-Bonferroni correction
# ---------------------------------------------------------------------------


def _hodges_lehmann_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Median of all pairwise differences a_i - b_j.

    For paired data the HL estimator of a location shift is conventionally
    the median of all pairwise sums (a_i + a_j) / 2 — but for the *shift
    between two samples* the natural pair-difference HL estimator is the
    median of a_i - b_j. We use the cross-pairwise form, which matches the
    classical two-sample shift point estimate.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    diffs = (a[:, None] - b[None, :]).ravel()
    return float(np.median(diffs))


def _hl_block_bootstrap_ci(
    paired_diffs: np.ndarray,
    n_bootstrap: int = 1000,
    level: float = 0.95,
    block_length: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Block-bootstrap CI for the Hodges-Lehmann shift on paired differences.

    Resamples the per-query paired differences (candidate - baseline) with
    a Politis-Romano stationary block bootstrap, computes the median of
    pairwise sums divided-by-two on each replicate, and returns the
    (1-level)/2 and 1-(1-level)/2 empirical quantiles.

    Note: when applied to *paired differences*, the HL estimator simplifies
    to the median of pairwise sums (d_i + d_j) / 2 (Walsh averages).
    """
    d = np.asarray(paired_diffs, dtype=float).ravel()
    n = d.size
    if n < 4:
        raise ValueError("HL bootstrap CI needs at least 4 paired observations.")
    if block_length is None:
        block_length = max(1, int(round(n ** (1 / 3))))
    p = 1.0 / float(block_length)
    if rng is None:
        rng = np.random.default_rng()
    hl = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
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
        sample = d[idx]
        # Walsh averages: median of (sample[i] + sample[j]) / 2 over i<=j.
        i_idx, j_idx = np.triu_indices(n)
        walsh = (sample[i_idx] + sample[j_idx]) / 2.0
        hl[b] = np.median(walsh)
    alpha = 1.0 - level
    return float(np.quantile(hl, alpha / 2.0)), float(np.quantile(hl, 1 - alpha / 2.0))


def paired_wilcoxon_with_holm(
    paired_observations: dict[str, np.ndarray],
    ranking: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank with Holm-Bonferroni correction.

    Parameters
    ----------
    paired_observations : dict[label, ndarray]
        For each pre-registered comparison label, a 2-column array (or a
        2-row array) where column 0 is the CANDIDATE per-query bytes-read
        and column 1 is the BASELINE per-query bytes-read. Length must be
        the workload size (100). For convenience, a 1-D array of
        per-query *differences* (candidate - baseline) is also accepted.
    ranking : list[str]
        Pre-registered importance ranking of the labels (prereg §11).
        Determines the Holm step-down ordering when p-values tie.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    pandas.DataFrame
        One row per comparison: label, p, p_holm, reject, hl_shift,
        hl_ci_low, hl_ci_high. Hodges-Lehmann shift is the Walsh median
        of the paired differences (candidate - baseline); negative means
        the candidate read fewer bytes. CI via block bootstrap on the
        per-query differences (1000 replicates, 95% level).
    """
    from scipy.stats import wilcoxon

    labels = list(ranking)
    # Sanity-check that every label in the ranking has an observation.
    missing = [lab for lab in labels if lab not in paired_observations]
    if missing:
        raise KeyError(f"Missing observations for labels: {missing}")

    raw_rows: list[dict] = []
    pvals: list[float] = []
    diffs_per_label: dict[str, np.ndarray] = {}
    for lab in labels:
        obs = np.asarray(paired_observations[lab], dtype=float)
        if obs.ndim == 1:
            d = obs
            cand = None
            base = None
        elif obs.ndim == 2:
            # Accept (2, n) or (n, 2).
            if obs.shape[0] == 2:
                cand, base = obs[0], obs[1]
            elif obs.shape[1] == 2:
                cand, base = obs[:, 0], obs[:, 1]
            else:
                raise ValueError(
                    f"Observations for {lab} must be (2,n), (n,2), or 1-D."
                )
            d = cand - base
        else:
            raise ValueError(f"Observations for {lab} must be 1-D or 2-D.")
        diffs_per_label[lab] = d
        # Wilcoxon needs at least one nonzero difference.
        if np.all(d == 0):
            p = 1.0
        else:
            try:
                _, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            except Exception:
                p = 1.0
        if not np.isfinite(p):
            p = 1.0
        pvals.append(float(p))
        raw_rows.append({"label": lab, "p": float(p)})

    # Holm-Bonferroni step-down. Sort labels by ascending p; the i-th
    # smallest p is compared to alpha / (m - i) where m is the family
    # size. A label is rejected only if it and every label with smaller p
    # also pass their thresholds.
    m = len(labels)
    order = np.argsort(pvals)
    p_sorted = [pvals[i] for i in order]
    # Holm-adjusted p-values: p_holm_(i) = max_{k<=i} min(1, (m - k) * p_(k))
    running_max = 0.0
    p_holm_sorted = []
    for k, p in enumerate(p_sorted):
        adj = min(1.0, (m - k) * p)
        running_max = max(running_max, adj)
        p_holm_sorted.append(running_max)

    p_holm = [0.0] * m
    for rank, idx in enumerate(order):
        p_holm[idx] = p_holm_sorted[rank]

    # Hodges-Lehmann shift + block-bootstrap CI per comparison.
    out_rows = []
    for i, lab in enumerate(labels):
        d = diffs_per_label[lab]
        # Walsh averages for paired HL on differences.
        nd = d.size
        i_idx, j_idx = np.triu_indices(nd)
        walsh = (d[i_idx] + d[j_idx]) / 2.0
        hl = float(np.median(walsh))
        try:
            lo, hi = _hl_block_bootstrap_ci(d, n_bootstrap=1000, level=0.95)
        except Exception:
            lo, hi = float("nan"), float("nan")
        out_rows.append(
            {
                "label": lab,
                "p": pvals[i],
                "p_holm": p_holm[i],
                "reject": p_holm[i] < alpha,
                "hl_shift": hl,
                "hl_ci_low": lo,
                "hl_ci_high": hi,
            }
        )
    return pd.DataFrame(out_rows, columns=[
        "label", "p", "p_holm", "reject", "hl_shift", "hl_ci_low", "hl_ci_high"
    ])


# ---------------------------------------------------------------------------
# Boundary-quality F1 (Amendment 1, M-S6)
# ---------------------------------------------------------------------------


def boundary_f1(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    tolerance: int = 50,
) -> float:
    """F1 score of predicted partition boundaries against ground truth.

    A predicted boundary is a true positive iff it lies within ``tolerance`` rows
    of some ground-truth boundary. The matching is one-to-one and greedy by
    nearest-distance: each ground-truth boundary may be matched by at most one
    predicted boundary. The greedy assignment iterates over all (pred, gt) pairs
    in ascending distance and assigns the first available pair, then removes both
    from the pool.

    Parameters
    ----------
    predicted : np.ndarray
        1-D integer array of predicted boundary positions. May be empty.
    ground_truth : np.ndarray
        1-D integer array of ground-truth boundary positions. Must be non-empty.
    tolerance : int
        Maximum row distance for a predicted boundary to count as a match.
        Defaults to 50; the v0.2.1 pilot passes
        ``max(window // 4, 50)`` from the HURST-CI window.

    Returns
    -------
    float
        F1 = 2 * P * R / (P + R). Returns 0.0 if both P and R are 0, or if the
        predicted array is empty (no positives to score). NaN entries in either
        input are dropped before scoring.

    Raises
    ------
    ValueError
        If ``ground_truth`` is empty.
    """
    pred = np.asarray(predicted, dtype=float).ravel()
    gt = np.asarray(ground_truth, dtype=float).ravel()
    # NaN-safe: drop non-finite entries before integer coercion.
    pred = pred[np.isfinite(pred)]
    gt = gt[np.isfinite(gt)]
    if gt.size == 0:
        raise ValueError("boundary_f1: ground_truth must be non-empty.")
    pred = pred.astype(np.int64)
    gt = gt.astype(np.int64)
    if pred.size == 0:
        # No predicted boundaries: precision is undefined, recall is 0.
        # Conventional F1 in change-point detection literature is 0.
        return 0.0

    tol = int(tolerance)
    # Build all candidate (pred_idx, gt_idx, distance) triples within tol.
    # Greedy nearest-match assignment guarantees one-to-one matching.
    dist = np.abs(pred[:, None] - gt[None, :])
    pi, gi = np.where(dist <= tol)
    if pi.size == 0:
        tp = 0
    else:
        candidate_d = dist[pi, gi]
        order = np.argsort(candidate_d, kind="stable")
        used_pred = np.zeros(pred.size, dtype=bool)
        used_gt = np.zeros(gt.size, dtype=bool)
        tp = 0
        for idx in order:
            p = int(pi[idx])
            g = int(gi[idx])
            if used_pred[p] or used_gt[g]:
                continue
            used_pred[p] = True
            used_gt[g] = True
            tp += 1

    fp = int(pred.size) - tp
    fn = int(gt.size) - tp
    if tp == 0 and (fp + fn) == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))
