"""Dataset loaders. D1 yfinance, D2 NAB, D3 synthetic fGn corpus.

Provenance and pinning are part of the pre-registration (prereg §4 and §16).
Data files are gitignored; checksums and commit hashes are committed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REGISTERED_FGN_SEED = 20260514
SP500_50_TICKERS_FILE = Path(__file__).parent / "data" / "sp500_50_tickers.txt"

# Per prereg §4. These are immutable.
FGN_LENGTHS = (4096, 16384, 65536)
FGN_HURSTS = (0.3, 0.5, 0.7, 0.85)

# Amendment 1 (D4). Per-segment H pool and adjacency constraint.
D4_SEGMENT_HURSTS = (0.30, 0.55, 0.70, 0.85)
D4_SEGMENT_LENGTH = 4096
D4_N_SEGMENTS = 4
D4_MIN_ADJACENT_GAP = 0.15
D4_MAX_RESAMPLE_ATTEMPTS = 50


@dataclass(frozen=True)
class DatasetSpec:
    """Audit record for one acquired series."""

    dataset_id: str
    series_id: str
    sha256: str
    n_rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    source: str


def sha256_of_dataframe(df: pd.DataFrame) -> str:
    """Stable hash of a dataframe contents for the provenance manifest."""
    serialized = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(serialized).hexdigest()


def load_d1_sp500(cache_dir: Path) -> dict[str, pd.DataFrame]:
    """D1 — 50 S&P 500 tickers, daily OHLCV, 2015-01 to 2026-04.

    Per prereg §13, D1 cannot be analyzed until the pre-registration lock is
    final at G1 (2026-07). This loader is intentionally stubbed at v0.2.0.
    """
    raise NotImplementedError(
        "D1 acquisition is locked until 2026-07 per prereg §13. "
        "The v0.2.0 release covers D3 (synthetic) only."
    )


def load_d2_nab(nab_repo_dir: Path) -> dict[str, pd.DataFrame]:
    """D2 — 10 Numenta Anomaly Benchmark series.

    Per prereg §13, D2 cannot be analyzed until the pre-registration lock is
    final. Stubbed at v0.2.0.
    """
    raise NotImplementedError(
        "D2 acquisition is locked until 2026-07 per prereg §13. "
        "The v0.2.0 release covers D3 (synthetic) only."
    )


def make_fgn_corpus(
    seed: int = REGISTERED_FGN_SEED,
    n_series: int = 1024,
) -> list[pd.Series]:
    """D3 — synthetic fractional Gaussian noise corpus.

    Per prereg §4. Lengths sampled uniformly from {4096, 16384, 65536};
    H sampled uniformly from {0.3, 0.5, 0.7, 0.85}. The (length, H) per
    series id is deterministic from ``seed``. Each series gets a synthetic
    DatetimeIndex starting 2020-01-01 at 1-minute frequency, and the
    true H is recorded as ``s.attrs['true_H']`` for downstream verification.

    Parameters
    ----------
    seed : int
        The registered seed is 20260514. Changing this on a real run would
        be a §14 deviation and must be logged.
    n_series : int
        Default 1024 per the registered protocol. Tests pass smaller values.

    Returns
    -------
    list[pandas.Series]
        Length ``n_series``. Each Series has a DatetimeIndex, dtype float64,
        and at minimum ``attrs['true_H']``, ``attrs['length']``,
        ``attrs['series_id']``, ``attrs['seed']``.
    """
    from fbm import FBM

    # Per-series RNGs so the (length, H, seed) of series id ``sid`` are
    # independent of ``n_series``. We derive a deterministic per-series
    # seed from the master seed via SeedSequence.spawn.
    master = np.random.SeedSequence(int(seed))
    children = master.spawn(n_series)

    corpus: list[pd.Series] = []
    for sid in range(n_series):
        srng = np.random.default_rng(children[sid])
        length = int(FGN_LENGTHS[int(srng.integers(0, len(FGN_LENGTHS)))])
        H = float(FGN_HURSTS[int(srng.integers(0, len(FGN_HURSTS)))])
        sseed = int(srng.integers(0, 2**31 - 1))
        # The fbm library uses numpy's legacy RandomState internally. Seed
        # it with a per-series seed inside a save/restore so the rest of
        # the process's RNG state is unaffected.
        prior_state = np.random.get_state()
        try:
            np.random.seed(sseed)
            gen = FBM(n=length, hurst=H, length=1.0, method="daviesharte")
            values = np.asarray(gen.fgn(), dtype=float).ravel()
        finally:
            np.random.set_state(prior_state)
        # FBM.fgn returns exactly n samples. Truncate defensively in case
        # any backend variant returns n+1.
        if values.size != length:
            values = values[:length]
        idx = pd.date_range("2020-01-01", periods=length, freq="1min")
        s = pd.Series(values, index=idx, name=f"fgn-{sid:04d}")
        s.attrs["true_H"] = H
        s.attrs["length"] = length
        s.attrs["series_id"] = sid
        s.attrs["seed"] = sseed
        s.attrs["dataset_id"] = "D3"
        corpus.append(s)
    return corpus


def make_regime_switching_corpus(
    seed: int = REGISTERED_FGN_SEED,
    n_series: int = 256,
) -> list[pd.Series]:
    """D4 — synthetic regime-switching fGn corpus (Amendment 1).

    Each series is four concatenated fGn segments of length ``D4_SEGMENT_LENGTH``,
    so total length is ``D4_SEGMENT_LENGTH * D4_N_SEGMENTS`` = 16384. The per-segment
    H is drawn from ``D4_SEGMENT_HURSTS`` (= {0.30, 0.55, 0.70, 0.85}) under the
    constraint that adjacent segments differ in H by at least
    ``D4_MIN_ADJACENT_GAP`` (= 0.15). The three internal segment-start positions
    constitute the ground-truth boundaries for M-S6.

    Determinism mirrors :func:`make_fgn_corpus`: a master ``SeedSequence(seed)``
    spawns ``n_series`` per-series sequences, each of which then spawns 4 per-segment
    sequences. The result for ``series_id = i`` is therefore invariant to
    ``n_series``: a call with ``n_series=4`` returns the same first 4 series as a
    call with ``n_series=256``.

    Per series the following ``attrs`` are populated:
      * ``dataset_id`` — ``"D4"``
      * ``series_id`` — int ``i``
      * ``seed`` — the integer seed passed to the ``fbm`` generator (informational;
        the actual segment values are seeded via the spawned ``SeedSequence``)
      * ``true_boundaries`` — ``np.array([4096, 8192, 12288])``
      * ``true_Hs`` — ``np.array([H1, H2, H3, H4])``

    Parameters
    ----------
    seed : int
        Registered seed (20260514). Changing this is a §14 deviation.
    n_series : int
        Number of series to generate. The amendment fixes the protocol value
        at 256; the v0.2.1 pilot in ``replicate.py`` uses 32 to keep wall time
        bounded.

    Returns
    -------
    list[pandas.Series]
        Length ``n_series``. Each series has a 1-minute ``DatetimeIndex`` starting
        2020-01-01 and dtype float64.

    Raises
    ------
    RuntimeError
        If the adjacent-H gap constraint cannot be satisfied within
        ``D4_MAX_RESAMPLE_ATTEMPTS`` attempts for some series.
    """
    from fbm import FBM

    master = np.random.SeedSequence(int(seed))
    series_seeds = master.spawn(int(n_series))

    total_length = D4_SEGMENT_LENGTH * D4_N_SEGMENTS
    # Internal segment starts (the ground-truth boundaries). The implicit
    # boundary at position 0 is not part of M-S6 ground truth: only the
    # three regime changes are.
    true_boundaries = np.asarray(
        [D4_SEGMENT_LENGTH * k for k in range(1, D4_N_SEGMENTS)],
        dtype=np.int64,
    )

    corpus: list[pd.Series] = []
    for sid in range(int(n_series)):
        s_seed_seq = series_seeds[sid]
        # The H schedule is drawn from a *separate* per-series RNG so that
        # the per-segment fGn generation streams (spawned next) are not
        # consumed by the schedule resampling loop. This keeps the per-segment
        # streams reproducible regardless of how many resample attempts the
        # adjacency constraint required.
        schedule_rng = np.random.default_rng(s_seed_seq)
        Hs: list[float] | None = None
        for _attempt in range(D4_MAX_RESAMPLE_ATTEMPTS):
            candidate = [
                float(D4_SEGMENT_HURSTS[int(schedule_rng.integers(0, len(D4_SEGMENT_HURSTS)))])
                for _ in range(D4_N_SEGMENTS)
            ]
            gaps = [abs(candidate[k + 1] - candidate[k]) for k in range(D4_N_SEGMENTS - 1)]
            if all(g >= D4_MIN_ADJACENT_GAP - 1e-12 for g in gaps):
                Hs = candidate
                break
        if Hs is None:
            raise RuntimeError(
                f"D4 series {sid}: could not satisfy adjacent-H gap "
                f">= {D4_MIN_ADJACENT_GAP} within {D4_MAX_RESAMPLE_ATTEMPTS} attempts."
            )

        # Per-segment fGn streams: spawn 4 deterministic SeedSequences off
        # the per-series sequence.
        segment_seeds = s_seed_seq.spawn(D4_N_SEGMENTS)
        segments: list[np.ndarray] = []
        # Record the first segment's integer seed for informational ``attrs["seed"]``.
        first_segment_int_seed: int | None = None
        for k in range(D4_N_SEGMENTS):
            seg_rng = np.random.default_rng(segment_seeds[k])
            seg_int_seed = int(seg_rng.integers(0, 2**31 - 1))
            if first_segment_int_seed is None:
                first_segment_int_seed = seg_int_seed
            prior_state = np.random.get_state()
            try:
                np.random.seed(seg_int_seed)
                gen = FBM(n=D4_SEGMENT_LENGTH, hurst=Hs[k], length=1.0, method="daviesharte")
                values = np.asarray(gen.fgn(), dtype=float).ravel()
            finally:
                np.random.set_state(prior_state)
            if values.size != D4_SEGMENT_LENGTH:
                values = values[:D4_SEGMENT_LENGTH]
            segments.append(values)

        full = np.concatenate(segments)
        idx = pd.date_range("2020-01-01", periods=total_length, freq="1min")
        s = pd.Series(full, index=idx, name=f"d4-{sid:04d}")
        s.attrs["dataset_id"] = "D4"
        s.attrs["series_id"] = sid
        s.attrs["seed"] = int(first_segment_int_seed) if first_segment_int_seed is not None else int(seed)
        s.attrs["true_boundaries"] = true_boundaries.copy()
        s.attrs["true_Hs"] = np.asarray(Hs, dtype=float)
        s.attrs["length"] = total_length
        corpus.append(s)
    return corpus
