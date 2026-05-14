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
