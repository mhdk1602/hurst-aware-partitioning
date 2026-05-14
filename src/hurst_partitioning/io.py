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

    First call downloads via yfinance and caches as parquet. Subsequent calls
    read from cache. Returns a dict[ticker -> DataFrame].
    """
    raise NotImplementedError(
        "load_d1_sp500 unimplemented in v0.1.0-prereg. The ticker list is "
        "registered at src/hurst_partitioning/data/sp500_50_tickers.txt "
        "(populated alongside the implementation at v0.2.0)."
    )


def load_d2_nab(nab_repo_dir: Path) -> dict[str, pd.DataFrame]:
    """D2 — 10 Numenta Anomaly Benchmark series.

    Reads from a NAB submodule pinned to a registered commit. The pinned
    commit hash lives at experiments/d2.commit and must NOT be advanced
    between the pre-registration and the empirical run.
    """
    raise NotImplementedError("load_d2_nab unimplemented in v0.1.0-prereg.")


def make_fgn_corpus(
    seed: int = REGISTERED_FGN_SEED,
    n_series: int = 1024,
) -> list[pd.Series]:
    """D3 — synthetic fractional Gaussian noise corpus.

    Lengths sampled from {4096, 16384, 65536}; H sampled from
    {0.3, 0.5, 0.7, 0.85}. The (length, H) per series is deterministic
    from ``seed``. The seed is registered and immutable.
    """
    raise NotImplementedError("make_fgn_corpus unimplemented in v0.1.0-prereg.")
