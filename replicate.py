"""Master replication script.

v0.2.0 — D3 (synthetic fGn) pilot ONLY. Per prereg §13, D1 (S&P) and D2
(NAB) cannot be analyzed until the pre-registration is locked final at
G1 (2026-07). The pilot generates the registered fGn corpus, fits the
CANDIDATE (HURST-CI) and the four implemented baselines, runs the 100-
query workload against each partition, and computes paired Wilcoxon with
Holm-Bonferroni for HURST-CI vs each baseline.

The output is written to ``experiments/d3-pilot-results.json``. Pilot
hyperparameters (rolling-window count, bootstrap replicates, corpus
subsample) are recorded in the JSON's ``pilot_settings`` block so the
full v0.3+ run can be compared against the pilot for sanity.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# When the package is not installed (e.g., on a dev machine where the
# system Python predates the requires-python pin), fall back to importing
# directly from ``src/``. This mirrors the conftest.py used by pytest.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hurst_partitioning import __version__  # noqa: E402
from hurst_partitioning.baselines import (  # noqa: E402
    EqualRowsPartitioner,
    FixedDailyPartitioner,
    FixedMonthlyPartitioner,
    VarianceCusumPartitioner,
)
from hurst_partitioning.benchmark import (  # noqa: E402
    archive_workload,
    generate_workload,
    measure,
    paired_wilcoxon_with_holm,
)
from hurst_partitioning.io import make_fgn_corpus  # noqa: E402
from hurst_partitioning.partitioner import HurstCIPartitioner  # noqa: E402


REPO_ROOT = Path(__file__).parent
PROTOCOL_PATH = REPO_ROOT / "prereg" / "h2-prereg-v1.md"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_PATH = EXPERIMENTS_DIR / "d3-pilot-results.json"
WORKLOAD_PATH = EXPERIMENTS_DIR / "query-library-v1.json"


def _pilot_settings() -> dict:
    """Single source of truth for pilot-only deviations from the full grid.

    The registered protocol calls for n_bootstrap=1000 and step=20 in the
    rolling Hurst-CI estimator. On the synthetic D3 pilot run we relax
    these to keep wall time bounded; the relaxations are recorded here
    and surfaced in the output JSON. The full v0.3+ run on D1/D2 will
    use the registered values.
    """
    return {
        "n_corpus": 1024,
        "persistent_subsample_per_H": 8,  # 8 H=0.7 + 8 H=0.85 series
        "pilot_series_length": 4096,
        "hurst_estimator": "dfa",
        "hurst_window": 500,
        "hurst_step": 200,
        "hurst_min_point_gap": 0.1,
        "hurst_ci_level": 0.95,
        "hurst_n_bootstrap": 50,
        "wilcoxon_alpha": 0.05,
        "hl_bootstrap_replicates": 1000,
        "row_byte_size": 16,
        "workload_seed": 20260514,
        "rng_seed": 20260514,
        "note": (
            "Pilot deviations vs prereg §5: rolling n_bootstrap reduced from 1000 to 50, "
            "step from 20 to 200, series subsampled to length=4096 only. These "
            "relax-for-speed choices are pilot-only; the v0.3 full run on D1/D2 "
            "uses the registered values."
        ),
    }


def _pick_persistent_subsample(corpus, settings) -> list:
    """Pick a balanced subsample of persistent (H>=0.6) D3 series."""
    by_H: dict[float, list] = {}
    for s in corpus:
        H = s.attrs["true_H"]
        if H < 0.6:
            continue
        if s.attrs["length"] != settings["pilot_series_length"]:
            continue
        by_H.setdefault(H, []).append(s)
    out = []
    for H, lst in sorted(by_H.items()):
        # Deterministic order: rely on series_id sort.
        lst_sorted = sorted(lst, key=lambda s: s.attrs["series_id"])
        out.extend(lst_sorted[: settings["persistent_subsample_per_H"]])
    return out


def _run_pilot() -> dict:
    settings = _pilot_settings()
    t_start = time.time()

    # --- 1. Generate the registered D3 corpus.
    print(f"hurst-aware-partitioning v{__version__} — D3 pilot")
    print(f"Generating fGn corpus (n={settings['n_corpus']}, seed={settings['rng_seed']})...")
    corpus = make_fgn_corpus(seed=settings["rng_seed"], n_series=settings["n_corpus"])
    persistent = _pick_persistent_subsample(corpus, settings)
    print(f"  persistent subsample: {len(persistent)} series")
    for s in persistent:
        print(
            f"    id={s.attrs['series_id']:4d}  H={s.attrs['true_H']}  len={s.attrs['length']}"
        )

    # --- 2. Generate the workload once (it is the same across series; the
    # registered protocol fixes indices uniformly within series_len). For
    # the pilot all subsampled series share the same length.
    series_len = settings["pilot_series_length"]
    print(f"Generating 100-query workload at series_len={series_len}...")
    queries = generate_workload(series_len, seed=settings["workload_seed"])
    digest = archive_workload(queries, WORKLOAD_PATH)
    print(f"  workload archived to {WORKLOAD_PATH.name} (sha256={digest[:16]}...)")

    # --- 3. Fit each partitioner on each series and run the workload.
    partitioner_factories: list[tuple[str, callable]] = [
        ("hurst-ci",      lambda: HurstCIPartitioner(
            estimator=settings["hurst_estimator"],
            window=settings["hurst_window"],
            step=settings["hurst_step"],
            min_point_gap=settings["hurst_min_point_gap"],
            ci_level=settings["hurst_ci_level"],
            n_bootstrap=settings["hurst_n_bootstrap"],
            rng_seed=settings["rng_seed"],
        )),
        ("fixed-daily",   FixedDailyPartitioner),
        ("fixed-monthly", FixedMonthlyPartitioner),
        ("variance-cusum", lambda: VarianceCusumPartitioner(h=5.0, k=0.5)),
        ("equal-rows",    lambda: EqualRowsPartitioner(chunk_size=10_000)),
    ]

    # bytes_read[policy][series_index] = list of per-query bytes
    bytes_read: dict[str, list[list[int]]] = {p: [] for p, _ in partitioner_factories}
    chunks_touched: dict[str, list[list[int]]] = {p: [] for p, _ in partitioner_factories}
    n_boundaries: dict[str, list[int]] = {p: [] for p, _ in partitioner_factories}
    timings: dict[str, float] = {p: 0.0 for p, _ in partitioner_factories}

    for s_idx, s in enumerate(persistent):
        print(f"[{s_idx + 1}/{len(persistent)}] series_id={s.attrs['series_id']} H={s.attrs['true_H']} ...")
        for policy_name, factory in partitioner_factories:
            partitioner = factory()
            t0 = time.time()
            partition = partitioner.fit(s)
            elapsed = time.time() - t0
            timings[policy_name] += elapsed
            n_boundaries[policy_name].append(int(partition.boundaries.size))
            per_query_bytes = []
            per_query_chunks = []
            for q in queries:
                res = measure(partition, q, row_byte_size=settings["row_byte_size"])
                per_query_bytes.append(int(res.bytes_read))
                per_query_chunks.append(int(res.chunks_touched))
            bytes_read[policy_name].append(per_query_bytes)
            chunks_touched[policy_name].append(per_query_chunks)
            print(
                f"    {policy_name:<16} fit={elapsed:6.1f}s  "
                f"n_chunks={partition.boundaries.size:>5}  "
                f"mean_bytes/query={np.mean(per_query_bytes):.0f}"
            )

    # --- 4. Paired Wilcoxon with Holm-Bonferroni for HURST-CI vs each baseline.
    # Pool per-query observations across all persistent series (concatenate)
    # so each comparison has 100 * |subsample| paired observations.
    baselines = ["fixed-daily", "fixed-monthly", "variance-cusum", "equal-rows"]
    paired: dict[str, np.ndarray] = {}
    for base in baselines:
        cand = np.array(bytes_read["hurst-ci"]).ravel()  # shape (n_series * 100,)
        bsl = np.array(bytes_read[base]).ravel()
        paired[f"D3-persistent × {base} × dfa"] = np.column_stack([cand, bsl])

    ranking = list(paired.keys())
    print("\nRunning paired Wilcoxon with Holm-Bonferroni...")
    table = paired_wilcoxon_with_holm(paired, ranking=ranking, alpha=settings["wilcoxon_alpha"])
    print(table.to_string(index=False))

    # --- 5. Summary stats and save.
    summary = {
        "version": __version__,
        "release": "v0.2.0",
        "dataset": "D3 (synthetic fGn pilot)",
        "n_persistent_series": len(persistent),
        "n_comparisons": int(len(table)),
        "n_rejections_after_holm": int(table["reject"].sum()),
        "median_hl_shift_across_comparisons": float(
            np.median(table["hl_shift"].values)
        ),
        "fit_seconds_total": {p: round(t, 2) for p, t in timings.items()},
        "wall_seconds_total": round(time.time() - t_start, 2),
    }
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    result = {
        "summary": summary,
        "pilot_settings": settings,
        "workload_sha256": digest,
        "subsample_series": [
            {
                "series_id": s.attrs["series_id"],
                "true_H": s.attrs["true_H"],
                "length": s.attrs["length"],
                "seed": s.attrs["seed"],
            }
            for s in persistent
        ],
        "n_boundaries_per_policy": n_boundaries,
        "wilcoxon_holm_table": table.to_dict(orient="records"),
        "per_policy_total_bytes": {
            p: int(np.sum(bytes_read[p])) for p in bytes_read
        },
        "per_policy_mean_bytes_per_query": {
            p: float(np.mean(bytes_read[p])) for p in bytes_read
        },
    }
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults written to {RESULTS_PATH}")
    return result


def main() -> int:
    print(f"Reading pre-registration: {PROTOCOL_PATH}")
    if not PROTOCOL_PATH.exists():
        print("ERROR: pre-registration document not found.", file=sys.stderr)
        return 1
    print("Per prereg §13, D1 and D2 are locked until 2026-07. Running D3 pilot only.")
    _run_pilot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
