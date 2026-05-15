"""Exploratory study: HURST-CI trigger sensitivity on D4 (post-prereg, exploratory only).

This script is NOT part of the registered protocol. It exists to gather evidence
for whether the registered HURST-CI trigger (CI-disjoint AND |ΔH| >= min_point_gap)
should be amended with a less conservative alternative.

What it computes
----------------
On a small subsample of D4 (regime-switching fGn corpus), roll DFA(2) at
window=500, step=100, n_bootstrap=30 across each series, then for each
threshold θ in {0.05, 0.10, 0.15, 0.20} compute two trigger F1 scores
against the true regime boundaries (positions 4096, 8192, 12288):

  - "point-gap-only": fire when |H(t) - H(t-step)| >= θ
  - "ci-plus-gap" (registered shape): fire when CI(t) ∩ CI(t-step) = ∅ AND |ΔH| >= θ

Tolerance for the boundary F1 match is the same as the amendment registered value:
max(window // 4, 50) = 125.

Interpretation
--------------
If point-gap-only delivers materially higher F1 than ci-plus-gap at any θ in the
sweep, the empirical evidence supports adding a less-conservative trigger as a
registered alternative in Amendment 2. If both stay near zero across the sweep,
the issue lies elsewhere (estimator variance at W=500, or the regime-switching
amplitudes themselves), and the right fix is a longer window or a different
estimator class.

Output
------
experiments/trigger-exploration-2026-05.json with the per-(series, θ, trigger)
F1 numbers and aggregate means.

Per prereg §14 (deviations protocol), this analysis is labeled exploratory.
Any decision derived from it that changes the registered protocol must be
published as Amendment 2 BEFORE D1 or D2 is analyzed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hurst_partitioning.benchmark import boundary_f1  # noqa: E402
from hurst_partitioning.estimators import _dfa2, block_bootstrap_ci  # noqa: E402
from hurst_partitioning.io import make_regime_switching_corpus  # noqa: E402


WINDOW = 500
STEP = 100
N_BOOTSTRAP = 30
TOLERANCE = max(WINDOW // 4, 50)
N_SERIES = 8
THETAS = (0.05, 0.10, 0.15, 0.20)
OUTPUT_PATH = Path(__file__).resolve().parent / "trigger-exploration-2026-05.json"


def _roll(values: np.ndarray, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    n = values.size
    positions: list[int] = []
    points: list[float] = []
    los: list[float] = []
    his: list[float] = []
    t = WINDOW
    while t <= n:
        win = values[t - WINDOW : t]
        try:
            p = float(_dfa2(win))
            lo, hi = block_bootstrap_ci(
                win, _dfa2, n_bootstrap=N_BOOTSTRAP, level=0.95, rng=rng
            )
        except Exception:
            t += STEP
            continue
        positions.append(t)
        points.append(p)
        los.append(lo)
        his.append(hi)
        t += STEP
    return (
        np.asarray(positions, dtype=int),
        np.asarray(points, dtype=float),
        np.asarray(los, dtype=float),
        np.asarray(his, dtype=float),
    )


def _apply_triggers(positions, points, los, his, theta):
    """Return (bounds_point_gap_only, bounds_ci_plus_gap) — each a 1-D np.ndarray."""
    pg: list[int] = []
    cg: list[int] = []
    for i in range(1, len(positions)):
        d = abs(points[i] - points[i - 1])
        if d >= theta:
            pg.append(int(positions[i]))
            ci_disjoint = (los[i] > his[i - 1]) or (his[i] < los[i - 1])
            if ci_disjoint:
                cg.append(int(positions[i]))
    return np.asarray(pg, dtype=int), np.asarray(cg, dtype=int)


def main() -> None:
    t0 = time.time()
    print(f"Exploration: window={WINDOW} step={STEP} n_bootstrap={N_BOOTSTRAP} tol={TOLERANCE}")
    print(f"D4 subsample: n_series={N_SERIES}")
    corpus = make_regime_switching_corpus(seed=20260514, n_series=N_SERIES)

    per_row = []
    for s in corpus:
        sid = int(s.attrs["series_id"])
        true_b = np.asarray(s.attrs["true_boundaries"], dtype=int)
        true_hs = np.asarray(s.attrs["true_Hs"], dtype=float).tolist()
        print(f"  series {sid} Hs={true_hs} ... rolling ...", flush=True)
        positions, points, los, his = _roll(s.values.astype(float).ravel(), rng_seed=sid)
        for theta in THETAS:
            pg_bounds, cg_bounds = _apply_triggers(positions, points, los, his, theta)
            f1_pg = boundary_f1(pg_bounds, true_b, tolerance=TOLERANCE)
            f1_cg = boundary_f1(cg_bounds, true_b, tolerance=TOLERANCE)
            per_row.append(
                {
                    "series_id": sid,
                    "true_Hs": true_hs,
                    "theta": float(theta),
                    "n_windows": int(positions.size),
                    "n_pred_point_gap_only": int(pg_bounds.size),
                    "n_pred_ci_plus_gap": int(cg_bounds.size),
                    "f1_point_gap_only": float(f1_pg),
                    "f1_ci_plus_gap": float(f1_cg),
                }
            )

    # Aggregate by theta
    by_theta = {}
    for theta in THETAS:
        rows = [r for r in per_row if r["theta"] == theta]
        n = len(rows)

        def _mean(key):
            return float(np.mean([r[key] for r in rows])) if rows else float("nan")

        def _se(key):
            if not rows:
                return float("nan")
            v = np.asarray([r[key] for r in rows], dtype=float)
            return float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

        by_theta[f"{theta}"] = {
            "n_series": n,
            "mean_f1_point_gap_only": _mean("f1_point_gap_only"),
            "se_f1_point_gap_only": _se("f1_point_gap_only"),
            "mean_f1_ci_plus_gap": _mean("f1_ci_plus_gap"),
            "se_f1_ci_plus_gap": _se("f1_ci_plus_gap"),
            "mean_n_pred_point_gap_only": _mean("n_pred_point_gap_only"),
            "mean_n_pred_ci_plus_gap": _mean("n_pred_ci_plus_gap"),
        }

    output = {
        "settings": {
            "window": WINDOW,
            "step": STEP,
            "n_bootstrap": N_BOOTSTRAP,
            "tolerance": TOLERANCE,
            "n_series": N_SERIES,
            "thetas": list(THETAS),
            "estimator": "dfa2",
            "exploratory_label": True,
            "registered_alternative": False,
        },
        "by_theta": by_theta,
        "per_row": per_row,
        "elapsed_s": time.time() - t0,
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print()
    print(f"Wrote {OUTPUT_PATH}")
    print()
    print("Aggregate (per θ):")
    print(f"{'theta':>6} {'F1 pg-only':>12} {'F1 ci+gap':>12} {'n pg':>6} {'n ci':>6}")
    for theta in THETAS:
        b = by_theta[f"{theta}"]
        print(
            f"{theta:>6} {b['mean_f1_point_gap_only']:>12.3f} "
            f"{b['mean_f1_ci_plus_gap']:>12.3f} "
            f"{b['mean_n_pred_point_gap_only']:>6.2f} "
            f"{b['mean_n_pred_ci_plus_gap']:>6.2f}"
        )


if __name__ == "__main__":
    main()
