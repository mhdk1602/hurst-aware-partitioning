# Amendment 2 to the H2 Pre-Registration

**Parent documents:**
- [`h2-prereg-v1.md`](./h2-prereg-v1.md) — archived on Zenodo at [10.5281/zenodo.20188014](https://doi.org/10.5281/zenodo.20188014).
- [`h2-prereg-v1.1-amendment-1.md`](./h2-prereg-v1.1-amendment-1.md) — archived on Zenodo at [10.5281/zenodo.20190347](https://doi.org/10.5281/zenodo.20190347).

**Amendment posted:** 2026-05-14
**Type:** Additive (new candidate trigger). Primary outcome M-PRIMARY and the §12 falsification rule of v1 are **unchanged**. Amendment 1's M-S6 secondary outcome is **unchanged**.
**Status:** Pre-registered. No D1 or D2 analyses have been performed prior to this amendment being archived.

---

## Why this amendment exists

The D3 pilot (v0.2.0) and the D4 pilot (v0.2.1 under Amendment 1) both surfaced the same operational pattern: the registered HURST-CI trigger ("rolling-window CI disjoint AND |ΔH| ≥ `min_point_gap`") almost never fires at the bootstrap iteration counts the pilots used (n_bootstrap = 30 or 50, against the registered 1000). On D4 specifically, HURST-CI's boundary F1 was **0.013**, against an H2-S4 disconfirmation threshold of 0.5.

An exploratory study (script: [`experiments/trigger-exploration-2026-05.py`](../experiments/trigger-exploration-2026-05.py); results: [`experiments/trigger-exploration-2026-05.json`](../experiments/trigger-exploration-2026-05.json); n_series=8, window=500, step=100, n_bootstrap=30, tolerance=125) compares the registered trigger against a less-conservative alternative that drops the CI-disjoint requirement and keeps only the point-gap criterion. Mean F1 against the D4 ground-truth boundaries:

| θ (point-gap threshold) | Registered CI+gap trigger F1 | Point-gap-only F1 | Pred. boundaries per series (pg-only) |
|---|---|---|---|
| 0.05 | 0.050 ± 0.050 | 0.117 ± 0.012 | 27.50 |
| **0.10** (registered default) | **0.050 ± 0.050** | **0.373 ± 0.048** | **4.62** |
| 0.15 | 0.050 ± 0.050 | 0.367 ± 0.026 | 3.12 |
| 0.20 | 0.050 ± 0.050 | 0.323 ± 0.052 | 2.88 |

The point-gap-only criterion at θ=0.10 returns mean F1 ≈ 0.37 with about 4.6 predicted boundaries per series (against 3 ground-truth boundaries). The registered CI+gap criterion at the same θ returns F1 ≈ 0.05 with about 1.4 predicted boundaries per series.

Per the §14 deviations protocol, the exploratory study is labeled exploratory and does not change v1. **This amendment is the formal, pre-registered consequence of that finding.**

Two interpretations of the finding remain open:

1. **The CI test is the bootstrap bottleneck.** At n_bootstrap=30 the per-window CI is wide; the registered n_bootstrap=1000 will narrow it by roughly √(1000/30) ≈ 5.8×, possibly enough to make CI-disjoint a meaningful trigger.
2. **The CI test is fundamentally too conservative.** Even at n_bootstrap=1000, the structural requirement that two adjacent windows' bootstrap CIs be fully disjoint may never fire on smooth-but-real regime transitions.

Amendment 2 does not resolve this question. It registers a second candidate trigger so the comparison is empirical rather than rhetorical.

## What this amendment adds

### New candidate trigger: GAP-ONLY

A boundary fires at position `t` iff `|H(t) - H(t-step)| ≥ min_point_gap`. Default `min_point_gap = 0.10` (matches the registered HURST-CI default). The CI-disjoint requirement is dropped.

GAP-ONLY is strictly less conservative than HURST-CI. On constant-H data (D3) it will produce more false positives than HURST-CI; on regime-switching data (D4) it should produce more true positives. The two triggers are reported side-by-side on every dataset.

### New hypothesis H2-S5 (trigger insensitivity on D1)

For each baseline B and each estimator E, the GAP-ONLY trigger's M-PRIMARY shift on D1 is within ±20% (relative) of the HURST-CI trigger's M-PRIMARY shift, where shifts are normalized to FIXED-DAILY. Predicted: if both triggers yield the same paper, the choice was uninformative; if one wins dramatically, the choice matters and we report the winner with explicit attention.

H2-S5 is disconfirmed if **either** trigger fails to reproduce the H2-P direction of effect on D1 against B1=FIXED-DAILY (i.e., produces a negative HL shift or one that crosses zero in CI). A disconfirmation of H2-S5 does **not** falsify H2-P — it just means the trigger choice matters and we report both numbers honestly.

### New hypothesis H2-S6 (boundary quality across triggers on D4)

On D4, with the registered `n_bootstrap = 1000`, the GAP-ONLY trigger achieves mean M-S6 ≥ 0.30 **and** the GAP-ONLY mean M-S6 is at least 2× the HURST-CI mean M-S6. Predicted: a clear boundary-quality advantage for the less-conservative trigger.

H2-S6 is disconfirmed if **either** GAP-ONLY F1 falls below 0.30 **or** the ratio (GAP-ONLY F1) / (HURST-CI F1) falls below 2. A null result here is exploratory with respect to v1 §12 falsification but it tells us the conservatism of the CI gate is not the binding constraint.

### Multiple-testing family extension

Amendment 1 grew the family from 48 to 52 (D4 × 4 baselines × DFA). Amendment 2 adds the GAP-ONLY trigger as a fifth comparison on each of the existing rows, plus its own D4 × 4-baseline block:

- D1 × B1..B4 × DFA × {HURST-CI, GAP-ONLY} = 8 new comparisons (the existing 4 plus 4 GAP-ONLY-paired versions of them; only the 4 GAP-ONLY ones are new because v1 already covers HURST-CI).
- D2 × B1..B4 × DFA × GAP-ONLY = 4 new.
- D3-persistent × B1..B4 × DFA × GAP-ONLY = 4 new.
- D4 × B1..B4 × DFA × GAP-ONLY = 4 new (M-S6 column).

Total new comparisons: **16**. New family size: **68** (52 from Amendment 1 + 16 from this amendment). Holm-Bonferroni is re-applied at family-wise α = 0.05 across all 68 comparisons. Pre-registered ranking: GAP-ONLY rows insert immediately after the corresponding HURST-CI row at every dataset×baseline pair.

### Implementation

- `src/hurst_partitioning/partitioner.py` will add a `HurstGapOnlyPartitioner` class with the same interface as `HurstCIPartitioner` but the simplified trigger (point-gap only, no CI computation). Landing at v0.2.2.
- `replicate.py` will run both triggers in the same pass; output keys will include the `trigger` field. The same per-series partitioner object accepts an `estimator` argument; no change to the registered estimator battery.
- The GAP-ONLY partitioner has the same `min_point_gap = 0.10` default and the same window/step grid; only the CI computation step is removed (so it is also strictly cheaper to compute than HURST-CI).

## What this amendment does NOT change

- **Primary outcome (§8 v1):** M-PRIMARY remains primary. Both triggers are tested against it.
- **Falsification rule (§12 v1):** unchanged. H2-P is rejected only if all three v1 §12 conditions hold for **at least one** of the two triggers — that is, if both triggers fail to meet the H2-P bar, H2-P falls. The amendment makes the test more conservative (harder to falsify), not less.
- **Datasets D1, D2, D3, D4 (§4 v1 + Amendment 1):** unchanged.
- **Estimator battery (§5 v1):** unchanged.
- **Baselines B1–B5 (§6 v1):** unchanged.
- **Workload (§7 v1):** unchanged.
- **Stopping rules (§13 v1):** unchanged. All 68 comparisons are computed in a single batch in one run of `replicate.py` before any results are inspected.
- **Software environment (§17 v1):** unchanged.
- **HURST-CI as a registered candidate:** unchanged. It remains the v1 candidate. GAP-ONLY is an *additional* candidate.

## Pre-commitment

Amendment 2 is being archived **before** the v0.2.2 implementation of GAP-ONLY lands and **before** any D1, D2, or D4-at-registered-n_bootstrap analyses are performed. The first runs of GAP-ONLY are part of the v0.3+ empirical run.

## What "pre-registered" means here

The exploratory trigger study reported above is what gives us the *reason* to add GAP-ONLY. It does not constitute evidence *for* H2-S6. The expected GAP-ONLY F1 of 0.30 on D4 at registered settings is a prediction; the registered comparison will report whether it survives.

## Authorship and conflicts

Sole author: Dineshkumar Malempati Hari ([ORCID 0009-0003-1036-9477](https://orcid.org/0009-0003-1036-9477)). No conflicts.

---

*This amendment is additive to v1 and Amendment 1. The cumulative pre-registration is: v1 (DOI 10.5281/zenodo.20188014) ∪ Amendment 1 (DOI 10.5281/zenodo.20190347) ∪ Amendment 2 (this file). The concept DOI 10.5281/zenodo.20188013 always resolves to the latest version.*
