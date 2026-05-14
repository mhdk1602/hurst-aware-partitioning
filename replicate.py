"""Master replication script. Executes the registered protocol end-to-end.

Stages
------
1. Acquire D1 (yfinance), D2 (NAB submodule), D3 (synthetic fGn).
2. For each series, compute the rolling-Hurst battery with block-bootstrap CIs.
3. For each policy (CANDIDATE + 4 baselines + optional ORACLE), compute the
   partition.
4. Run the 100-query workload against each partition under chunk-level I/O
   accounting (the primary outcome metric M-PRIMARY).
5. Paired Wilcoxon with Holm-Bonferroni correction over the 48-comparison
   family.
6. Apply the §13 falsification rule and write the summary to
   research/paper/tables/.

This script is the artifact reviewers should be able to run unattended to
reproduce every number in the eventual paper.

The v0.1.0-prereg release prints the registered protocol and exits without
executing it, to enforce the "no analysis before pre-registration lock"
contract of the registration.
"""
from __future__ import annotations

import sys
from pathlib import Path

from hurst_partitioning import __version__


PROTOCOL_PATH = Path(__file__).parent / "prereg" / "h2-prereg-v1.md"


def main() -> int:
    print(f"hurst-aware-partitioning {__version__}")
    print(f"Reading pre-registration: {PROTOCOL_PATH}")
    if not PROTOCOL_PATH.exists():
        print("ERROR: pre-registration document not found.", file=sys.stderr)
        return 1
    print()
    print("This release (v0.1.0-prereg) archives the registered protocol and")
    print("reference scaffolding only. The execution path will be wired at")
    print("v0.2.0 once the implementation is complete on D3 (synthetic).")
    print()
    print("To inspect the registered protocol, read prereg/h2-prereg-v1.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
