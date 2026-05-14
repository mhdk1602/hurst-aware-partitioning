"""Top-level conftest. Ensures ``src/`` is on sys.path for tests.

The package is normally installed via ``pip install -e .``. This conftest
lets pytest run without an editable install on systems where ``pip install -e .``
is awkward (e.g., when the local Python predates the ``requires-python`` pin).
The behavior of the registered analysis is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
