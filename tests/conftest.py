"""Make the example fixtures importable from the test suite.

``tests/`` and ``examples/`` are siblings, so without this the CLI tests
(which use the same offline fixtures the examples use) would have to duplicate
them -- and a duplicated fixture is a fixture that silently drifts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))
