"""
gympn_path.py
-------------
Make `gympn` importable.

gympn (https://github.com/bpogroup/gympn) is the Petri-net simulation layer the
four experiment drivers build their processes on.  It is not published on PyPI,
so it is installed either straight from git

    pip install git+https://github.com/bpogroup/gympn.git

-- in which case this module does nothing -- or by cloning the repository, in
which case point GYMPN_PATH at the clone:

    Windows (PowerShell)   $env:GYMPN_PATH = "C:\\path\\to\\gympn"
    Linux / macOS          export GYMPN_PATH=/path/to/gympn

Import this module before importing anything from `gympn`.  It never raises: if
gympn cannot be found the failure is left to the real import that follows, so
the traceback points at the line that actually needed it, with the note below
attached.
"""

from __future__ import annotations

import os
import sys
from importlib.util import find_spec

HINT = (
    "gympn is not importable.  Install it with\n"
    "    pip install git+https://github.com/bpogroup/gympn.git\n"
    "or clone it and set GYMPN_PATH to the clone directory."
)


def ensure_on_path() -> None:
    """Put a GYMPN_PATH clone on sys.path when gympn is not already installed."""
    if find_spec("gympn") is not None:
        return

    root = os.environ.get("GYMPN_PATH")
    if root and os.path.isdir(root) and root not in sys.path:
        sys.path.append(root)

    if find_spec("gympn") is None:
        # Not fatal here -- the caller's own `from gympn... import ...` raises
        # next, and this makes sure the reason is on screen when it does.
        print(HINT, file=sys.stderr)


ensure_on_path()
