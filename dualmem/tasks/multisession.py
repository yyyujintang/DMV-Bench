"""Multi-session task schema — re-exported from the project-root canonical
location at `tasks/schema/multisession.py`.

The canonical module lives outside `VisMem-Diag/` because Python module names
cannot contain dashes; the project-root `tasks/` package is importable from
both the website-build pipeline (which generates the JSONs) and from
dualmem-diag's evaluation code. This shim imports it and re-exports the
public API under the `dualmem.tasks` namespace for callers that already use
that path.

Add the project root to sys.path on import so this works even when called
from inside `VisMem-Diag/` as the working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]   # …/DualMem_A
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tasks.schema.multisession import (  # noqa: E402
    MultiSessionTask,
    SessionRef,
    SessionResult,
    MultiSessionResult,
    load_multi_session,
    save_multi_session,
    score_session,
    score_cumulative,
    DEFAULT_POOL,
    DEFAULT_MULTI_POOL,
)

__all__ = [
    "MultiSessionTask",
    "SessionRef",
    "SessionResult",
    "MultiSessionResult",
    "load_multi_session",
    "save_multi_session",
    "score_session",
    "score_cumulative",
    "DEFAULT_POOL",
    "DEFAULT_MULTI_POOL",
]
