"""Agent task schemas + generators.

An AgentTask describes ONE episode the Playwright agent has to complete:
navigate to an anchor URL, browse filler pages, then on a recall instruction
return to the right product (or add it to a wishlist) — success scored by
URL match against the ground-truth pattern.
"""

from dualmem.tasks.spec import AgentTask
from dualmem.tasks.generator import (
    generate_same_instance_tasks,
    generate_cross_category_tasks,
    generate_long_horizon_tasks,
    generate_tasks,
    GENERATORS,
)
from dualmem.tasks.multisession import (
    MultiSessionTask,
    SessionRef,
    SessionResult,
    MultiSessionResult,
    load_multi_session,
    save_multi_session,
    score_session,
    score_cumulative,
)

__all__ = [
    "AgentTask",
    "generate_same_instance_tasks",
    "generate_cross_category_tasks",
    "generate_long_horizon_tasks",
    "generate_tasks",
    "GENERATORS",
    "MultiSessionTask",
    "SessionRef",
    "SessionResult",
    "MultiSessionResult",
    "load_multi_session",
    "save_multi_session",
    "score_session",
    "score_cumulative",
]
