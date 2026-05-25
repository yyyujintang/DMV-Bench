"""TextOnly baseline: memory stores only the encoded text (leakage-controlled).

At oracle injection, returns the trial's encode_text (no image).
At full pipeline, encodes text + recall query → retrieves nearest by token overlap.
"""

from __future__ import annotations

import re
from typing import List, Optional

from dualmem.types import Trial


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z]+", s.lower()))


class TextOnly:
    name = "TextOnly"

    def __init__(self):
        self._mem: List[Trial] = []

    def reset(self):
        self._mem = []

    def encode(self, trial: Trial):
        self._mem.append(trial)

    def retrieve(self, query_text: str) -> Optional[Trial]:
        if not self._mem:
            return None
        q = _tokens(query_text)
        scored = [(len(q & _tokens(m.encode_text)), m) for m in self._mem]
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def oracle_inject(self, trial: Trial) -> dict:
        return {"text": f"Memory text (verbal channel only):\n\"{trial.encode_text}\"", "images": []}
