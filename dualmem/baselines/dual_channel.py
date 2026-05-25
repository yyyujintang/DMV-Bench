"""DualChannel baseline: stores both visual (raw image) and verbal (caption) channels.

Maps to CoMEM-analog in Proposal_A. At injection time, BOTH the raw image
AND the structured caption are provided to the VLM, simulating Paivio's
dual-coding theory.
"""

from __future__ import annotations

import re
from typing import List, Optional

from dualmem.types import Trial
from dualmem.baselines.caption import _caption_for


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class DualChannel:
    name = "DualChannel"

    def __init__(self):
        self._mem: List[Trial] = []
        self._captions: dict = {}

    def reset(self):
        self._mem = []
        self._captions = {}

    def encode(self, trial: Trial):
        self._mem.append(trial)
        self._captions[trial.anchor_slug] = _caption_for(trial)

    def retrieve(self, query_text: str) -> Optional[Trial]:
        if not self._mem:
            return None
        q = _tokens(query_text)
        scored = [(len(q & _tokens(self._captions[m.anchor_slug])), m) for m in self._mem]
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def oracle_inject(self, trial: Trial) -> dict:
        cap = self._captions.get(trial.anchor_slug) or _caption_for(trial)
        return {
            "text": f"Memory (dual-channel, image + caption):\nCaption: \"{cap}\"",
            "images": [trial.anchor_path],
        }
