"""Action vocabulary + ReAct parser.

The VLM is asked to emit ReAct-style "Thought: ... / Action: ..." text.
We parse the Action line into a small set of structured action types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Action:
    type: str                # "navigate" | "click_index" | "click_product" | "add_to_wishlist" | "done" | "noop"
    url: Optional[str] = None
    url_hash: Optional[str] = None
    index: Optional[int] = None   # for click_index
    raw: str = ""             # original text for logging


_ACTION_RE = re.compile(r"action\s*:\s*(.+?)$", re.I | re.M)
_NAV_RE = re.compile(r"navigate\s*\(?\s*[\"']?(/[^\s\"'\)]+)", re.I)
_CLICK_HASH_RE = re.compile(r"click(?:_product)?\s*\(?\s*[\"']?(?:/product/)?([0-9a-f]{6,16})\b", re.I)
_CLICK_INDEX_RE = re.compile(r"click_index\s*\(?\s*\[?\s*([0-9])\s*\]?\s*\)?", re.I)
_WISH_RE = re.compile(r"(add_to_wishlist|add\s+to\s+wishlist|wishlist\s*\()", re.I)
_DONE_RE = re.compile(r"\bdone\b", re.I)


def parse_react_response(text: str) -> Action:
    """Find the LAST "Action:" line in the VLM response and parse it."""
    if not text:
        return Action(type="noop", raw="")
    # Last Action line (allowing multiple Thought/Action chains).
    matches = list(_ACTION_RE.finditer(text))
    if not matches:
        # Look for any action keyword in the whole text as fallback.
        action_text = text
    else:
        action_text = matches[-1].group(1).strip()

    # Order matters: detect more specific first.
    m = _WISH_RE.search(action_text)
    if m:
        h = _CLICK_HASH_RE.search(action_text)
        return Action(type="add_to_wishlist",
                      url_hash=h.group(1) if h else None,
                      raw=action_text)
    m = _CLICK_INDEX_RE.search(action_text)
    if m:
        return Action(type="click_index", index=int(m.group(1)), raw=action_text)
    m = _NAV_RE.search(action_text)
    if m:
        return Action(type="navigate", url=m.group(1), raw=action_text)
    m = _CLICK_HASH_RE.search(action_text)
    if m:
        return Action(type="click_product", url_hash=m.group(1),
                      url=f"/product/{m.group(1)}", raw=action_text)
    if _DONE_RE.search(action_text):
        return Action(type="done", raw=action_text)
    return Action(type="noop", raw=action_text)
