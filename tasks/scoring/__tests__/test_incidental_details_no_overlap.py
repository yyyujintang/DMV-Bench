"""Phase 3 acceptance test: STYLE_DETAILS vocabulary has zero cross-style
overlap. Per user-locked override (Step 4 override 2).
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest


def _load_tag_module():
    """Load `tools/tag_incidental_details.py` as a module without installing
    the `tools` package (it has no __init__.py)."""
    here = pathlib.Path(__file__).resolve()
    target = here.parents[3] / "tools" / "tag_incidental_details.py"
    spec = importlib.util.spec_from_file_location("tag_incidental_details", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_style_details_no_cross_style_overlap():
    mod = _load_tag_module()
    seen: dict[str, str] = {}
    for style, items in mod.STYLE_DETAILS.items():
        for item in items:
            assert item not in seen, (
                f"detail {item!r} appears in both {seen[item]!r} and {style!r} — "
                "STYLE_DETAILS must be strictly style-discriminative (v2 §5.3)"
            )
            seen[item] = style
    # Sanity: every style still has ≥ 2 details (so tagging can pick 1–2).
    for style, items in mod.STYLE_DETAILS.items():
        assert len(items) >= 2, f"style {style!r} has < 2 details"


def test_assert_no_overlap_helper_runs():
    """`tools.tag_incidental_details.assert_no_overlap()` must succeed."""
    mod = _load_tag_module()
    mod.assert_no_overlap()  # raises SystemExit on overlap; here = ok
