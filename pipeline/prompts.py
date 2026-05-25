"""Prompt loader for `prompts/v{N}/*.yaml`.

Versioning policy (set by user 2026-05-15): the active prompt set
lives in `prompts/v{N}/`. When we revise prompts, we COPY the
current version to `prompts/v{N+1}/` and edit only the new copy —
v1 stays frozen, v2/v3/... accumulate. The loader auto-selects the
highest-numbered version unless `DMV_PROMPTS_VERSION=vN` is set.

This means: every manifest record carries `prompt_hash` = sha256 of
the YAML bytes of the version that produced it, and the YAML file
itself still exists on disk at `prompts/vN/anchor.yaml` for audit.

`build_prompt(spec)` is the single seam between the generator driver
and the prompt template. Every input field consumed by the template
must appear in `spec`.

Snapshot-tested in `tests/pipeline/test_prompt_snapshot.py`.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
PROMPTS_ROOT = REPO / "prompts"


def select_prompts_dir() -> Path:
    """Pick the active prompts/v{N}/ directory.

    Order: env override → highest v{N} on disk → fallback to v1.
    """
    override = os.environ.get("DMV_PROMPTS_VERSION")
    if override:
        path = PROMPTS_ROOT / override
        if path.is_dir():
            return path
    versions: list[tuple[int, Path]] = []
    for p in PROMPTS_ROOT.iterdir():
        m = re.fullmatch(r"v(\d+)", p.name)
        if m and p.is_dir():
            versions.append((int(m.group(1)), p))
    if not versions:
        return PROMPTS_ROOT / "v1"
    versions.sort()
    return versions[-1][1]


PROMPTS_DIR = select_prompts_dir()
ANCHOR_PATH = PROMPTS_DIR / "anchor.yaml"
EDIT_PATH = PROMPTS_DIR / "edit.yaml"


class PromptConfig:
    """Frozen view over `prompts/v1/anchor.yaml`.

    The class is intentionally simple — no inheritance, no lazy
    re-loading. Callers should build once at startup and reuse.
    """

    def __init__(self, path: Path = ANCHOR_PATH):
        self.path = path
        self.raw_bytes = path.read_bytes()
        self.data: dict[str, Any] = yaml.safe_load(self.raw_bytes)
        self.template: str = self.data["template"]
        self.slots: list[str] = list(self.data["slots"])
        # Optional fields — present in anchor.yaml, absent (or empty) in edit.yaml.
        # detail_phrases is required (vision QC needs it regardless of mode).
        self.form_features: dict[str, str] = self.data.get("form_features", {})
        self.detail_phrases: dict[str, str] = self.data.get("detail_phrases", {})
        self.strict_clauses: list[str] = list(self.data.get("strict_clauses", []))
        # Per-category overrides — use these when spec carries a
        # `categorySlug` AND the override exists for that (category,
        # style). Falls back silently to the style-level form_features
        # / base strict_clauses when not configured.
        self.form_features_by_category: dict[str, dict[str, str]] = (
            self.data.get("form_features_by_category", {}) or {}
        )
        self.extra_strict_by_category: dict[str, list[str]] = (
            self.data.get("extra_strict_by_category", {}) or {}
        )
        # Optional — passed to backends that support classifier-free
        # guidance (Qwen-Image, Qwen-Image-Edit-2509). Stored as a
        # single string; PromptConfig.negative_prompt is "" when the
        # YAML omits it.
        raw_neg = self.data.get("negative_prompt", "") or ""
        # Normalise whitespace so multi-line YAML blocks become a
        # single-line comma-friendly clause.
        self.negative_prompt: str = " ".join(raw_neg.split()).strip()
        # Hash the bytes verbatim. Any edit (even whitespace) ⇒ new hash.
        self.prompt_hash: str = hashlib.sha256(self.raw_bytes).hexdigest()[:16]

    def detail_phrase_for(self, details: list[str]) -> str:
        """Join 1-2 detail slugs into a single human-readable clause.

        If multiple slugs are present, they are joined with ' and ' so
        the QC checker can verify either form ('shows X', 'shows X
        and Y') in the rendered image.
        """
        if not details:
            raise ValueError("incidentalDetails empty — generator should "
                             "tag a detail before calling build_prompt")
        parts = []
        for slug in details:
            if slug not in self.detail_phrases:
                raise KeyError(
                    f"detail slug {slug!r} not in detail_phrases — add it "
                    f"to prompts/v1/anchor.yaml"
                )
            parts.append(self.detail_phrases[slug])
        return " and ".join(parts)

    def build_prompt(self, spec: dict[str, Any]) -> str:
        """Substitute slots and return the final prompt string.

        Required spec fields depend on which template; the union covers
        anchor.yaml (text-to-image, full studio prompt) and edit.yaml
        (instruction, no FORM/MATERIAL):
          - noun           (e.g. "sofa")
          - style          (e.g. "modern")
          - color          (e.g. "caramel")
          - incidentalDetails  (list[str]) — at least 1 slug
          - material       (anchor only)
        """
        # Always-required.
        for f in ("noun", "style", "color", "incidentalDetails"):
            if f not in spec:
                raise KeyError(f"spec missing required field {f!r}")
        kwargs: dict[str, str] = {
            "noun": spec["noun"],
            "color": spec["color"],
            "style": spec["style"],
            "detail_phrase": self.detail_phrase_for(spec["incidentalDetails"]),
        }
        # FORM/MATERIAL/STRICT — only present when the template needs them.
        if "{form}" in self.template:
            style = spec["style"]
            cat = spec.get("categorySlug")
            # Per-category form_features override the style-only one
            # when both are present in the registry.
            cat_form = (
                self.form_features_by_category.get(cat, {}).get(style)
                if cat else None
            )
            if cat_form is not None:
                form_raw = cat_form
            elif style in self.form_features:
                form_raw = self.form_features[style]
            else:
                raise KeyError(
                    f"unknown style {style!r}; "
                    f"valid: {sorted(self.form_features)}"
                )
            kwargs["form"] = form_raw.strip().replace("\n", " ")
        if "{material}" in self.template:
            if "material" not in spec:
                raise KeyError("spec missing required field 'material'")
            kwargs["material"] = spec["material"]
        if "{strict_clauses}" in self.template:
            cat = spec.get("categorySlug")
            extras = self.extra_strict_by_category.get(cat, []) if cat else []
            kwargs["strict_clauses"] = ", ".join(self.strict_clauses + list(extras))
        return self.template.format(**kwargs).strip()


def load_anchor() -> PromptConfig:
    """Module-level loader for the text-to-image (Phase A) template;
    cached per process."""
    if not hasattr(load_anchor, "_cache"):
        load_anchor._cache = PromptConfig(ANCHOR_PATH)  # type: ignore[attr-defined]
    return load_anchor._cache  # type: ignore[attr-defined]


def load_edit() -> PromptConfig:
    """Module-level loader for the instruction template used by image-
    edit backends (Phase B). Falls back to the anchor template when
    edit.yaml is absent so the t2i CLI path keeps working before
    Phase B is rolled out."""
    if not hasattr(load_edit, "_cache"):
        path = EDIT_PATH if EDIT_PATH.exists() else ANCHOR_PATH
        load_edit._cache = PromptConfig(path)  # type: ignore[attr-defined]
    return load_edit._cache  # type: ignore[attr-defined]
