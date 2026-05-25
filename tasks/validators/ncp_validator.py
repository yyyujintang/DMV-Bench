"""
NCP validator (proposal_tasks.md §4.3).

Two layers:

  1. Schema/contract — handled by Pydantic when the TaskInstance is
     constructed; this module wraps that for surface-uniformity.

  2. HTTP rendering — drives the dev server via /api/session,
     advances to each recall turn, fetches the expected page, and
     runs Python-ported audit rules on the HTML.

The Python audit mirrors `VisMem-Diag/env/frontend/lib/audit.ts` 1:1:

  · anchor_image_visible       — anchor primary_image path in any src=
  · anchor_link_visible        — /product/<urlHash> in any href=
  · anchor_attribute_visible   — data-variant-id=<id> on an element
                                  WITHOUT aria-hidden="true" (the
                                  placeholder element is exempt)

Usage:
  from tasks.validators.ncp_validator import validate_task, ValidationReport
  report = validate_task(task, base_url="http://localhost:3000",
                          catalogue=catalogue)
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import requests

from ..generators._common import VariantCatalogue
from ..schema.task_instance import TaskInstance


# ---------- audit primitives ----------

@dataclass
class Violation:
    rule: str            # "MR1" | "MR2"
    type: str
    severity: str        # "critical" | "warning"
    description: str
    evidence: str


@dataclass
class AnchorIdentity:
    variant_id: str
    url_hash: str
    primary_image: str


def _esc(s: str) -> str:
    return re.escape(s)


def audit_html(html: str, anchors: list[AnchorIdentity]) -> list[Violation]:
    """Mirror of lib/audit.ts auditHtml()."""
    violations: list[Violation] = []
    for a in anchors:
        # MR1: anchor image in src=
        literal = re.compile(rf'src=["\'][^"\']*{_esc(a.primary_image)}[^"\']*["\']')
        if literal.search(html):
            violations.append(Violation(
                rule="MR1",
                type="anchor_image_visible",
                severity="critical",
                description=f"Anchor image {a.primary_image} appears in a rendered src=",
                evidence=a.primary_image,
            ))
        # MR1: anchor image via Next image optimiser
        enc = urllib.parse.quote(a.primary_image, safe="")
        if enc != a.primary_image:
            optimised = re.compile(rf'/_next/image\?[^"\']*url={_esc(enc)}')
            if optimised.search(html):
                violations.append(Violation(
                    rule="MR1",
                    type="anchor_image_visible",
                    severity="critical",
                    description=f"Anchor image {a.primary_image} appears via Next image optimiser",
                    evidence=enc,
                ))
        # MR1: anchor link /product/<urlHash>
        href_re = re.compile(rf'href=["\']/product/{_esc(a.url_hash)}["\']')
        if href_re.search(html):
            violations.append(Violation(
                rule="MR1",
                type="anchor_link_visible",
                severity="critical",
                description=f"Anchor link /product/{a.url_hash} appears in an href",
                evidence=a.url_hash,
            ))
        # MR1: data-variant-id outside placeholder
        attr_re = re.compile(rf'data-variant-id=["\']{_esc(a.variant_id)}["\']')
        for m in attr_re.finditer(html):
            idx = m.start()
            el_start = html.rfind("<", 0, idx)
            el_end = html.find(">", idx)
            if el_start < 0 or el_end < 0:
                continue
            el_open = html[el_start:el_end + 1]
            if re.search(r'aria-hidden=["\']true["\']', el_open):
                continue  # placeholder — exempt
            violations.append(Violation(
                rule="MR1",
                type="anchor_attribute_visible",
                severity="critical",
                description=f"Anchor variant id {a.variant_id} renders on a non-placeholder element",
                evidence=el_open[:200],
            ))
    return violations


# ---------- HTTP probe ----------

@dataclass
class ValidationReport:
    passed: bool
    schema_ok: bool = True
    schema_reason: Optional[str] = None
    recall_turns_audited: int = 0
    violations: list[Violation] = field(default_factory=list)
    transport_errors: list[str] = field(default_factory=list)


def _anchor_identity(task: TaskInstance, catalogue: VariantCatalogue) -> list[AnchorIdentity]:
    out: list[AnchorIdentity] = []
    for a_id in task.ncp_metadata.anchor_variant_ids:
        v = catalogue.by_id(a_id)
        out.append(AnchorIdentity(
            variant_id=v.id, url_hash=v.url_hash, primary_image=v.primary_image,
        ))
    return out


def validate_task(
    task: TaskInstance,
    catalogue: VariantCatalogue,
    base_url: str = "http://localhost:3000",
    request_timeout: float = 30.0,
) -> ValidationReport:
    """
    Schema check (Pydantic, already done at construction) + HTTP probe.

    Flow per task:
      1. POST /api/session with the task's TaskSpec → cookie
      2. For each recall turn:
         a. POST /api/session/turn {"setTurn": <idx>}
         b. GET <expected_url for that turn>
         c. audit the HTML
      3. Aggregate violations
    """
    report = ValidationReport(passed=True)
    anchors = _anchor_identity(task, catalogue)

    # Use a session so the cookie persists across requests.
    s = requests.Session()
    try:
        r = s.post(
            f"{base_url}/api/session",
            json={"taskSpec": task.to_task_spec()},
            timeout=request_timeout,
        )
        r.raise_for_status()
    except Exception as e:
        report.passed = False
        report.transport_errors.append(f"create session failed: {e}")
        return report

    for recall_idx in task.ncp_metadata.recall_turn_indices:
        # Advance to recall_idx
        try:
            r = s.post(
                f"{base_url}/api/session/turn",
                json={"setTurn": recall_idx},
                timeout=request_timeout,
            )
            r.raise_for_status()
        except Exception as e:
            report.passed = False
            report.transport_errors.append(
                f"advance to turn {recall_idx} failed: {e}"
            )
            continue

        # Render the page the agent SHOULD see at this turn.
        target_turn = next((t for t in task.turns if t.turn_index == recall_idx), None)
        if target_turn is None or not target_turn.expected_url:
            report.transport_errors.append(
                f"recall turn {recall_idx} has no expected_url to render"
            )
            continue

        try:
            page = s.get(f"{base_url}{target_turn.expected_url}",
                         timeout=request_timeout)
            page.raise_for_status()
        except Exception as e:
            report.passed = False
            report.transport_errors.append(
                f"GET {target_turn.expected_url} failed: {e}"
            )
            continue

        violations = audit_html(page.text, anchors)
        report.recall_turns_audited += 1
        report.violations.extend(violations)

    if any(v.severity == "critical" for v in report.violations):
        report.passed = False
    return report
