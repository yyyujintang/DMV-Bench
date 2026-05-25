"""Ground-truth resolver for PD (Preference Drift).

Given a per-turn dialogue trajectory + preference state machine + the
catalogue, compute the single product URL that satisfies the final
active preference set and is visually closest to the "positively
reacted" centroid.

Per proposal_tasks_v2.md §7.5:
  1. Hard filter — remove products violating any categorical/numerical
     constraint in P_final.
  2. Visual scoring — CLIP cosine to the centroid of products the user
     positively reacted to during Phases 1–2.
  3. Uniqueness ε margin (default 0.05); else Ambiguous.
"""
from __future__ import annotations

from typing import Any

from ..generators._common import VariantCatalogue, Variant
from .uniqueness import (
    Ambiguous,
    DEFAULT_EPSILON,
    ResolutionResult,
    resolve,
)


def hard_filter(
    catalogue: VariantCatalogue,
    p_final: dict[str, Any],
    candidate_pool: list[Variant] | None = None,
) -> list[Variant]:
    """Apply categorical / numerical filters from `p_final` to the
    variant catalogue. Returns the subset that satisfies *all* active
    preferences. `material` and `ornament` slots are text-only checks
    against Product.material and ProductVariant.incidentalDetails."""
    pool = candidate_pool or catalogue.variants
    out: list[Variant] = []
    style_req = p_final.get("style")
    cat_req = p_final.get("category")
    color_req = p_final.get("color")
    price_max = p_final.get("price_max")
    material_req = p_final.get("material")
    ornament_req = p_final.get("ornament")
    for v in pool:
        if style_req and v.style_slug != style_req:
            continue
        if cat_req and v.category_slug != cat_req:
            continue
        if color_req and v.color_name.lower() != color_req.lower():
            continue
        if price_max is not None and v.price > price_max:
            continue
        if material_req and material_req.lower() not in (v.product_material or "").lower():
            continue
        if ornament_req:
            # ornament is treated as a tag-style check against
            # incidentalDetails; we don't have those on Variant directly,
            # so the generator handles this via cross_turn_predicates and
            # the resolver skips ornament here unless the catalogue
            # surfaces it.
            pass
        out.append(v)
    return out


def resolve_pd(
    catalogue: VariantCatalogue,
    p_final: dict[str, Any],
    positive_reaction_hashes: list[str],
    cache: dict[str, list[float]],
    epsilon: float = DEFAULT_EPSILON,
    exclude_hashes: list[str] | None = None,
) -> ResolutionResult:
    """Run the full PD resolution policy.

    The GT must be a NEW product (not one the user already reacted to in
    setup) — we exclude `positive_reaction_hashes ∪ exclude_hashes` from
    the candidate pool before scoring.

    Raises:
      Ambiguous if top-1 margin < epsilon
      ValueError if hard filter yields 0 candidates after exclusions
    """
    candidates = hard_filter(catalogue, p_final)
    if not candidates:
        raise ValueError(f"hard filter yielded 0 candidates for {p_final}")
    if not positive_reaction_hashes:
        raise ValueError("no positive reactions recorded — cannot centroid")
    blocked = set(positive_reaction_hashes) | set(exclude_hashes or [])
    filtered = [v for v in candidates if v.url_hash not in blocked]
    if not filtered:
        raise ValueError(
            f"all {len(candidates)} candidates were excluded (anchors/seen)"
        )
    return resolve(
        [v.url_hash for v in filtered],
        positive_reaction_hashes,
        cache=cache,
        epsilon=epsilon,
    )
