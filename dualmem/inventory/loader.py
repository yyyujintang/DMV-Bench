"""Inventory loader: read manifest.json, produce Trial instances.

This is the bridge from the static on-disk inventory to the runtime
trial objects consumed by Ceiling runners.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from dualmem.inventory.spec import ProductSpec, InventoryManifest, LEAKAGE_TEMPLATES, render_leakage, CATEGORIES, VARIANTS_PER_CATEGORY
from dualmem.types import TaskCell, Trial


_CAT_BY_ID = {c["id"]: c for c in CATEGORIES}


def load_manifest(inventory_dir: str) -> InventoryManifest:
    p = Path(inventory_dir) / "manifest.json"
    d = json.loads(p.read_text())

    products = {}
    for cat_id, cat_products in d["products"].items():
        products[cat_id] = {}
        for v_name, info in cat_products.items():
            products[cat_id][v_name] = ProductSpec(
                category=cat_id,
                variant=v_name,
                slug=info["slug"],
                image_by_tier={int(k): str(Path(inventory_dir) / v) for k, v in info["image_by_tier"].items()},
                hue_deg=info.get("hue_deg_t1", 0.0),
            )
    leakage = {
        cat_id: {int(lvl): t for lvl, t in lvls.items()}
        for cat_id, lvls in d["leakage_text"].items()
    }
    return InventoryManifest(products=products, leakage_text=leakage, cells=[])


def make_trial(cell: TaskCell, manifest: InventoryManifest, n_candidates: int = 4) -> Trial:
    """Construct a runtime Trial from a TaskCell.

    For DESCRIPTIVE mechanism:
        - anchor = variant `cell.variant` at grain_tier
        - candidates = all 4 variants of `cell.category` at grain_tier,
          shuffled by `cell.seed`
        - encode/recall text = leakage_level template for category

    For ANALOGUE (cross-category):
        - anchor = variant at grain_tier
        - candidates = 4 variants from a *different* category that share
          the anchor's hue_deg±20° (style-match probe). Pilot: same-cat fallback.

    For TIP_OF_TONGUE:
        - same as DESCRIPTIVE plus filler_steps = 50.

    Pilot scope: only DESCRIPTIVE is fully wired here. ANALOGUE +
    TIP_OF_TONGUE fall back to descriptive plus a marker in distractors_info
    so the runner can adjust filler step count. Full implementation is
    future work.
    """
    rng = random.Random(int(f"{cell.seed}{cell.grain_tier}{cell.leakage_level}", 10) ^ hash(cell.category + cell.variant))
    cat_products = manifest.products[cell.category]
    anchor_spec = cat_products[cell.variant]

    # Distractors: the other 3 variants in this category.
    distractor_variants = [v for v in cat_products if v != cell.variant]
    rng.shuffle(distractor_variants)
    candidates = [cell.variant] + distractor_variants[: n_candidates - 1]
    rng.shuffle(candidates)
    correct_index = candidates.index(cell.variant)

    candidate_paths = [cat_products[v].image_by_tier[cell.grain_tier] for v in candidates]
    candidate_slugs = [cat_products[v].slug for v in candidates]

    leakage_str = manifest.leakage_text[cell.category][cell.leakage_level]

    filler = 0
    if cell.mechanism == "tip_of_tongue":
        filler = 50

    return Trial(
        cell=cell,
        anchor_path=anchor_spec.image_by_tier[cell.grain_tier],
        anchor_slug=anchor_spec.slug,
        candidate_paths=candidate_paths,
        candidate_slugs=candidate_slugs,
        correct_index=correct_index,
        encode_text=leakage_str,
        recall_text=leakage_str,
        filler_steps=filler,
        distractors_info=[{"variant": v, "slug": cat_products[v].slug} for v in candidates],
    )
