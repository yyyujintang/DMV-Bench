"""Real-photo inventory builder using Nano Banana.

Per (category, tier), generates ONE base reference photo of the product
in a neutral color. Then for each of 4 variants at that tier, asks
Nano Banana to edit that reference to a specific color (tier-controlled
delta from a base hue). Result: 4 variants × 3 tiers per category, all
sharing the same product silhouette / lighting / framing.

CRITICAL: A single base per (category, tier) means tier-1 / tier-2 / tier-3
have THREE separate base photos. That's intentional: at fine grain we want
variants that share the closest possible non-color attributes, which is
best achieved by basing them on a tier-specific reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from dualmem.inventory.spec import CATEGORIES, VARIANTS_PER_CATEGORY, LEAKAGE_TEMPLATES, render_leakage
from dualmem.inventory.nano_banana import NanoBanana, save_image


# ---------------------------------------------------------------------------
# Category descriptions (rich enough for Nano Banana to produce real photos)
# ---------------------------------------------------------------------------
CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "sofa":    "modern two-seat loveseat with sloped armrests and slender brushed-metal legs",
    "lamp":    "modern table lamp with a fabric drum lampshade and a slim turned-wood base",
    "vase":    "tall slender ceramic floor vase with a matte glazed finish and gentle taper",
    "rug":     "plush rectangular area rug with a low-pile uniform texture, displayed flat",
    "cushion": "square decorative throw cushion with linen-like cover and visible piping along the edges",
    "chair":   "modern accent armchair with curved boucle-style upholstery and wooden tapered legs",
    "table":   "small round side table with a solid-wood top and three slender hairpin legs",
    "shelf":   "single-tier floating wall shelf, simple rectangular bracket-less profile",
    "mirror":  "round wall mirror with a clean metal frame, no decorative engraving",
    "planter": "medium stoneware indoor planter with a smooth matte finish and slight taper",
}


# ---------------------------------------------------------------------------
# Per-tier color ladder.
# Tier 1 (coarse): 4 named colors from far corners of the color wheel.
# Tier 2 (medium): 4 named colors within the same hue family.
# Tier 3 (fine):   4 named colors that are nearly identical neutrals.
#
# We use COLOR NAMES (not RGB) in the prompts so that Nano Banana renders
# realistic colors of those names. We trust the model's color knowledge.
# Variant-level ground truth is the (tier_id, variant_idx) pair, not RGB.
# ---------------------------------------------------------------------------
TIER_COLORS: Dict[int, List[str]] = {
    1: ["a deep navy blue color",
        "a warm terracotta red color",
        "a soft sage green color",
        "a bright marigold yellow color"],
    2: ["a sky blue color",
        "a teal blue-green color",
        "a steel blue color",
        "a deep cobalt blue color"],
    3: ["a charcoal gray color",
        "a slate gray color",
        "a graphite gray color",
        "a gunmetal gray color"],
}


# ---------------------------------------------------------------------------
# Base color per tier — the "anchor" reference photo's color before any
# variant edit. The variants then differ from each other (not from this
# anchor) by re-coloring to TIER_COLORS[tier][i].
# ---------------------------------------------------------------------------
TIER_BASE_COLOR: Dict[int, str] = {
    1: "a neutral cream-beige",
    2: "a neutral light gray",
    3: "a neutral medium gray",
}


def _attribute_for_category(cat_id: str) -> str:
    """What attribute is being varied? For most items, 'color of the main surface'."""
    if cat_id in {"rug", "cushion"}:
        return "fabric color"
    if cat_id in {"vase", "planter"}:
        return "glaze color"
    return "primary surface color"


def build_real_inventory(
    out_dir: str,
    nano: NanoBanana,
    categories: List[dict] = None,
    tiers: List[int] = (1, 2, 3),
    image_size: int = 512,
    verbose: bool = True,
) -> dict:
    categories = categories or CATEGORIES
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "descriptions").mkdir(parents=True, exist_ok=True)
    (out / "_refs").mkdir(parents=True, exist_ok=True)

    products = {}
    leakage_text = {}
    base_paths = {}      # (cat, tier) -> reference photo path
    t_start = time.time()
    n_calls = 0

    for cat in categories:
        cat_id = cat["id"]
        desc = CATEGORY_DESCRIPTIONS.get(cat_id, cat["label"])
        (out / "images" / cat_id).mkdir(parents=True, exist_ok=True)
        products[cat_id] = {}

        # Leakage text (no API needed).
        leakage_text[cat_id] = {}
        for lvl, tpl in LEAKAGE_TEMPLATES.items():
            text = render_leakage(tpl, cat)
            leakage_text[cat_id][lvl] = text
            (out / "descriptions" / f"{cat_id}_l{lvl}.txt").write_text(text)

        attribute = _attribute_for_category(cat_id)

        # Generate one base reference per tier.
        for tier in tiers:
            ref_path = out / "_refs" / f"{cat_id}_t{tier}.png"
            if ref_path.exists():
                if verbose: print(f"  [skip] reference {ref_path} exists")
            else:
                ref_desc = f"{desc} in {TIER_BASE_COLOR[tier]} finish"
                if verbose: print(f"  [gen ref] {cat_id} t{tier}: {ref_desc[:80]}")
                img = nano.generate_reference(ref_desc)
                n_calls += 1
                save_image(img, str(ref_path), size=image_size)
            base_paths[(cat_id, tier)] = str(ref_path)

        # For each tier × variant, generate a variant by editing the tier reference.
        for v_idx in range(VARIANTS_PER_CATEGORY):
            v_name = f"var_{chr(ord('a') + v_idx)}"
            slug = f"{cat_id}{v_idx}"
            spec = {"slug": slug, "image_by_tier": {}}
            for tier in tiers:
                ref_path = base_paths[(cat_id, tier)]
                color = TIER_COLORS[tier][v_idx]
                delta = f"Recolor the {attribute} to {color}. Do not change anything else."
                target_path = out / "images" / cat_id / f"{v_name}_t{tier}.png"
                if target_path.exists():
                    if verbose: print(f"  [skip] {target_path} exists")
                else:
                    if verbose: print(f"  [edit] {cat_id} {v_name} t{tier}: {color}")
                    img = nano.edit_to_variant(ref_path, attribute, delta)
                    n_calls += 1
                    save_image(img, str(target_path), size=image_size)
                spec["image_by_tier"][tier] = str(target_path.relative_to(out))
            products[cat_id][v_name] = spec

        if verbose:
            print(f"  [done] {cat_id}: {sum(1 for _ in base_paths)} refs + {VARIANTS_PER_CATEGORY * len(tiers)} variants  | elapsed {time.time() - t_start:.1f}s, calls={n_calls}")

    # Merge with any pre-existing manifest so a partial build does not erase
    # other categories that were generated in prior calls.
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text())
        except Exception:
            prev = {}
    else:
        prev = {}

    merged_products = dict(prev.get("products", {}))
    merged_leakage = dict(prev.get("leakage_text", {}))
    for cat_id in products:
        merged_products[cat_id] = {
            v_name: {
                "slug": (spec["slug"] if isinstance(spec, dict) else spec.slug),
                "image_by_tier": (spec["image_by_tier"] if isinstance(spec, dict) else spec.image_by_tier),
            }
            for v_name, spec in products[cat_id].items()
        }
    for cat_id, lvls in leakage_text.items():
        merged_leakage[cat_id] = {str(lvl): t for lvl, t in lvls.items()}

    manifest_dict = {
        "n_categories": len(merged_products),
        "variants_per_category": VARIANTS_PER_CATEGORY,
        "tiers": list(tiers),
        "leakage_levels": [0, 1, 2, 3, 4],
        "image_size": image_size,
        "generator": "nano_banana / gemini-2.5-flash-image",
        "tier_colors": {str(k): v for k, v in TIER_COLORS.items()},
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "n_api_calls_last_run": n_calls,
        "products": merged_products,
        "leakage_text": merged_leakage,
    }
    manifest_path.write_text(json.dumps(manifest_dict, indent=2))
    return manifest_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/vismem_diag")
    parser.add_argument("--categories", default="vase",
                        help="comma-separated category ids; 'all' for the full 10")
    parser.add_argument("--tiers", default="1,2,3",
                        help="comma-separated tier ids")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    if args.categories == "all":
        cats = CATEGORIES
    else:
        wanted = [c.strip() for c in args.categories.split(",") if c.strip()]
        by_id = {c["id"]: c for c in CATEGORIES}
        cats = [by_id[c] for c in wanted if c in by_id]
    tiers = [int(t) for t in args.tiers.split(",")]
    if not cats:
        raise SystemExit("no valid categories selected")

    nano = NanoBanana()
    print(f"[real-inventory] building {len(cats)} cat × {VARIANTS_PER_CATEGORY} var × {len(tiers)} tier")
    m = build_real_inventory(args.out, nano, categories=cats, tiers=tiers,
                             image_size=args.image_size)
    print(f"  wrote {args.out}/manifest.json")
    print(f"  api calls: {m['n_api_calls']}")


if __name__ == "__main__":
    main()
