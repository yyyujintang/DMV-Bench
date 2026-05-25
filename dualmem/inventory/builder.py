"""Procedural inventory builder.

Generates, for each (category, variant) pair, three calibrated grain-tier
images. Tier-1 variants differ by large hue jumps (~120 deg apart);
tier-2 by ~30 deg; tier-3 by ~10 deg + small lightness/texture deltas
— the "easy to see, hard to name" sweet spot Proposal_A targets.

Images are rendered with PIL only (no Pillow plugins required) into
square 384x384 cards with a category-specific silhouette + variant fill.
The silhouettes are intentionally simple — we are not trying to fool a
human into thinking these are real products; we are trying to test
whether a VLM can recall the right card from memory.

ΔE (perceptual color distance, CIE76 approximation) is logged into the
manifest so the calibration step can verify the tier ladder.
"""

from __future__ import annotations

import colorsys
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFilter

from dualmem.inventory.spec import (
    CATEGORIES,
    VARIANTS_PER_CATEGORY,
    LEAKAGE_TEMPLATES,
    ProductSpec,
    InventoryManifest,
    render_leakage,
)


# ---------------------------------------------------------------------------
# Grain ladder: hue/lightness/texture deltas per tier.
# These are the headline knobs. They will be re-calibrated by the
# human-rater study; the defaults below are the v0 ladder.
# ---------------------------------------------------------------------------
TIER_HUE_DELTAS = {
    1: 90.0,    # coarse: 4 variants 90° apart on color wheel (red/yellow/green/blue family)
    2: 30.0,    # medium: 4 variants 30° apart, within-family but distinguishable
    3: 10.0,    # fine: nearly-adjacent hues
}
TIER_LIGHTNESS_DELTAS = {
    1: 0.0,
    2: 0.04,
    3: 0.06,
}
TIER_TEXTURE = {
    1: False,
    2: False,
    3: True,   # subtle weave at fine tier — "easy to see, hard to name"
}

# Anchor hue rotates per category so different cats don't collide.
CATEGORY_BASE_HUE = {
    "sofa": 10.0, "lamp": 50.0, "vase": 100.0, "rug": 150.0,
    "cushion": 200.0, "chair": 240.0, "table": 280.0, "shelf": 320.0,
    "mirror": 0.0, "planter": 70.0,
}

IMAGE_SIZE = 384
BG = (245, 244, 240)


def _hsl_to_rgb(h_deg: float, s: float, l: float) -> Tuple[int, int, int]:
    """HSL (h in degrees) → 8-bit RGB."""
    r, g, b = colorsys.hls_to_rgb(h_deg / 360.0, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def _rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """sRGB → CIE Lab (D65). Used only for ΔE logging; no perceptual claims."""
    r, g, b = [c / 255.0 for c in rgb]
    def gamma(x):
        return ((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92
    r, g, b = map(gamma, (r, g, b))
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    xn, yn, zn = 0.95047, 1.0, 1.08883
    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)
    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    return (L, a, b_)


def delta_e76(rgb1, rgb2) -> float:
    L1, a1, b1 = _rgb_to_lab(rgb1)
    L2, a2, b2 = _rgb_to_lab(rgb2)
    return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


# ---------------------------------------------------------------------------
# Silhouettes per category (very stylized — we are not faking photos).
# ---------------------------------------------------------------------------
def _draw_silhouette(draw: ImageDraw.ImageDraw, shape: str, fill, outline=(50, 50, 50)):
    W, H = IMAGE_SIZE, IMAGE_SIZE
    if shape == "sofa":
        draw.rounded_rectangle((40, 200, W-40, 320), radius=24, fill=fill, outline=outline, width=2)
        draw.rounded_rectangle((40, 150, 130, 280), radius=18, fill=fill, outline=outline, width=2)
        draw.rounded_rectangle((W-130, 150, W-40, 280), radius=18, fill=fill, outline=outline, width=2)
        draw.rectangle((60, 320, 80, 360), fill=(80, 80, 80))
        draw.rectangle((W-80, 320, W-60, 360), fill=(80, 80, 80))
    elif shape == "lamp":
        draw.polygon([(W//2-70, 110), (W//2+70, 110), (W//2+50, 200), (W//2-50, 200)], fill=fill, outline=outline)
        draw.rectangle((W//2-6, 200, W//2+6, 310), fill=(80, 70, 50))
        draw.ellipse((W//2-50, 305, W//2+50, 340), fill=(70, 60, 40))
    elif shape == "vase":
        draw.polygon([(W//2-60, 130), (W//2-30, 110), (W//2+30, 110), (W//2+60, 130),
                      (W//2+40, 320), (W//2-40, 320)], fill=fill, outline=outline)
    elif shape == "rug":
        draw.rounded_rectangle((60, 130, W-60, 320), radius=8, fill=fill, outline=outline, width=2)
        for y in range(150, 320, 20):
            draw.line((80, y, W-80, y), fill=outline, width=1)
    elif shape == "cushion":
        draw.rounded_rectangle((90, 130, W-90, 310), radius=40, fill=fill, outline=outline, width=2)
    elif shape == "chair":
        draw.rounded_rectangle((100, 110, W-100, 250), radius=12, fill=fill, outline=outline, width=2)
        draw.rounded_rectangle((100, 230, W-100, 290), radius=8, fill=fill, outline=outline, width=2)
        draw.rectangle((110, 290, 130, 340), fill=fill, outline=outline)
        draw.rectangle((W-130, 290, W-110, 340), fill=fill, outline=outline)
    elif shape == "table":
        draw.rounded_rectangle((60, 180, W-60, 220), radius=4, fill=fill, outline=outline, width=2)
        draw.rectangle((90, 220, 110, 340), fill=fill, outline=outline)
        draw.rectangle((W-110, 220, W-90, 340), fill=fill, outline=outline)
    elif shape == "shelf":
        draw.rounded_rectangle((40, 200, W-40, 230), radius=2, fill=fill, outline=outline, width=2)
        draw.rectangle((60, 230, 80, 320), fill=(80, 70, 60))
        draw.rectangle((W-80, 230, W-60, 320), fill=(80, 70, 60))
    elif shape == "mirror":
        draw.ellipse((90, 90, W-90, W-90), fill=fill, outline=outline, width=4)
        draw.ellipse((110, 110, W-110, W-110), fill=(230, 230, 235), outline=None)
    elif shape == "planter":
        draw.polygon([(W//2-80, 200), (W//2+80, 200), (W//2+60, 320), (W//2-60, 320)],
                     fill=fill, outline=outline)
        draw.ellipse((W//2-90, 90, W//2+90, 220), fill=(60, 110, 60), outline=(40, 80, 40), width=2)
    else:
        draw.rectangle((80, 120, W-80, W-80), fill=fill, outline=outline, width=2)


def _draw_card(spec: ProductSpec, cat_meta: dict, with_texture: bool) -> Image.Image:
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BG)
    draw = ImageDraw.Draw(img)
    rgb = _hsl_to_rgb(spec.hue_deg, spec.sat, spec.lightness)
    _draw_silhouette(draw, cat_meta["shape"], rgb)

    # Soft floor shadow
    draw.ellipse((40, 335, IMAGE_SIZE - 40, 360), fill=(220, 220, 220))

    if with_texture:
        # Add a subtle, deterministic dot/weave overlay tied to texture_id.
        overlay = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BG)
        odraw = ImageDraw.Draw(overlay)
        rng = random.Random(hash((spec.category, spec.variant, spec.texture_id)) & 0xFFFFFFFF)
        for _ in range(400):
            x = rng.randint(40, IMAGE_SIZE - 40)
            y = rng.randint(120, IMAGE_SIZE - 60)
            dot = (max(0, rgb[0] - 18), max(0, rgb[1] - 18), max(0, rgb[2] - 18))
            odraw.ellipse((x, y, x + 2, y + 2), fill=dot)
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.6))
        img = Image.blend(img, overlay, alpha=0.25)
    return img


# ---------------------------------------------------------------------------
# Variant scheme: for each grain tier, place 4 variants on the color wheel
# around the category's base hue, separated by TIER_HUE_DELTAS[tier].
# ---------------------------------------------------------------------------
def _variant_hsl(cat_id: str, variant_idx: int, tier: int) -> Tuple[float, float, float]:
    base = CATEGORY_BASE_HUE.get(cat_id, 0.0)
    dh = TIER_HUE_DELTAS[tier]
    # 4 variants on the color wheel, evenly spaced by dh. Asymmetric
    # placement (variant i at base + i*dh) keeps all variants distinct
    # even at tier 1 where a symmetric placement would have wrapped
    # variant_0 and variant_3 onto the same hue.
    h = (base + variant_idx * dh) % 360
    s = 0.55
    l = 0.55 + (variant_idx % 2) * TIER_LIGHTNESS_DELTAS[tier]
    return h, s, l


def build_inventory(out_dir: str, categories: List[dict] = None, seed: int = 0) -> InventoryManifest:
    """Generate the full image+text inventory under out_dir/.

    Layout:
        out_dir/images/<cat>/<variant>_t<tier>.png
        out_dir/descriptions/<cat>_l<level>.txt
        out_dir/manifest.json   (consumed by the loader)

    Returns the InventoryManifest in memory (also serialized to disk).
    """
    categories = categories or CATEGORIES
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "descriptions").mkdir(parents=True, exist_ok=True)

    products: Dict[str, Dict[str, ProductSpec]] = {}
    leakage_text: Dict[str, Dict[int, str]] = {}
    delta_e_log: Dict[str, Dict[int, float]] = {}

    for cat in categories:
        cat_id = cat["id"]
        (out / "images" / cat_id).mkdir(parents=True, exist_ok=True)
        products[cat_id] = {}
        delta_e_log[cat_id] = {}

        # Variant slugs are 8-char deterministic hashes for v9-style URL stability.
        for v_idx in range(VARIANTS_PER_CATEGORY):
            v_name = f"var_{chr(ord('a') + v_idx)}"
            slug = f"{cat_id}{v_idx}{seed:02d}"
            spec = ProductSpec(
                category=cat_id, variant=v_name, slug=slug,
                hue_deg=0, sat=0.55, lightness=0.55,
                pattern_id=v_idx, texture_id=v_idx,
            )
            for tier in (1, 2, 3):
                h, s, l = _variant_hsl(cat_id, v_idx, tier)
                spec_at_tier = ProductSpec(
                    category=cat_id, variant=v_name, slug=slug,
                    hue_deg=h, sat=s, lightness=l,
                    pattern_id=v_idx, texture_id=v_idx,
                )
                img = _draw_card(spec_at_tier, cat, with_texture=TIER_TEXTURE[tier])
                fname = f"{v_name}_t{tier}.png"
                path = out / "images" / cat_id / fname
                img.save(path, format="PNG")
                spec.image_by_tier[tier] = str(path.relative_to(out))
            products[cat_id][v_name] = spec

        # ΔE between variant_0 and variant_1 at each tier (canonical pair).
        for tier in (1, 2, 3):
            h0, s0, l0 = _variant_hsl(cat_id, 0, tier)
            h1, s1, l1 = _variant_hsl(cat_id, 1, tier)
            de = delta_e76(_hsl_to_rgb(h0, s0, l0), _hsl_to_rgb(h1, s1, l1))
            delta_e_log[cat_id][tier] = round(de, 2)

        leakage_text[cat_id] = {}
        for level, tpl in LEAKAGE_TEMPLATES.items():
            text = render_leakage(tpl, cat)
            leakage_text[cat_id][level] = text
            (out / "descriptions" / f"{cat_id}_l{level}.txt").write_text(text)

    manifest_dict = {
        "n_categories": len(categories),
        "variants_per_category": VARIANTS_PER_CATEGORY,
        "tiers": [1, 2, 3],
        "leakage_levels": [0, 1, 2, 3, 4],
        "image_size": IMAGE_SIZE,
        "delta_e_by_tier": delta_e_log,
        "products": {
            cat_id: {
                v_name: {
                    "slug": spec.slug,
                    "image_by_tier": spec.image_by_tier,
                    "hue_deg_t1": _variant_hsl(cat_id, list(products[cat_id]).index(v_name), 1)[0],
                }
                for v_name, spec in cat_products.items()
            }
            for cat_id, cat_products in products.items()
        },
        "leakage_text": {
            cat_id: {str(lvl): t for lvl, t in lvls.items()}
            for cat_id, lvls in leakage_text.items()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest_dict, indent=2))

    return InventoryManifest(products=products, leakage_text=leakage_text, cells=[])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/vismem_diag")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-categories", type=int, default=len(CATEGORIES))
    args = parser.parse_args()
    cats = CATEGORIES[: args.n_categories]
    m = build_inventory(args.out, categories=cats, seed=args.seed)
    print(f"Built inventory: {len(cats)} categories × {VARIANTS_PER_CATEGORY} variants × 3 tiers")
    print(f"  → {args.out}/")
