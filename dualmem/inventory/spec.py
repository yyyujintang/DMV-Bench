"""Inventory spec: categories, variant scheme, manifest schema.

We use procedurally generated SVG-style synthetic product cards rather
than Gemini-generated photographs because (a) the result is exactly
reproducible from a seed, (b) we control ΔE perfectly, and (c) there
is no risk of latent text-leakage from a generative model embedding
brand cues into the image. The trade-off — synthetic cards are visually
unlike real e-commerce photos — is acceptable for a diagnostic
benchmark whose target is decomposing memory failures, not testing
domain transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Ten product categories. The "shape_family" controls the procedural
# silhouette drawn into the synthetic image; "label" is what appears
# on the catalog card; "leakage_template_key" indexes into LEAKAGE_TEMPLATES.
CATEGORIES: List[dict] = [
    {"id": "sofa",    "shape": "sofa",    "label": "Studio Loveseat",       "noun": "loveseat",     "material": "performance fabric"},
    {"id": "lamp",    "shape": "lamp",    "label": "Reading Lamp",          "noun": "table lamp",   "material": "powder-coated steel"},
    {"id": "vase",    "shape": "vase",    "label": "Ceramic Vase",          "noun": "vase",         "material": "matte glazed ceramic"},
    {"id": "rug",     "shape": "rug",     "label": "Area Rug",              "noun": "area rug",     "material": "wool blend"},
    {"id": "cushion", "shape": "cushion", "label": "Throw Cushion",         "noun": "cushion",      "material": "linen cover"},
    {"id": "chair",   "shape": "chair",   "label": "Accent Chair",          "noun": "armchair",     "material": "boucle upholstery"},
    {"id": "table",   "shape": "table",   "label": "Side Table",            "noun": "side table",   "material": "solid oak"},
    {"id": "shelf",   "shape": "shelf",   "label": "Wall Shelf",            "noun": "floating shelf","material": "engineered wood"},
    {"id": "mirror",  "shape": "mirror",  "label": "Wall Mirror",           "noun": "round mirror", "material": "brass frame"},
    {"id": "planter", "shape": "planter", "label": "Indoor Planter",        "noun": "planter",      "material": "stoneware"},
]

VARIANTS_PER_CATEGORY = 4   # 4 cards per category (one is the anchor in any trial)


# Five leakage levels. Per category, we instantiate via .format(noun=..., material=...).
# Level 0 (masked): no descriptive content beyond a uniform stock blurb.
# Level 1 (coarse): coarse class signal, useless for ranking within-class.
# Level 2 (material): mentions function/material; cannot distinguish variants on color/texture.
# Level 3 (specific): mentions concrete features but still attribute-ambiguous.
# Level 4 (full): full marketing copy with adjectives that could leak variant clues.
#
# IMPORTANT: NONE of these templates mention color/texture variant-specific words.
# That's enforced — they only mention shared attributes. The agent must use vision
# (or memory) to discriminate among variants. The leakage axis controls how much
# OTHER signal (category/material/feature) is available to text-only memory.
LEAKAGE_TEMPLATES: Dict[int, str] = {
    0: "Studio Living Collection product. Designed for modern interiors. Dimensions and care details on the spec sheet.",
    1: "A piece from the Studio Living Collection. Suitable for a wide range of contemporary rooms.",
    2: "A {noun} from the Studio Living Collection. Constructed with {material} on a hardwood-style frame for everyday durability.",
    3: "A two-seat {noun} from the Studio Living Collection, sized for compact rooms. Built around a hardwood-style frame and finished in {material}. Includes brushed metal legs and a 20-degree sloped backrest.",
    4: "Crafted for contemporary living spaces, this {noun} from our Studio Living Collection features a clean modern silhouette with sloped armrests and slender brushed metal legs. The upholstery is durable {material} ideal for everyday use. Studio Living Collection — designed for small apartments, reading nooks, and modern lounges. Dimensions: 64\" W x 33\" D x 33\" H.",
}


@dataclass
class ProductSpec:
    """One (category, variant) — the actual on-disk product."""
    category: str
    variant: str
    slug: str
    # Per grain tier, the path to that variant's image.
    image_by_tier: Dict[int, str] = field(default_factory=dict)
    # The variant's appearance descriptor in HSL space (used by the procedural
    # renderer and by ΔE-based calibration). NOT shown in the page — the
    # whole point is that text never leaks color names.
    hue_deg: float = 0.0
    sat: float = 0.55
    lightness: float = 0.55
    pattern_id: int = 0
    # Texture "code" used at fine grain tier (e.g. 0 = smooth, 1 = subtle weave).
    texture_id: int = 0


@dataclass
class InventoryManifest:
    """The on-disk manifest joined to TaskCells at runtime."""
    products: Dict[str, Dict[str, ProductSpec]]   # category -> variant -> ProductSpec
    leakage_text: Dict[str, Dict[int, str]]       # category -> level -> rendered string
    cells: List[dict]                             # serialized TaskCells (filled by builder)

    def get_image(self, category: str, variant: str, grain_tier: int) -> str:
        return self.products[category][variant].image_by_tier[grain_tier]

    def get_leakage(self, category: str, level: int) -> str:
        return self.leakage_text[category][level]


def render_leakage(template: str, cat: dict) -> str:
    """Substitute category-specific nouns/materials into the leakage template."""
    return template.format(noun=cat["noun"], material=cat["material"])
