#!/usr/bin/env python
"""Assign one unique incidental cue to every catalogue product and emit the
image-edit prompt that renders it.

Stage 2 of catalogue construction. Stage 1 (`pipeline/generate.py`) produces
the 1,000 base studio photographs; this script decides which cue each product
carries and writes the prompt that edits it onto the base photo.
`scripts/run_cue_edits.py` then executes those prompts.

The assignment is a bijection, which is what makes probe ground truth
unambiguous: within a category, `style_idx` selects the cue object (10 objects,
one per style) and `prod_idx` selects the colour (10 colours), giving 100
distinct (object, colour) pairs per category. Object vocabularies are disjoint
across categories, so all 1,000 pairs are globally unique and a query like
"the red wool scarf" resolves to exactly one product.

Writes:
    data/vismem_diag_v2/prompts/cue_edit/<cat>/<style>/<idx>.txt
    data/vismem_diag_v2/prompts/cue_edit_manifest.json
    data/vismem_diag_v2/cue_registry.json     (url_hash -> cue, read at run time)

No API calls; pure text emission.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = REPO_ROOT / "data" / "vismem_diag_v2"
VOCAB_PATH = V2_ROOT / "cue_vocab.json"
BASE_MANIFEST = V2_ROOT / "prompts" / "base_manifest.json"
EDIT_PROMPTS_DIR = V2_ROOT / "prompts" / "cue_edit"
WITH_CUE_DIR = V2_ROOT / "images" / "with_cue"
EDIT_MANIFEST = V2_ROOT / "prompts" / "cue_edit_manifest.json"
REGISTRY = V2_ROOT / "cue_registry.json"

STYLES = ["modern", "minimalist", "vintage", "industrial", "scandinavian",
          "bohemian", "mid_century", "rustic", "japandi", "art_deco"]

# The cue-edit template, verbatim as used for the released catalogue.
#   - imperative "Add ..." for the edit model
#   - an explicit placement clause per (category, object)
#   - "keep the rest unchanged" so the model does not morph the product
#   - an explicit scale bound so the cue does not dominate the photo
EDIT_PROMPT_TEMPLATE = (
    "Add a small {color} {object_name}, {placement}, in a naturally placed "
    "way. The object should be subtle and modest in size — clearly visible "
    "but not dominating the scene. Keep the {cat_noun} itself and the "
    "background completely unchanged. Photographic realism, no text, no "
    "watermark, no caption overlay."
)

CATEGORY_NOUN = {
    "chair": "chair", "sofa": "sofa", "lamp": "lamp", "cushion": "cushion",
    "vase": "vase", "rug": "rug", "table": "side table",
    "bookshelf": "bookshelf", "plant_pot": "plant pot",
    "wall_art": "wall art piece",
}


def product_url_hash(cat: str, style: str, prod_idx: int) -> str:
    """The frozen urlHash binding a catalogue product to its storefront URL."""
    key = f"{cat}|{style}|{prod_idx:02d}|v2"
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def emit_cue_edit_prompts() -> None:
    vocab = json.loads(VOCAB_PATH.read_text())
    base = json.loads(BASE_MANIFEST.read_text())
    base_by_key = {(r["cat"], r["style"], r["prod_idx"]): r for r in base["rows"]}

    EDIT_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    WITH_CUE_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows, registry_rows = [], []
    seen_cues: dict = {}
    colors = vocab["colors"]

    for cat, spec in vocab["categories"].items():
        objects = spec["objects"]
        cat_noun = CATEGORY_NOUN[cat]
        for style_idx, style in enumerate(STYLES):
            obj = objects[style_idx]
            for prod_idx in range(10):
                color = colors[prod_idx]

                key = (cat, obj["name"], color)
                if key in seen_cues:
                    raise RuntimeError(
                        f"duplicate cue {key} at ({cat}, {style}, {prod_idx}) — "
                        f"first seen at {seen_cues[key]}")
                seen_cues[key] = (cat, style, prod_idx)

                base_row = base_by_key.get((cat, style, prod_idx))
                if base_row is None:
                    raise RuntimeError(
                        f"missing base photograph for ({cat}, {style}, {prod_idx}) — "
                        f"run pipeline/generate.py first")

                prompt = EDIT_PROMPT_TEMPLATE.format(
                    color=color, object_name=obj["name"],
                    placement=obj["placement"], cat_noun=cat_noun)

                prompt_path = EDIT_PROMPTS_DIR / cat / style / f"{prod_idx:02d}.txt"
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_path.write_text(prompt)

                out_img = WITH_CUE_DIR / cat / style / f"{prod_idx:02d}.png"
                url_hash = product_url_hash(cat, style, prod_idx)

                manifest_rows.append({
                    "cat": cat, "style": style, "prod_idx": prod_idx,
                    "url_hash": url_hash,
                    "base_image_path": base_row["image_path"],
                    "edit_prompt_path": str(prompt_path.relative_to(REPO_ROOT)),
                    "output_image_path": str(out_img.relative_to(REPO_ROOT)),
                    "cue": {"object": obj["name"], "color": color,
                            "placement": obj["placement"]},
                })
                registry_rows.append({
                    "cat": cat, "style": style, "prod_idx": prod_idx,
                    "url_hash": url_hash,
                    "cue_object": obj["name"], "cue_color": color,
                    "cue_id": f"{cat}::{obj['name']}::{color}",
                })

    EDIT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    EDIT_MANIFEST.write_text(json.dumps(
        {"version": "v2.3a", "n_prompts": len(manifest_rows),
         "rows": manifest_rows}, indent=2))
    REGISTRY.write_text(json.dumps(
        {"version": "v2.3a", "n_cues": len(registry_rows),
         "rows": registry_rows}, indent=2))

    n_cues = len({r["cue_id"] for r in registry_rows})
    n_pairs = len({(r["cue_object"], r["cue_color"]) for r in registry_rows})
    n_hashes = len({r["url_hash"] for r in registry_rows})
    print(f"wrote {len(manifest_rows)} edit prompts")
    print(f"  unique cue_ids           {n_cues:5d}  (target 1000)")
    print(f"  unique (object, colour)  {n_pairs:5d}  (target 1000 — bijection)")
    print(f"  unique url_hashes        {n_hashes:5d}  (target 1000)")
    print(f"  manifest  {EDIT_MANIFEST.relative_to(REPO_ROOT)}")
    print(f"  registry  {REGISTRY.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    emit_cue_edit_prompts()
