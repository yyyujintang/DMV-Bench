"""Generate AgentTask instances for the three Proposal_A mechanisms.

The three generators mirror the three human-annotation tasks the website
already exposes under `env/frontend/app/annotate/{phase1,task2,phase2}`:

  - Mechanism 1 / `same_instance`     : agent ↔ /annotate/phase1   (4AFC, K=5)
  - Mechanism 2 / `cross_category`    : agent ↔ /annotate/task2    (catA→catB, K=6)
  - Mechanism 3 / `long_horizon`      : agent ↔ /annotate/phase2   (chain K=8)

Same ground-truth semantics on both sides — the agent's recall_collection_url
is the same /category page the human picks candidates on. URL-match TSR =
clicking the variant with `internalVariantKey == anchor.internalVariantKey`
(for Mech 1 / 3 anchor and ground-truth share both cat AND varKey; for Mech 2
catA != catB but varKey is the same).
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dualmem.inventory import load_manifest
from dualmem.tasks.spec import AgentTask


# Plural -> singular noun used in the recall instruction.
SINGULAR = {
    "vases": "vase",
    "lamps": "lamp",
    "cushions": "cushion",
    "chairs": "chair",
    "rugs": "rug",
    "tables": "side table",
}


def _load_url_map(db_path: str) -> Dict[str, Dict[str, dict]]:
    """Read the Next.js SQLite to map (category_slug, variant_internalKey, grainTier) → product details.

    Returns: {category_slug: {f"{variant_key}_t{tier}": {urlHash, primaryImage, varKey}}}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT c.slug AS category, v.internalVariantKey AS variant_key,
               v.grainTier AS tier, v.urlHash, v.primaryImage
        FROM ProductVariant v
        JOIN Product p ON v.productId = p.id
        JOIN Category c ON p.categoryId = c.id
    """).fetchall()
    conn.close()
    out: Dict[str, Dict[str, dict]] = {}
    for r in rows:
        out.setdefault(r["category"], {})[f"{r['variant_key']}_t{r['tier']}"] = {
            "urlHash": r["urlHash"],
            "primaryImage": r["primaryImage"],
            "varKey": r["variant_key"],
        }
    return out


def _cat_id_for_inventory(cat_slug: str, manifest) -> Optional[str]:
    """DB stores plural ('chairs'); manifest stores singular ('chair'). Resolve."""
    if cat_slug in manifest.products:
        return cat_slug
    singular = cat_slug.rstrip("s")
    if singular in manifest.products:
        return singular
    if cat_slug.endswith("s") and cat_slug[:-1] in manifest.products:
        return cat_slug[:-1]
    return None


def _filler_pool_from_other_cats(
    url_map: Dict[str, Dict[str, dict]],
    exclude_cats: List[str],
    tier: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """Return (url, cat, lookup_key) tuples for filler walks. Optionally tier-locked."""
    pool = []
    for oc, items in url_map.items():
        if oc in exclude_cats:
            continue
        for k, info in items.items():
            if tier is not None and not k.endswith(f"_t{tier}"):
                continue
            pool.append(("/product/" + info["urlHash"], oc, k))
    return pool


# ---------------------------------------------------------------------------
# Mechanism 1 — same_instance
# ---------------------------------------------------------------------------

def generate_same_instance_tasks(
    categories: Optional[List[str]] = None,
    tiers: List[int] = (1, 2, 3),
    n_seeds: int = 1,
    n_filler: int = 5,
    db_path: str = "env/frontend/prisma/dev.db",
    inventory_dir: str = "data/vismem_diag",
) -> List[AgentTask]:
    """Mechanism 1: anchor → 5 fillers → recall same variant at /category/<catA>.

    Mirrors annotate/phase1 (4AFC, intra-category recall).
    """
    url_map = _load_url_map(db_path)
    manifest = load_manifest(inventory_dir)
    if categories is None:
        categories = sorted(url_map.keys())

    tasks: List[AgentTask] = []
    for cat_slug in categories:
        if cat_slug not in url_map:
            continue
        cat_inv = _cat_id_for_inventory(cat_slug, manifest)
        if cat_inv is None:
            continue
        cat_products = manifest.products[cat_inv]
        noun = SINGULAR.get(cat_slug, cat_slug.rstrip("s"))

        for tier in tiers:
            for v_name, spec in cat_products.items():
                lookup_key = f"{v_name}_t{tier}"
                if lookup_key not in url_map[cat_slug]:
                    continue
                anchor_info = url_map[cat_slug][lookup_key]
                for seed in range(n_seeds):
                    rng = random.Random(f"si|{cat_slug}|{v_name}|t{tier}|s{seed}")
                    pool = _filler_pool_from_other_cats(url_map, [cat_slug], tier=tier)
                    rng.shuffle(pool)
                    filler_urls = [item[0] for item in pool[:n_filler]]

                    candidates = sorted(
                        [("/product/" + info["urlHash"], vk)
                         for vk, info in url_map[cat_slug].items()
                         if vk.endswith(f"_t{tier}")]
                    )
                    candidate_urls = [c[0] for c in candidates]
                    correct_index = next(
                        (i for i, c in enumerate(candidates) if c[1] == lookup_key), -1
                    )

                    task_id = f"si_{cat_slug}_{v_name}_t{tier}_s{seed}"
                    tasks.append(AgentTask(
                        task_id=task_id,
                        mechanism="same_instance",
                        category=cat_slug,
                        grain_tier=tier,
                        leakage_level=4,
                        seed=seed,
                        anchor_url="/product/" + anchor_info["urlHash"],
                        anchor_slug=spec.slug,
                        anchor_image_path=spec.image_by_tier[tier],
                        filler_urls=filler_urls,
                        recall_instruction=(
                            f"Earlier I viewed a {noun} I liked. "
                            f"Please go back to the {cat_slug.title()} collection "
                            f"and add the same {noun} to my wishlist."
                        ),
                        recall_collection_url=f"/category/{cat_slug}?tier={tier}&layout={task_id}",
                        expected_url_pattern=f"^/product/{anchor_info['urlHash']}",
                        candidate_urls=candidate_urls,
                        correct_index=correct_index,
                    ))
    return tasks


# ---------------------------------------------------------------------------
# Mechanism 2 — cross_category (analogue / style transfer)
# ---------------------------------------------------------------------------

def generate_cross_category_tasks(
    categories: Optional[List[str]] = None,
    tiers: List[int] = (2, 3),
    n_seeds: int = 1,
    n_filler: int = 6,
    db_path: str = "env/frontend/prisma/dev.db",
    inventory_dir: str = "data/vismem_diag",
) -> List[AgentTask]:
    """Mechanism 2: anchor in catA → 6 fillers from cats != {catA, catB}
    → recall to /category/catB; ground truth = catB variant with same varKey.

    Mirrors annotate/task2 (cross-category style matching). Tier-1 omitted by
    default per task2.ts comment ("tier-1 trivial").
    """
    url_map = _load_url_map(db_path)
    manifest = load_manifest(inventory_dir)
    if categories is None:
        categories = sorted(url_map.keys())
    all_cats = sorted(url_map.keys())

    tasks: List[AgentTask] = []
    for cat_a in categories:
        if cat_a not in url_map:
            continue
        cat_a_inv = _cat_id_for_inventory(cat_a, manifest)
        if cat_a_inv is None:
            continue
        a_products = manifest.products[cat_a_inv]
        noun_a = SINGULAR.get(cat_a, cat_a.rstrip("s"))

        for tier in tiers:
            for v_name, spec in a_products.items():
                lookup_key = f"{v_name}_t{tier}"
                if lookup_key not in url_map[cat_a]:
                    continue
                anchor_info = url_map[cat_a][lookup_key]

                for seed in range(n_seeds):
                    rng = random.Random(f"cc|{cat_a}|{v_name}|t{tier}|s{seed}")
                    # Choose catB deterministically from other categories that
                    # also expose the same varKey at this tier (so ground-truth
                    # exists).
                    eligible_b = [
                        c for c in all_cats
                        if c != cat_a and lookup_key in url_map[c]
                    ]
                    if not eligible_b:
                        continue
                    cat_b = rng.choice(eligible_b)
                    noun_b = SINGULAR.get(cat_b, cat_b.rstrip("s"))
                    gt_info = url_map[cat_b][lookup_key]

                    # Filler: from cats other than catA and catB. Tier-locked
                    # mixed varKeys (so the walk doesn't leak the color theme).
                    pool = _filler_pool_from_other_cats(
                        url_map, exclude_cats=[cat_a, cat_b], tier=tier
                    )
                    rng.shuffle(pool)
                    filler_urls = [item[0] for item in pool[:n_filler]]
                    if len(filler_urls) < n_filler:
                        continue  # not enough other-cat variants at this tier

                    candidates = sorted(
                        [("/product/" + info["urlHash"], vk)
                         for vk, info in url_map[cat_b].items()
                         if vk.endswith(f"_t{tier}")]
                    )
                    candidate_urls = [c[0] for c in candidates]
                    correct_index = next(
                        (i for i, c in enumerate(candidates) if c[1] == lookup_key), -1
                    )

                    task_id = f"cc_{cat_a}_to_{cat_b}_{v_name}_t{tier}_s{seed}"
                    tasks.append(AgentTask(
                        task_id=task_id,
                        mechanism="cross_category",
                        category=cat_a,
                        grain_tier=tier,
                        leakage_level=4,
                        seed=seed,
                        anchor_url="/product/" + anchor_info["urlHash"],
                        anchor_slug=spec.slug,
                        anchor_image_path=spec.image_by_tier[tier],
                        filler_urls=filler_urls,
                        recall_instruction=(
                            f"Earlier I viewed a {noun_a} I liked. "
                            f"Now please go to the {cat_b.title()} collection "
                            f"and add the {noun_b} that best matches the "
                            f"{noun_a}'s color/style theme to my wishlist."
                        ),
                        recall_collection_url=f"/category/{cat_b}?tier={tier}&layout={task_id}",
                        expected_url_pattern=f"^/product/{gt_info['urlHash']}",
                        candidate_urls=candidate_urls,
                        correct_index=correct_index,
                    ))
    return tasks


# ---------------------------------------------------------------------------
# Mechanism 3 — long_horizon (chain / tip-of-tongue)
# ---------------------------------------------------------------------------

def generate_long_horizon_tasks(
    categories: Optional[List[str]] = None,
    tiers: List[int] = (2, 3),
    n_seeds: int = 1,
    chain_length: int = 8,
    db_path: str = "env/frontend/prisma/dev.db",
    inventory_dir: str = "data/vismem_diag",
) -> List[AgentTask]:
    """Mechanism 3: same recall as Mech 1, but with a long K=8 filler chain
    drawn from MIXED categories AND mixed tiers (so visual interference is
    maximal, exercising memory retention rather than perception).

    Mirrors annotate/phase2 (chain task). Tier-1 omitted by default.
    """
    url_map = _load_url_map(db_path)
    manifest = load_manifest(inventory_dir)
    if categories is None:
        categories = sorted(url_map.keys())

    tasks: List[AgentTask] = []
    for cat_slug in categories:
        if cat_slug not in url_map:
            continue
        cat_inv = _cat_id_for_inventory(cat_slug, manifest)
        if cat_inv is None:
            continue
        cat_products = manifest.products[cat_inv]
        noun = SINGULAR.get(cat_slug, cat_slug.rstrip("s"))

        for tier in tiers:
            for v_name, spec in cat_products.items():
                lookup_key = f"{v_name}_t{tier}"
                if lookup_key not in url_map[cat_slug]:
                    continue
                anchor_info = url_map[cat_slug][lookup_key]

                for seed in range(n_seeds):
                    rng = random.Random(f"lh|{cat_slug}|{v_name}|t{tier}|s{seed}")
                    # Mixed-tier other-cat pool (more visual interference).
                    pool = _filler_pool_from_other_cats(
                        url_map, exclude_cats=[cat_slug], tier=None
                    )
                    rng.shuffle(pool)
                    filler_urls = [item[0] for item in pool[:chain_length]]
                    if len(filler_urls) < chain_length:
                        continue

                    candidates = sorted(
                        [("/product/" + info["urlHash"], vk)
                         for vk, info in url_map[cat_slug].items()
                         if vk.endswith(f"_t{tier}")]
                    )
                    candidate_urls = [c[0] for c in candidates]
                    correct_index = next(
                        (i for i, c in enumerate(candidates) if c[1] == lookup_key), -1
                    )

                    task_id = f"lh_{cat_slug}_{v_name}_t{tier}_s{seed}"
                    tasks.append(AgentTask(
                        task_id=task_id,
                        mechanism="long_horizon",
                        category=cat_slug,
                        grain_tier=tier,
                        leakage_level=4,
                        seed=seed,
                        anchor_url="/product/" + anchor_info["urlHash"],
                        anchor_slug=spec.slug,
                        anchor_image_path=spec.image_by_tier[tier],
                        filler_urls=filler_urls,
                        recall_instruction=(
                            f"A while ago I viewed a {noun} I liked. "
                            f"After browsing many other products, please return "
                            f"to the {cat_slug.title()} collection and add the "
                            f"same {noun} to my wishlist."
                        ),
                        recall_collection_url=f"/category/{cat_slug}?tier={tier}&layout={task_id}",
                        expected_url_pattern=f"^/product/{anchor_info['urlHash']}",
                        candidate_urls=candidate_urls,
                        correct_index=correct_index,
                    ))
    return tasks


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

GENERATORS = {
    "same_instance": generate_same_instance_tasks,
    "cross_category": generate_cross_category_tasks,
    "long_horizon": generate_long_horizon_tasks,
}


def generate_tasks(mechanism: str, **kwargs) -> List[AgentTask]:
    if mechanism not in GENERATORS:
        raise ValueError(f"unknown mechanism: {mechanism!r}; choices={list(GENERATORS)}")
    return GENERATORS[mechanism](**kwargs)


if __name__ == "__main__":
    import json
    for mech in ("same_instance", "cross_category", "long_horizon"):
        ts = generate_tasks(mech, categories=["chairs"], tiers=[2, 3], n_seeds=1)
        print(f"=== {mech}: {len(ts)} tasks ===")
        if ts:
            print(json.dumps(ts[0].asdict(), indent=2))
