#!/usr/bin/env python
"""Cue-vocabulary leakage audit — the check that actually backs the paper's claim.

Scans every user-visible text field of the shipped storefront seed against the
full cue vocabulary (100 object phrases, their head nouns, and the 10 colours).
The contract requires ZERO hits. Exits non-zero on any occurrence.

    python scripts/audit_cue_leakage.py
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

# Seeder-only metadata; not in the Prisma schema, never rendered.
NON_RENDERED = {"_cue_object", "_cue_color"}


def text_fields(seed: dict):
    for section in ("categories", "collections", "products", "variants"):
        for row in seed.get(section, []):
            for field, value in row.items():
                if field in NON_RENDERED or not isinstance(value, str) or not value:
                    continue
                yield section, field, value


def scan(fields, vocab, label):
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in vocab) + r")\b", re.I)
    counts, examples = collections.Counter(), []
    for section, field, value in fields:
        for m in pattern.finditer(value):
            counts[m.group(0).lower()] += 1
            if len(examples) < 5:
                examples.append(f"{section}.{field}: {value[:80]!r} -> {m.group(0)!r}")
    total = sum(counts.values())
    print(f"  {label:34s} {len(vocab):4d} terms -> {total} hits")
    for line in examples:
        print(f"      {line}")
    return total


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Path(__file__).resolve().parents[1] / "data" / "vismem_diag_v2"

    seed = json.loads((root / "seed_v2.json").read_text())
    rows = json.loads((root / "cue_registry.json").read_text())["rows"]

    objects = sorted({r["cue_object"] for r in rows})
    colours = sorted({r["cue_color"] for r in rows})
    heads = sorted({o.split()[-1].strip("()") for o in objects})

    pairs = [(r["cue_object"], r["cue_color"]) for r in rows]
    print(f"cue registry: {len(rows)} products, {len(set(pairs))} distinct (object, colour) pairs")
    if len(set(pairs)) != len(rows):
        print("  FAIL: cue vocabulary is not bijective", file=sys.stderr)
        return 1

    fields = list(text_fields(seed))
    print(f"scanning {len(fields)} rendered text fields")
    total = 0
    for vocab, label in ((objects, "cue objects (full phrase)"),
                         (heads, "cue object head nouns"),
                         (colours, "cue colours")):
        total += scan(fields, vocab, label)

    if total:
        print(f"\nL2 leakage audit FAILED: {total} cue-vocabulary occurrences", file=sys.stderr)
        return 1
    print("\nL2 leakage audit PASSED: 0 cue-vocabulary occurrences in any rendered text field")
    return 0


if __name__ == "__main__":
    sys.exit(main())
