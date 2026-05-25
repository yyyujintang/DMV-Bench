"""Inventory: procedurally generated 3-axis benchmark assets.

Axis 1 (Visual granularity): tier-1/2/3 ΔE-calibrated color/texture deltas.
Axis 2 (Text leakage): 5 templated descriptions per category × leakage level.
Axis 3 (Ceiling) is wired in the runner, not the inventory.
"""
from dualmem.inventory.spec import (
    CATEGORIES, VARIANTS_PER_CATEGORY, ProductSpec, InventoryManifest,
)
from dualmem.inventory.builder import build_inventory
from dualmem.inventory.loader import load_manifest, make_trial

__all__ = [
    "CATEGORIES", "VARIANTS_PER_CATEGORY", "ProductSpec", "InventoryManifest",
    "build_inventory", "load_manifest", "make_trial",
]
