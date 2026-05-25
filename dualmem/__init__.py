"""VisMem-Diag: A Diagnostic Benchmark for Decomposing Visual Memory Failures.

Three orthogonal axes:
    1. Visual granularity (grain tier 1-3)
    2. Text leakage (level 0-4)
    3. Three-layer ceiling decomposition (perception / oracle-retrieval / full-pipeline)

Three task mechanisms:
    descriptive / analogue / tip_of_tongue
"""

from dualmem.types import (
    TaskCell, Trial, CellResult,
    GrainTier, LeakageLevel, Mechanism, Ceiling,
)

__all__ = [
    "TaskCell", "Trial", "CellResult",
    "GrainTier", "LeakageLevel", "Mechanism", "Ceiling",
]

__version__ = "0.1.0"
