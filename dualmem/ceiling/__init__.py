"""Three ceiling runners measuring perception / oracle-retrieval / full-pipeline.

Each runner has the same evaluate(trial, system, vlm) -> CellResult shape so
the unified runner can dispatch by `cell.ceiling`.
"""

from dualmem.ceiling.perception import run_perception
from dualmem.ceiling.oracle_retrieval import run_oracle_retrieval
from dualmem.ceiling.full_pipeline import run_full_pipeline

__all__ = ["run_perception", "run_oracle_retrieval", "run_full_pipeline"]
