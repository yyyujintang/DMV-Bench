"""VisMem-Diag analysis: layer-gap tables, retrieval-verification figures,
2D heatmaps stratified by granularity × leakage.
"""
from dualmem.analysis.figures import (
    layer_gap_figure,
    cube_figure,
    granularity_leakage_heatmap,
    leakage_comparison_figure,
    summary_table_to_markdown,
)

__all__ = [
    "layer_gap_figure",
    "cube_figure",
    "granularity_leakage_heatmap",
    "leakage_comparison_figure",
    "summary_table_to_markdown",
]
