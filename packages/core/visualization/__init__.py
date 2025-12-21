from .heatmap import paint_importance_heatmap, export_heatmap
from .renderer import render_comparison

# For webapp compatibility - expose webapp-specific visualization functions
try:
    from ..webapp.visualization import mesh_to_plotly, mesh_with_colors

    __all__ = [
        "paint_importance_heatmap",
        "export_heatmap",
        "render_comparison",
        "mesh_to_plotly",
        "mesh_with_colors",
    ]
except ImportError:
    __all__ = ["paint_importance_heatmap", "export_heatmap", "render_comparison"]
