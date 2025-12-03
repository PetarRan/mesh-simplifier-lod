import numpy as np
import trimesh
from pathlib import Path


def paint_importance_heatmap(mesh, importance, colormap="viridis"):
    """
    paint vertex importance as vertex colors
    returns new mesh with colors
    """
    from matplotlib import cm
    import matplotlib.pyplot as plt

    # get colormap
    cmap = cm.get_cmap(colormap)

    # normalize importance to [0, 1]
    imp_norm = (importance - importance.min()) / \
        (importance.max() - importance.min() + 1e-8)

    # apply colormap
    colors = cmap(imp_norm)[:, :3]  # rgb only, drop alpha

    # create new mesh with colors
    mesh_colored = mesh.copy()
    mesh_colored.visual.vertex_colors = (colors * 255).astype(np.uint8)

    return mesh_colored


def export_heatmap(mesh, importance, output_path, colormap="viridis"):
    """
    export importance heatmap as colored mesh + png render
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # paint mesh
    mesh_colored = paint_importance_heatmap(mesh, importance, colormap)

    # save colored mesh
    mesh_path = output_path.with_suffix(".obj")
    mesh_colored.export(str(mesh_path))
    print(f"saved heatmap mesh: {mesh_path}")

    # render to png
    try:
        from ..rendering import OffscreenRenderer

        renderer = OffscreenRenderer(resolution=512)
        views = renderer.render_views(mesh_colored, num_views=1)

        # save first view
        png_path = output_path.with_suffix(".png")
        from PIL import Image

        img = Image.fromarray(views[0]["image"])
        img.save(png_path)
        print(f"saved heatmap render: {png_path}")

        renderer.cleanup()
    except Exception as e:
        print(f"warning: could not render heatmap png: {e}")

    return mesh_colored
