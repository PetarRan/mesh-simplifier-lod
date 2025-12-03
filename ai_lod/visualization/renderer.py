import numpy as np
from pathlib import Path
from PIL import Image


def render_comparison(meshes, labels, output_path, resolution=512):
    """
    render side-by-side comparison of multiple meshes
    meshes: list of trimesh objects
    labels: list of strings
    output_path: where to save comparison image
    """
    from ..rendering import OffscreenRenderer

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = OffscreenRenderer(resolution=resolution)
    renders = []

    for mesh, label in zip(meshes, labels):
        print(f"rendering {label}...")
        views = renderer.render_views(mesh, num_views=1)
        renders.append(views[0]["image"])

    renderer.cleanup()

    # stitch images horizontally
    total_width = resolution * len(renders)
    combined = np.zeros((resolution, total_width, 3), dtype=np.uint8)

    for i, render in enumerate(renders):
        combined[:, i * resolution: (i + 1) * resolution] = render

    # save
    img = Image.fromarray(combined)
    img.save(output_path)
    print(f"saved comparison: {output_path}")

    return combined
