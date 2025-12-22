#!/usr/bin/env python3
"""
Generate LOD comparison image showing Original, Traditional LOD, and Perceptual LOD side by side.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import trimesh

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core"))

from qem_simplifier import QEMSimplifier
from ai_importance import SaliencyExtractor, project_importance_to_vertices
from rendering import OffscreenRenderer
from preprocessing import load_mesh


def render_comparison_view(mesh, resolution=800, use_wireframe=False):
    """Render mesh - copied from working comparison script"""
    mesh_centered = mesh.copy()
    mesh_centered.vertices -= mesh_centered.centroid

    # Rotate bunny to stand upright - swap Y and Z, then adjust orientation
    vertices_rotated = mesh_centered.vertices.copy()
    # Swap Y and Z to make bunny stand up
    temp = vertices_rotated[:, 1].copy()
    vertices_rotated[:, 1] = vertices_rotated[:, 2]
    vertices_rotated[:, 2] = temp
    mesh_centered.vertices = vertices_rotated

    # Create figure for this mesh
    fig = plt.figure(figsize=(resolution / 100, resolution / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # Create triangles for plotting
    vertices = mesh_centered.vertices
    faces = mesh_centered.faces

    # Determine colors
    if use_wireframe:
        # Use light gray with edge lines for better geometry visibility
        facecolors = (0.85, 0.85, 0.85)
        edgecolors = (0.3, 0.3, 0.3)
    else:
        facecolors = (0.85, 0.85, 0.85)
        edgecolors = "none"

    # Create collection of triangles
    poly3d = Poly3DCollection(
        vertices[faces],
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=0.1,
        alpha=1.0,
    )
    ax.add_collection3d(poly3d)

    # Set viewing angle
    ax.view_init(elev=20, azim=135)

    # Set axis limits
    bounds = mesh_centered.bounds
    extent = bounds[1] - bounds[0]
    max_extent = np.max(extent)
    center = (bounds[0] + bounds[1]) / 2

    ax.set_xlim(center[0] - max_extent / 2, center[0] + max_extent / 2)
    ax.set_ylim(center[1] - max_extent / 2, center[1] + max_extent / 2)
    ax.set_zlim(center[2] - max_extent / 2, center[2] + max_extent / 2)

    # Hide axes
    ax.set_axis_off()

    # Set background color
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Convert to image
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Save and reload approach
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=100, bbox_inches="tight", facecolor="white")
        from PIL import Image

        img = np.array(Image.open(tmp.name))
        os.unlink(tmp.name)

    plt.close(fig)
    return img


def main():
    """Generate comparison image"""
    # Use the existing bunny mesh
    mesh_path = "test_meshes/bunny/reconstruction/bun_zipper.ply"
    output_path = "test_meshes/compare/lod_comparison.png"

    if not Path(mesh_path).exists():
        print(f"Mesh not found: {mesh_path}")
        print("Please download the Stanford bunny mesh first")
        return

    print("Loading mesh...")
    mesh = load_mesh(mesh_path)
    original_faces = len(mesh.faces)

    print("Extracting AI importance...")
    renderer = OffscreenRenderer(resolution=256)
    views = renderer.render_views(mesh, num_views=6)

    extractor = SaliencyExtractor(model_name="facebook/dinov2-small")
    saliency_maps = extractor.extract_multi_view_saliency(views)
    importance = project_importance_to_vertices(mesh, views, saliency_maps)
    renderer.cleanup()

    print("Generating LODs at 50% reduction...")

    # Traditional LOD
    qem_traditional = QEMSimplifier(mesh, importance=None, alpha=0.0)
    traditional_lod = qem_traditional.simplify(target_ratio=0.5)

    # Perceptual LOD
    qem_perceptual = QEMSimplifier(mesh, importance=importance, alpha=1.0)
    perceptual_lod = qem_perceptual.simplify(target_ratio=0.5)

    print("Rendering comparison...")

    # Render all three versions
    original_img = render_comparison_view(mesh, resolution=900, use_wireframe=True)
    traditional_img = render_comparison_view(
        traditional_lod, resolution=900, use_wireframe=True
    )
    perceptual_img = render_comparison_view(
        perceptual_lod, resolution=900, use_wireframe=True
    )

    # Convert all to RGB if they have alpha channel
    if original_img.shape[2] == 4:
        original_img = original_img[:, :, :3]
    if traditional_img.shape[2] == 4:
        traditional_img = traditional_img[:, :, :3]
    if perceptual_img.shape[2] == 4:
        perceptual_img = perceptual_img[:, :, :3]

    # Get actual rendered dimensions
    height = original_img.shape[0]
    width = original_img.shape[1] * 3

    # Combine images horizontally
    combined = np.zeros((height, width, 3), dtype=np.uint8)

    # Place images
    w = original_img.shape[1]
    combined[:, 0:w] = original_img
    combined[:, w : 2 * w] = traditional_img
    combined[:, 2 * w : 3 * w] = perceptual_img

    # Add labels using PIL
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(combined)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except:
        font = ImageFont.load_default()

    # Add labels at bottom
    labels = ["Original", "Traditional LOD", "Perceptual LOD"]
    w = original_img.shape[1]
    positions = [w // 2, w + w // 2, 2 * w + w // 2]

    for label, pos in zip(labels, positions):
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        x = pos - text_width // 2
        y = height - 80

        # Add background for text
        draw.rectangle(
            [x - 10, y - 5, x + text_width + 10, y + 40], fill=(255, 255, 255, 200)
        )
        draw.text((x, y), label, font=font, fill=(0, 0, 0))

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.array(pil_img)).save(output_path)

    print(f"Saved comparison to: {output_path}")
    print(f"Original: {original_faces:,} faces")
    print(f"Traditional: {len(traditional_lod.faces):,} faces")
    print(f"Perceptual: {len(perceptual_lod.faces):,} faces")


if __name__ == "__main__":
    main()
