#!/usr/bin/env python3
"""
Quick test of matplotlib rendering to verify it works before running full pipeline
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

matplotlib.use("Agg")

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core"))

from preprocessing import load_mesh
from ai_importance import SaliencyExtractor
from rendering import OffscreenRenderer
from ai_importance.projection import project_importance_to_vertices
from visualization.heatmap import paint_importance_heatmap


def render_comparison_view(mesh, resolution=800, use_vertex_colors=False):
    """Same rendering function from compare_ai_vs_standard.py"""
    mesh_centered = mesh.copy()
    mesh_centered.vertices -= mesh_centered.centroid

    # Rotate bunny to stand upright - swap Y and Z, then adjust orientation
    vertices_rotated = mesh_centered.vertices.copy()
    # Swap Y and Z to make bunny stand up
    temp = vertices_rotated[:, 1].copy()
    vertices_rotated[:, 1] = vertices_rotated[:, 2]
    vertices_rotated[:, 2] = temp  # Negative to get correct orientation
    mesh_centered.vertices = vertices_rotated

    fig = plt.figure(figsize=(resolution / 100, resolution / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    vertices = mesh_centered.vertices
    faces = mesh_centered.faces

    # Determine colors
    if use_vertex_colors and hasattr(mesh_centered.visual, "vertex_colors"):
        # Use vertex colors from the mesh (for heatmap)
        vertex_colors = mesh_centered.visual.vertex_colors[:, :3] / 255.0
        facecolors = vertex_colors[faces].mean(axis=1)  # Average vertex colors per face
        edgecolors = "none"
    else:
        # Use light gray with edge lines for better geometry visibility
        facecolors = (0.85, 0.85, 0.85)
        edgecolors = (0.3, 0.3, 0.3)

    poly3d = Poly3DCollection(
        vertices[faces],
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=0.1,
        alpha=1.0,
    )
    ax.add_collection3d(poly3d)

    # Set viewing angle - try different angles for bunny
    # elev: positive = looking up from below, negative = looking down from above
    # azim: rotation around vertical axis
    ax.view_init(elev=20, azim=135)

    bounds = mesh_centered.bounds
    extent = bounds[1] - bounds[0]
    max_extent = np.max(extent)
    center = (bounds[0] + bounds[1]) / 2

    ax.set_xlim(center[0] - max_extent / 2, center[0] + max_extent / 2)
    ax.set_ylim(center[1] - max_extent / 2, center[1] + max_extent / 2)
    ax.set_zlim(center[2] - max_extent / 2, center[2] + max_extent / 2)

    ax.set_axis_off()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    # Modern matplotlib uses buffer_rgba() instead of tostring_rgb()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape((h, w, 4))[:, :, :3]  # Drop alpha channel, keep RGB

    plt.close(fig)

    return img


def main():
    mesh_path = Path("test_meshes/bunny/reconstruction/bun_zipper.ply")

    if not mesh_path.exists():
        print(f"Error: Mesh not found at {mesh_path}")
        return

    print("Loading mesh...")
    mesh = load_mesh(str(mesh_path))
    print(
        f"Loaded mesh with {len(mesh.vertices):,} vertices and {len(mesh.faces):,} faces"
    )

    print("\nTesting AI saliency extraction...")
    try:
        # Initialize AI components
        saliency_extractor = SaliencyExtractor()
        renderer = OffscreenRenderer(resolution=256)

        # Render views for saliency extraction
        print("Rendering views for AI saliency...")
        views = renderer.render_views(mesh, num_views=6)

        # Extract saliency maps
        print("Extracting AI saliency maps...")
        saliency_maps = saliency_extractor.extract_multi_view_saliency(views)

        # Project saliency to vertices
        print("Projecting saliency to mesh vertices...")
        importance = project_importance_to_vertices(mesh, views, saliency_maps)

        # Apply heatmap to mesh
        print("Applying heatmap to mesh...")
        mesh_with_heatmap = paint_importance_heatmap(mesh, importance, colormap="hot")

        # Render with heatmap
        print("Rendering with AI heatmap...")
        img = render_comparison_view(mesh_with_heatmap, use_vertex_colors=True)
        print(f"✓ SUCCESS! Rendered image shape: {img.shape}")
        print(f"  Image dimensions: {img.shape[0]}x{img.shape[1]}")
        print(f"  Channels: {img.shape[2]}")

        # Save test image
        output_path = Path("test_render_ai.png")
        plt.imsave(output_path, img)
        print(f"\n✓ AI heatmap render saved to: {output_path}")

        # Also save normal render for comparison
        img_normal = render_comparison_view(mesh)
        output_normal = Path("test_render_normal.png")
        plt.imsave(output_normal, img_normal)
        print(f"✓ Normal render saved to: {output_normal}")

        print("\nAI saliency rendering works! Check if face is properly highlighted.")

    except Exception as e:
        print(f"\n✗ FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
