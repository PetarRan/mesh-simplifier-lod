#!/usr/bin/env python3
"""
script to generate comprehensive AI vs Standard QEM comparison
"""

from visualization import paint_importance_heatmap
from qem_simplifier import QEMSimplifier
from ai_importance import SaliencyExtractor, project_importance_to_vertices
from rendering import OffscreenRenderer
from preprocessing import load_mesh
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core"))


def simple_orient_mesh(mesh):
    """Simple orientation that works for most common mesh files"""
    mesh_centered = mesh.copy()
    mesh_centered.vertices -= mesh_centered.centroid

    vertices = mesh_centered.vertices

    # Try different common orientations and pick the one that looks most reasonable
    # We'll check which orientation puts most vertices in the upper half of Y (standing upright)
    orientations = []

    # Option 1: Original orientation
    orientations.append(vertices.copy())

    # Option 2: Swap Y and Z (common for many mesh formats)
    vertices_yz = vertices.copy()
    temp = vertices_yz[:, 1].copy()
    vertices_yz[:, 1] = vertices_yz[:, 2]
    vertices_yz[:, 2] = temp
    orientations.append(vertices_yz)

    # Option 3: Swap Y and -Z (another common variant)
    vertices_ynegz = vertices.copy()
    temp = vertices_ynegz[:, 1].copy()
    vertices_ynegz[:, 1] = -vertices_ynegz[:, 2]
    vertices_ynegz[:, 2] = temp
    orientations.append(vertices_ynegz)

    # Option 4: Original but flipped horizontally (for backwards models)
    vertices_flip = vertices.copy()
    vertices_flip[:, 0] = -vertices_flip[:, 0]
    orientations.append(vertices_flip)

    # Score each orientation based on how "upright" it is
    # Good orientation: most vertices should have positive Y (above ground)
    # and the model should extend more in X than in Z depth (facing forward)
    best_score = -float("inf")
    best_orientation = 0

    for i, oriented_vertices in enumerate(orientations):
        # Score 1: How many vertices are above ground (Y > 0)
        above_ground_ratio = np.sum(oriented_vertices[:, 1] > 0) / len(
            oriented_vertices
        )

        # Score 2: Width vs depth ratio (model should be wider than deep)
        width = oriented_vertices[:, 0].max() - oriented_vertices[:, 0].min()
        depth = oriented_vertices[:, 2].max() - oriented_vertices[:, 2].min()
        width_depth_ratio = width / (depth + 1e-6)  # Avoid division by zero

        # Combined score
        score = above_ground_ratio * 0.7 + min(width_depth_ratio, 2.0) * 0.3

        if score > best_score:
            best_score = score
            best_orientation = i

    # Apply the best orientation
    mesh_centered.vertices = orientations[best_orientation]

    # Additional rotation to pitch model forward (look ahead instead of down)
    # This rotates around X axis to tilt head down
    angle_x = np.pi / 2  # +90 degrees (positive = look down)
    cos_x = np.cos(angle_x)
    sin_x = np.sin(angle_x)

    rotation_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])

    # Apply pitch rotation
    vertices = mesh_centered.vertices
    mesh_centered.vertices = vertices @ rotation_x.T

    return mesh_centered


def render_comparison_view(mesh, resolution=800, use_vertex_colors=False):
    mesh_centered = simple_orient_mesh(mesh)
    # Create figure for this mesh
    fig = plt.figure(figsize=(resolution / 100, resolution / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # Create triangles for plotting
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

    # Create collection of triangles
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
    # Modern matplotlib uses buffer_rgba() instead of tostring_rgb()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape((h, w, 4))[:, :, :3]  # Drop alpha channel, keep RGB

    plt.close(fig)

    return img


def create_comparison_figure(mesh_path, output_dir, ratios=[0.5, 0.2, 0.05]):
    """Create comprehensive comparison figure"""
    print(f"\nProcessing {mesh_path.name}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    print("  Loading mesh...")
    mesh = load_mesh(str(mesh_path))
    original_faces = len(mesh.faces)

    # Extract AI importance
    print("  Extracting AI importance...")
    renderer = OffscreenRenderer(resolution=256)
    views = renderer.render_views(mesh, num_views=6)

    extractor = SaliencyExtractor(model_name="facebook/dinov2-small")
    saliency_maps = extractor.extract_multi_view_saliency(views)
    importance = project_importance_to_vertices(mesh, views, saliency_maps)
    renderer.cleanup()

    print(
        f"  Importance stats: min={importance.min():.3f}, max={importance.max():.3f}, mean={importance.mean():.3f}"
    )

    # Generate LODs for each ratio
    results = []

    for ratio in ratios:
        print(f"\n  Generating LODs at {ratio * 100:.0f}% ratio...")

        # Standard QEM
        print("    Standard QEM...")
        qem_standard = QEMSimplifier(mesh, importance=None, alpha=0.0)
        lod_standard = qem_standard.simplify(target_ratio=ratio)

        # AI-guided QEM
        print("    AI-guided QEM...")
        qem_ai = QEMSimplifier(mesh, importance=importance, alpha=1.0)
        lod_ai = qem_ai.simplify(target_ratio=ratio)

        results.append(
            {
                "ratio": ratio,
                "standard": lod_standard,
                "ai": lod_ai,
                "standard_faces": len(lod_standard.faces),
                "ai_faces": len(lod_ai.faces),
            }
        )

    # Create comprehensive comparison figure
    print("\n  Rendering comparison images...")

    # Figure with 4 rows: original + 3 LOD levels
    # Each row has 3 columns: Standard QEM, AI-guided QEM, Difference heatmap
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 3, hspace=0.15, wspace=0.1)

    row_labels = ["Original", "LOD 50%", "LOD 20%", "LOD 5%"]

    for row_idx, (label, result) in enumerate(zip(row_labels, [None] + results)):
        # Original row
        if result is None:
            # Render original
            img = render_comparison_view(mesh)

            ax = fig.add_subplot(gs[row_idx, 0])
            ax.imshow(img)
            ax.set_title(
                f"{label}\n{original_faces:,} faces", fontsize=16, fontweight="bold"
            )
            ax.axis("off")

            # Importance heatmap
            heatmap_mesh = paint_importance_heatmap(mesh, importance)
            heatmap_img = render_comparison_view(heatmap_mesh, use_vertex_colors=True)

            ax = fig.add_subplot(gs[row_idx, 1])
            ax.imshow(heatmap_img)
            ax.set_title(
                "AI Importance Heatmap\n(warmer = more important)",
                fontsize=16,
                fontweight="bold",
            )
            ax.axis("off")

            # Stats
            ax = fig.add_subplot(gs[row_idx, 2])
            ax.axis("off")
            stats_text = (
                f"Mesh Statistics:\n"
                f"Faces: {original_faces:,}\n"
                f"Vertices: {len(mesh.vertices):,}\n\n"
                f"Importance Map:\n"
                f"Min: {importance.min():.3f}\n"
                f"Max: {importance.max():.3f}\n"
                f"Mean: {importance.mean():.3f}\n"
                f"Std: {importance.std():.3f}\n\n"
                f"Coverage: {(importance > 0).sum() / len(importance) * 100:.1f}%"
            )
            ax.text(
                0.1,
                0.5,
                stats_text,
                fontsize=14,
                verticalalignment="center",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        else:
            # LOD comparison row
            ratio = result["ratio"]

            # Standard QEM
            img_standard = render_comparison_view(result["standard"])
            ax = fig.add_subplot(gs[row_idx, 0])
            ax.imshow(img_standard)
            ax.set_title(
                f"Standard QEM\n{result['standard_faces']:,} faces ({result['standard_faces'] / original_faces * 100:.1f}%)",
                fontsize=14,
            )
            ax.axis("off")

            # AI-guided QEM
            img_ai = render_comparison_view(result["ai"])
            ax = fig.add_subplot(gs[row_idx, 1])
            ax.imshow(img_ai)
            ax.set_title(
                f"AI-Guided QEM (α=1.0)\n{result['ai_faces']:,} faces ({result['ai_faces'] / original_faces * 100:.1f}%)",
                fontsize=14,
            )
            ax.axis("off")

            # Metrics comparison
            ax = fig.add_subplot(gs[row_idx, 2])
            ax.axis("off")

            metrics_text = (
                f"Target Ratio: {ratio * 100:.0f}%\n\n"
                f"Standard QEM:\n"
                f"  Faces: {result['standard_faces']:,}\n"
                f"  Vertices: {len(result['standard'].vertices):,}\n"
                f"  Actual: {result['standard_faces'] / original_faces * 100:.1f}%\n\n"
                f"AI-Guided QEM:\n"
                f"  Faces: {result['ai_faces']:,}\n"
                f"  Vertices: {len(result['ai'].vertices):,}\n"
                f"  Actual: {result['ai_faces'] / original_faces * 100:.1f}%\n\n"
                f"Difference:\n"
                f"  Δ Faces: {abs(result['ai_faces'] - result['standard_faces']):,}"
            )
            ax.text(
                0.1,
                0.5,
                metrics_text,
                fontsize=12,
                verticalalignment="center",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            )

    fig.suptitle(
        f"AI-Guided vs Standard QEM Simplification: {mesh_path.stem}",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )

    output_path = output_dir / f"comparison_{mesh_path.stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\n  Saved comparison to: {output_path}")

    # Save metrics as text
    metrics_path = output_dir / f"metrics_{mesh_path.stem}.txt"
    with open(metrics_path, "w") as f:
        f.write(f"AI vs Standard QEM Comparison: {mesh_path.name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"Original Mesh: {original_faces:,} faces, {len(mesh.vertices):,} vertices\n\n"
        )

        f.write("Importance Map Statistics:\n")
        f.write(f"  Min: {importance.min():.4f}\n")
        f.write(f"  Max: {importance.max():.4f}\n")
        f.write(f"  Mean: {importance.mean():.4f}\n")
        f.write(f"  Std: {importance.std():.4f}\n")
        f.write(
            f"  Coverage: {(importance > 0).sum() / len(importance) * 100:.2f}%\n\n"
        )

        for i, result in enumerate(results, 1):
            ratio = result["ratio"]
            f.write(f"\nLOD Level {i} (Target: {ratio * 100:.0f}%):\n")
            f.write("-" * 80 + "\n")
            f.write(f"Standard QEM:\n")
            f.write(
                f"  Faces: {result['standard_faces']:,} ({result['standard_faces'] / original_faces * 100:.2f}%)\n"
            )
            f.write(f"  Vertices: {len(result['standard'].vertices):,}\n")
            f.write(f"\nAI-Guided QEM (α=1.0):\n")
            f.write(
                f"  Faces: {result['ai_faces']:,} ({result['ai_faces'] / original_faces * 100:.2f}%)\n"
            )
            f.write(f"  Vertices: {len(result['ai'].vertices):,}\n")
            f.write(f"\nDifference:\n")
            f.write(
                f"  Δ Faces: {abs(result['ai_faces'] - result['standard_faces']):,}\n"
            )

    print(f"  Saved metrics to: {metrics_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare AI vs standard QEM")
    parser.add_argument("--mesh", required=True, help="Path to mesh file")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    mesh_name = mesh_path.stem
    output_dir = Path(f"output/{mesh_name}_compare")

    if not mesh_path.exists():
        print(f"Error: Mesh not found at {mesh_path}")
        return

    print("=" * 80)
    print("AI-Guided vs Standard QEM Comparison Generator")
    print("=" * 80)

    # Generate comprehensive comparison
    results = create_comparison_figure(mesh_path, output_dir, ratios=[0.5, 0.2, 0.05])

    print("\n" + "=" * 80)
    print("Comparison generation complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
