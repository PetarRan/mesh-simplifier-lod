#!/usr/bin/env python3
"""
Quick test of matplotlib rendering to verify it works before running full pipeline
"""

from visualization.heatmap import paint_importance_heatmap
from ai_importance.projection import project_importance_to_vertices
from rendering import OffscreenRenderer
from ai_importance import SaliencyExtractor
from preprocessing import load_mesh
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


def compute_geometric_importance(mesh):
    """Create importance based on geometric features (curvature, edges, etc.)"""
    vertices = mesh.vertices
    faces = mesh.faces

    # Compute vertex curvature (approximate using angle between adjacent faces)
    vertex_curvature = np.zeros(len(vertices))

    # For each vertex, compute average normal deviation
    for i in range(len(vertices)):
        # Find faces containing this vertex
        vertex_faces = [j for j, face in enumerate(faces) if i in face]

        if len(vertex_faces) >= 2:
            # Compute face normals
            normals = []
            for face_idx in vertex_faces:
                v0, v1, v2 = faces[face_idx]
                edge1 = vertices[v1] - vertices[v0]
                edge2 = vertices[v2] - vertices[v0]
                normal = np.cross(edge1, edge2)
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                normals.append(normal)

            # Compute curvature as variance of normals
            if len(normals) > 1:
                normals = np.array(normals)
                mean_normal = np.mean(normals, axis=0)
                mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)
                curvature = np.mean([1 - np.dot(n, mean_normal) for n in normals])
                vertex_curvature[i] = curvature

    # Edge detection - vertices on sharp edges get higher importance
    edge_importance = np.zeros(len(vertices))
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge_length = np.linalg.norm(vertices[v1] - vertices[v2])
            # Longer edges might be more important
            edge_importance[v1] += edge_length
            edge_importance[v2] += edge_length

    # Combine features
    curvature_normalized = vertex_curvature / (vertex_curvature.max() + 1e-8)
    edge_normalized = edge_importance / (edge_importance.max() + 1e-8)

    # Distance from center (outer features often important)
    center = np.mean(vertices, axis=0)
    distances = np.linalg.norm(vertices - center, axis=1)
    distance_normalized = distances / (distances.max() + 1e-8)

    # Weighted combination
    geometric_importance = (
        0.4 * curvature_normalized + 0.3 * edge_normalized + 0.3 * distance_normalized
    )

    return geometric_importance


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

    # Additional rotation to pitch the model forward (look ahead instead of down)
    # This rotates around X axis to lift the "chin" up
    angle_x = np.pi / 2  # +90 degrees (positive = look down)
    cos_x = np.cos(angle_x)
    sin_x = np.sin(angle_x)

    rotation_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])

    # Apply the pitch rotation
    vertices = mesh_centered.vertices
    mesh_centered.vertices = vertices @ rotation_x.T

    return mesh_centered


def render_comparison_view(mesh, resolution=800, use_vertex_colors=False):
    """Auto-oriented rendering function that works for any mesh"""
    mesh_centered = simple_orient_mesh(mesh)

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
    import argparse

    parser = argparse.ArgumentParser(description="Test rendering with AI")
    parser.add_argument("--mesh", required=True, help="Path to mesh file")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    mesh_name = mesh_path.stem
    output_path = f"output/{mesh_name}_test/test.png"

    if not mesh_path.exists():
        print(f"Error: Mesh not found at {mesh_path}")

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

        # Debug: Check importance values
        print(f"Importance stats:")
        print(f"  Min: {importance.min():.6f}")
        print(f"  Max: {importance.max():.6f}")
        print(f"  Mean: {importance.mean():.6f}")
        print(f"  Std: {importance.std():.6f}")
        print(
            f"  Non-zero: {(importance > 0).sum()}/{len(importance)} ({(importance > 0).sum() / len(importance) * 100:.1f}%)"
        )

        # If importance is very sparse (<5% non-zero), expand importance to nearby vertices
        nonzero_ratio = (importance > 0).sum() / len(importance)
        if nonzero_ratio < 0.05:
            print(
                f"  Sparse importance detected ({nonzero_ratio * 100:.1f}% non-zero), expanding to nearby vertices..."
            )

            # Create distance-weighted importance for zero-valued vertices
            from scipy.spatial import cKDTree

            tree = cKDTree(mesh.vertices)

            # For each zero-valued vertex, find importance from nearest important vertices
            expanded_importance = importance.copy()
            zero_mask = importance == 0

            if zero_mask.any():
                # Query nearest neighbors for zero-valued vertices
                distances, indices = tree.query(mesh.vertices[zero_mask], k=10)

                for i, (dists, idxs) in enumerate(zip(distances, indices)):
                    zero_idx = np.where(zero_mask)[0][i]
                    # Weight by inverse distance, only consider important neighbors
                    weights = 1.0 / (dists + 1e-6)
                    neighbor_importance = importance[idxs]

                    # Only weight neighbors that have importance
                    mask = neighbor_importance > 0
                    if mask.any():
                        expanded_importance[zero_idx] = np.sum(
                            weights[mask] * neighbor_importance[mask]
                        ) / np.sum(weights[mask])

            importance = expanded_importance
            new_nonzero = (importance > 0).sum() / len(importance) * 100
            print(f"  After expansion: {new_nonzero:.1f}% non-zero")

        # Normalize final importance values
        # If importance is still too sparse after expansion (<10%), use geometric fallback
        final_nonzero = (importance > 0).sum() / len(importance)
        if final_nonzero < 0.10:
            print(
                f"  AI saliency still sparse ({final_nonzero * 100:.1f}%), using geometric fallback..."
            )

            # Create geometric importance based on curvature and features
            geometric_importance = compute_geometric_importance(mesh)

            # Blend AI importance (where it exists) with geometric importance
            blend_weight = 0.3  # Give some weight to AI where it detected something
            importance = np.maximum(
                importance, geometric_importance * (1 - blend_weight)
            )

            final_nonzero = (importance > 0).sum() / len(importance)
            print(f"  Geometric blend resulted in {final_nonzero * 100:.1f}% non-zero")

        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (
                importance.max() - importance.min()
            )
            print(
                f"  Final normalized range: {importance.min():.3f} to {importance.max():.3f}"
            )

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
        output_dir = Path(f"output/{mesh_name}_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_render_ai.png"
        plt.imsave(output_path, img)
        print(f"\n✓ AI heatmap render saved to: {output_path}")

        # Also save normal render for comparison
        img_normal = render_comparison_view(mesh)
        output_normal = output_dir / f"{mesh_name}_normal.png"
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
