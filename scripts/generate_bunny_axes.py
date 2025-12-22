#!/usr/bin/env python3
"""
Generate Stanford Bunny with clear axis visualization for thesis.
Shows X (red), Y (green), Z (blue) axes with the bunny model.
"""

from preprocessing import load_mesh
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib

matplotlib.use("Agg")

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core"))


def create_axis_visualization(mesh_path, output_path):
    """Create visualization of bunny with clear axis indicators"""
    print("Loading mesh...")
    mesh = load_mesh(mesh_path)

    # Center and rotate mesh
    mesh_centered = mesh.copy()
    mesh_centered.vertices -= mesh_centered.centroid

    # Rotate bunny to stand upright
    vertices_rotated = mesh_centered.vertices.copy()
    temp = vertices_rotated[:, 1].copy()
    vertices_rotated[:, 1] = vertices_rotated[:, 2]
    vertices_rotated[:, 2] = temp
    mesh_centered.vertices = vertices_rotated

    print("Creating visualization...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Render mesh
    vertices = mesh_centered.vertices
    faces = mesh_centered.faces

    poly3d = Poly3DCollection(
        vertices[faces],
        facecolors=(0.8, 0.8, 0.8),
        edgecolors=(0.4, 0.4, 0.4),
        linewidths=0.1,
        alpha=0.9,
    )
    ax.add_collection3d(poly3d)

    # Calculate axis length based on mesh bounds
    bounds = mesh_centered.bounds
    extent = bounds[1] - bounds[0]
    axis_length = np.max(extent) * 0.4
    center = (bounds[0] + bounds[1]) / 2

    # Create axes
    origin = center - [0, 0, axis_length * 0.3]  # Move origin down slightly

    # X-axis (red)
    x_end = origin + [axis_length, 0, 0]
    ax.plot3D(
        [origin[0], x_end[0]],
        [origin[1], x_end[1]],
        [origin[2], x_end[2]],
        "r-",
        linewidth=3,
        label="X-axis",
    )
    ax.text(
        x_end[0], x_end[1], x_end[2], "X", fontsize=14, fontweight="bold", color="red"
    )

    # Y-axis (green)
    y_end = origin + [0, axis_length, 0]
    ax.plot3D(
        [origin[0], y_end[0]],
        [origin[1], y_end[1]],
        [origin[2], y_end[2]],
        "g-",
        linewidth=3,
        label="Y-axis",
    )
    ax.text(
        y_end[0], y_end[1], y_end[2], "Y", fontsize=14, fontweight="bold", color="green"
    )

    # Z-axis (blue)
    z_end = origin + [0, 0, axis_length]
    ax.plot3D(
        [origin[0], z_end[0]],
        [origin[1], z_end[1]],
        [origin[2], z_end[2]],
        "b-",
        linewidth=3,
        label="Z-axis",
    )
    ax.text(
        z_end[0], z_end[1], z_end[2], "Z", fontsize=14, fontweight="bold", color="blue"
    )

    # Arrows are already created by plot3D lines with proper styling

    # Set viewing angle
    ax.view_init(elev=15, azim=45)

    # Set axis limits
    max_extent = np.max(extent) * 0.7
    ax.set_xlim(center[0] - max_extent, center[0] + max_extent)
    ax.set_ylim(center[1] - max_extent, center[1] + max_extent)
    ax.set_zlim(center[2] - max_extent * 0.5, center[2] + max_extent * 1.2)

    # Labels and styling
    ax.set_xlabel("X", fontsize=12, fontweight="bold", color="red")
    ax.set_ylabel("Y", fontsize=12, fontweight="bold", color="green")
    ax.set_zlabel("Z", fontsize=12, fontweight="bold", color="blue")

    ax.set_title(
        "Stanford Bunny u Koordinatnom Sistemu", fontsize=16, fontweight="bold", pad=20
    )

    # Add grid for better depth perception
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc="upper left", fontsize=10)

    # Set background
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved axis visualization to: {output_path}")


def main():
    mesh_path = "test_meshes/bunny/reconstruction/bun_zipper.ply"
    output_path = "test_meshes/compare/bunny_with_axes.png"

    if not Path(mesh_path).exists():
        print(f"Mesh not found: {mesh_path}")
        return

    create_axis_visualization(mesh_path, output_path)


if __name__ == "__main__":
    main()
