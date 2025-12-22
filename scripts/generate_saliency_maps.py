#!/usr/bin/env python3
"""
Generate 2D saliency maps with 3D mesh overlay for thesis assets.
Shows saliency heatmaps from multiple viewing angles.
"""

from preprocessing import load_mesh
from rendering import OffscreenRenderer
from ai_importance import SaliencyExtractor
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors
from PIL import Image

matplotlib.use("Agg")

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "core"))


def create_saliency_visualization(mesh_path, output_path):
    """Create 2x3 grid of saliency maps with mesh overlay"""
    print("Loading mesh...")
    mesh = load_mesh(mesh_path)

    print("Rendering views...")

    # Create custom renderer with much closer camera
    from copy import deepcopy
    import pyrender

    renderer = OffscreenRenderer(resolution=512)

    # Direct custom rendering for close-up shots
    mesh_centered = deepcopy(mesh)
    original_centroid = mesh.centroid
    mesh_centered.vertices -= original_centroid

    mesh_pr = pyrender.Mesh.from_trimesh(mesh_centered, smooth=True)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    scene.add(mesh_pr)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [2, 5, 2]
    scene.add(light, pose=light_pose)

    # Much closer camera - this should make bunny much larger
    distance = 0.25  # Extremely close!
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    views = []
    for i in range(6):
        angle = 2 * np.pi * i / 6
        elevation = np.sin(angle * 2) * 0.3

        rotation_y = np.array(
            [
                [np.cos(angle), 0, -np.sin(angle), 0],
                [0, 1, 0, 0],
                [np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1],
            ]
        )

        rotation_x = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(elevation), -np.sin(elevation), 0],
                [0, np.sin(elevation), np.cos(elevation), 0],
                [0, 0, 0, 1],
            ]
        )

        base_pose = np.eye(4)
        base_pose[2, 3] = distance

        cam_pose = rotation_y @ rotation_x @ base_pose
        cam_node = scene.add(camera, pose=cam_pose)

        color, depth = renderer.renderer.render(scene)

        pose_original = cam_pose.copy()
        pose_original[:3, 3] += original_centroid

        views.append({"image": color, "depth": depth, "pose": pose_original})
        scene.remove_node(cam_node)

    renderer.cleanup()

    print("Extracting saliency...")
    extractor = SaliencyExtractor(model_name="facebook/dinov2-small")
    saliency_maps = extractor.extract_multi_view_saliency(views)

    print("Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "2D Vizuelna Značajnost (Saliency) iz Svakog Ugla", fontsize=16, fontweight="bold"
    )

    angle_labels = ["Desna", "Prednja", "Leva", "Zadnja", "Donja", "Gornja"]

    for i, (ax, view, saliency, label) in enumerate(
        zip(axes.flat, views, saliency_maps, angle_labels)
    ):
        # Show RGB view
        ax.imshow(view["image"], alpha=0.7)

        # Overlay saliency heatmap
        saliency_resized = np.array(
            Image.fromarray((saliency * 255).astype(np.uint8)).resize(
                view["image"].shape[:2][::-1]
            )
        )
        cmap = cm.get_cmap("hot")
        colored_saliency = cmap(saliency_resized / 255.0)
        ax.imshow(colored_saliency, alpha=0.5, cmap="hot")

        ax.set_title(f"{label} Strana", fontsize=12, fontweight="bold")
        ax.axis("off")

    # Add colorbar
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    sm = plt.cm.ScalarMappable(
        cmap="hot", norm=matplotlib.colors.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Saliency Intenzitet", rotation=270,
                   labelpad=20, fontsize=12)

    plt.tight_layout(rect=(0, 0, 0.9, 0.96))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Saved saliency visualization to: {output_path}")


def main():
    mesh_path = "test_meshes/bunny/reconstruction/bun_zipper.ply"
    output_path = "test_meshes/compare/saliency_maps.png"

    if not Path(mesh_path).exists():
        print(f"Mesh not found: {mesh_path}")
        return

    from PIL import Image

    create_saliency_visualization(mesh_path, output_path)


if __name__ == "__main__":
    main()
