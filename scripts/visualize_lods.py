#!/usr/bin/env python3
"""visualize lod levels side-by-side with technical stats"""
import trimesh
import pyrender
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def render_mesh(mesh, resolution=800, importance=None):
    """render a single mesh view with optional importance heatmap"""
    # center mesh
    mesh_centered = mesh.copy()
    mesh_centered.vertices -= mesh.centroid

    # setup scene with importance coloring if available
    if importance is not None:
        # create color map from importance values
        cmap = plt.colormaps.get_cmap('plasma')
        colors = cmap(importance)[:, :3]  # RGB only
        mesh_centered.visual.vertex_colors = (colors * 255).astype(np.uint8)
        mesh_pr = pyrender.Mesh.from_trimesh(mesh_centered, smooth=True)
    else:
        mesh_pr = pyrender.Mesh.from_trimesh(mesh_centered, smooth=True)

    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
    scene.add(mesh_pr)

    # lighting - more dramatic
    light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    light_pose1 = np.eye(4)
    light_pose1[:3, 3] = [3, 5, 3]
    scene.add(light1, pose=light_pose1)

    # second light for better definition
    light2 = pyrender.DirectionalLight(color=[0.8, 0.8, 1.0], intensity=2.0)
    light_pose2 = np.eye(4)
    light_pose2[:3, 3] = [-2, 2, 1]
    scene.add(light2, pose=light_pose2)

    # camera - better angle, closer zoom
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)  # narrower FOV = bigger bunny

    # rotation around Y axis (25 degrees)
    angle = np.pi / 7.2
    rotation_y = np.array([
        [np.cos(angle), 0, -np.sin(angle), 0],
        [0, 1, 0, 0],
        [np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

    # base pose (camera closer = bigger mesh)
    base_pose = np.eye(4)
    base_pose[2, 3] = 0.7  # closer camera

    cam_pose = rotation_y @ base_pose
    scene.add(camera, pose=cam_pose)

    # render
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    color, _ = renderer.render(scene)
    renderer.delete()

    return color

def add_technical_overlay(image, stats):
    """add technical stats overlay"""
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)

    # fonts
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        stats_font = ImageFont.truetype("/System/Library/Fonts/Courier New.ttc", 18)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        title_font = stats_font = small_font = ImageFont.load_default()

    # semi-transparent background for text
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # top banner
    banner_height = 70
    overlay_draw.rectangle([(0, 0), (img.width, banner_height)], fill=(0, 0, 0, 200))

    # bottom stats panel
    panel_height = 160
    overlay_draw.rectangle([(0, img.height - panel_height), (img.width, img.height)],
                          fill=(0, 0, 0, 200))

    # composite overlay
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)

    # title at top
    title = f"LOD{stats['level']}"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = bbox[2] - bbox[0]
    draw.text(((img.width - title_width) // 2, 20), title, font=title_font, fill=(0, 255, 255))

    # stats at bottom
    y = img.height - panel_height + 15
    line_spacing = 22

    stats_lines = [
        f"Faces:      {stats['faces']:>8,}",
        f"Vertices:   {stats['vertices']:>8,}",
        f"Reduction:  {stats['reduction']:>7.2f}%",
        f"Tri/Vert:   {stats['tri_per_vert']:>8.2f}",
        f"Edge Count: {stats['edges']:>8,}",
    ]

    for i, line in enumerate(stats_lines):
        draw.text((20, y + i * line_spacing), line, font=stats_font, fill=(0, 255, 100))

    # add percentage badge
    if stats['level'] > 0:
        badge_text = f"{stats['percentage']:.0f}%"
        badge_bbox = draw.textbbox((0, 0), badge_text, font=title_font)
        badge_width = badge_bbox[2] - badge_bbox[0]
        badge_x = img.width - badge_width - 30
        draw.text((badge_x, 20), badge_text, font=title_font, fill=(255, 100, 100))

    return np.array(img)

def create_colorbar(height=800, width=60):
    """create a vertical colorbar for importance visualization"""
    colorbar = np.zeros((height, width, 3), dtype=np.uint8)
    cmap = plt.colormaps.get_cmap('plasma')

    for i in range(height):
        value = 1.0 - (i / height)  # top = high importance
        color = np.array(cmap(value)[:3]) * 255
        color = color.astype(np.uint8)
        colorbar[i, :] = color

    img = Image.fromarray(colorbar)
    draw = ImageDraw.Draw(img)

    # add labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()

    draw.text((5, 10), "HIGH", font=font, fill=(255, 255, 255))
    draw.text((10, height - 25), "LOW", font=font, fill=(255, 255, 255))

    return np.array(img)

def main():
    output_dir = "test_meshes/output"

    print("🔍 analyzing LOD meshes...")

    # load LOD meshes
    lods = []
    for i in range(4):
        lod_path = f"{output_dir}/lod{i}.obj"
        if not os.path.exists(lod_path):
            continue

        mesh = trimesh.load(lod_path)
        lods.append((i, mesh))
        print(f"  lod{i}: {len(mesh.faces):,} faces, {len(mesh.vertices):,} vertices")

    if not lods:
        print("❌ no LOD files found")
        return

    original_mesh = lods[0][1]
    resolution = 800

    # === RENDER STANDARD LOD COMPARISON ===
    print("\n🎨 rendering standard LOD comparison...")
    renders = []

    for i, mesh in lods:
        print(f"  rendering lod{i}...")
        img = render_mesh(mesh, resolution)

        # calculate stats
        stats = {
            'level': i,
            'faces': len(mesh.faces),
            'vertices': len(mesh.vertices),
            'edges': len(mesh.edges_unique),
            'tri_per_vert': len(mesh.faces) / len(mesh.vertices),
            'reduction': (1 - len(mesh.faces) / len(original_mesh.faces)) * 100,
            'percentage': (len(mesh.faces) / len(original_mesh.faces)) * 100
        }

        img_labeled = add_technical_overlay(img, stats)
        renders.append(img_labeled)

    # combine horizontally
    total_width = resolution * len(renders)
    combined = np.zeros((resolution, total_width, 3), dtype=np.uint8)

    for i, render in enumerate(renders):
        combined[:, i * resolution:(i + 1) * resolution] = render

    output_path = f"{output_dir}/lod_comparison.png"
    Image.fromarray(combined).save(output_path)
    print(f"✅ saved: {output_path}")

    # === RENDER IMPORTANCE HEATMAP IF AVAILABLE ===
    importance_path = f"{output_dir}/importance.npy"
    if os.path.exists(importance_path):
        print("\n🔥 rendering importance heatmap visualization...")
        importance = np.load(importance_path)

        # render with heatmap coloring
        heatmap_renders = []
        for i, mesh in lods:  # all LODs with heatmap
            print(f"  rendering lod{i} heatmap...")

            # map importance to current mesh vertices (simple nearest neighbor)
            if i == 0:
                mesh_importance = importance
            else:
                # for simplified meshes, map original importance to new vertices
                from scipy.spatial import cKDTree
                tree = cKDTree(original_mesh.vertices)
                _, indices = tree.query(mesh.vertices)
                mesh_importance = importance[indices]

            img = render_mesh(mesh, resolution, importance=mesh_importance)

            stats = {
                'level': i,
                'faces': len(mesh.faces),
                'vertices': len(mesh.vertices),
                'edges': len(mesh.edges_unique),
                'tri_per_vert': len(mesh.faces) / len(mesh.vertices),
                'reduction': (1 - len(mesh.faces) / len(original_mesh.faces)) * 100,
                'percentage': (len(mesh.faces) / len(original_mesh.faces)) * 100
            }

            img_labeled = add_technical_overlay(img, stats)
            heatmap_renders.append(img_labeled)

        # add colorbar
        colorbar = create_colorbar(resolution, 80)

        # combine with colorbar
        total_width = resolution * len(heatmap_renders) + 80
        combined_heatmap = np.ones((resolution, total_width, 3), dtype=np.uint8) * 40

        for i, render in enumerate(heatmap_renders):
            combined_heatmap[:, i * resolution:(i + 1) * resolution] = render

        # add colorbar on right
        combined_heatmap[:, -80:] = colorbar

        heatmap_path = f"{output_dir}/lod_importance_heatmap.png"
        Image.fromarray(combined_heatmap).save(heatmap_path)
        print(f"✅ saved: {heatmap_path}")

        # importance stats
        print(f"\n📊 importance statistics:")
        print(f"  range: [{importance.min():.3f}, {importance.max():.3f}]")
        print(f"  mean:  {importance.mean():.3f}")
        print(f"  std:   {importance.std():.3f}")
        coverage = (importance > 0).sum() / len(importance) * 100
        print(f"  vertex coverage: {coverage:.1f}%")
    else:
        print(f"\n⚠️  no importance data found at {importance_path}")
        print("   run without --no-ai to generate importance heatmaps")

    # save individual renders
    print(f"\n💾 saving individual renders...")
    for i, (lod_idx, _) in enumerate(lods):
        individual_path = f"{output_dir}/lod{lod_idx}_render.png"
        Image.fromarray(renders[i]).save(individual_path)
        print(f"  {individual_path}")

    print(f"\n✨ visualization complete!")

if __name__ == "__main__":
    main()
