import numpy as np
from scipy.spatial import KDTree
from pathlib import Path


def hausdorff_distance(mesh_a, mesh_b, sample_points=10000):
    """
    compute hausdorff distance between two meshes
    samples points on surfaces and computes max min-distance
    """
    # sample points on surfaces
    pts_a = mesh_a.sample(sample_points)
    pts_b = mesh_b.sample(sample_points)

    # build kdtrees
    tree_a = KDTree(pts_a)
    tree_b = KDTree(pts_b)

    # distances from a to b
    dist_a_to_b, _ = tree_b.query(pts_a)
    # distances from b to a
    dist_b_to_a, _ = tree_a.query(pts_b)

    # hausdorff is max of both directions
    hausdorff = max(dist_a_to_b.max(), dist_b_to_a.max())

    return {
        "hausdorff": hausdorff,
        "mean_a_to_b": dist_a_to_b.mean(),
        "mean_b_to_a": dist_b_to_a.mean(),
        "rms": np.sqrt(np.mean(dist_a_to_b**2 + dist_b_to_a**2) / 2),
    }


def mesh_metrics(mesh, mesh_path=None):
    """compute basic mesh metrics"""
    metrics = {
        "num_vertices": len(mesh.vertices),
        "num_faces": len(mesh.faces),
        "surface_area": mesh.area,
        "volume": mesh.volume if mesh.is_watertight else None,
        "bounds": mesh.bounds.tolist(),
        "is_watertight": mesh.is_watertight,
    }

    # file size if path provided
    if mesh_path:
        path = Path(mesh_path)
        if path.exists():
            metrics["file_size_kb"] = path.stat().st_size / 1024

    return metrics


def compare_meshes(original, simplified, label="simplified"):
    """
    compare original vs simplified mesh
    returns dict with all metrics
    """
    print(f"comparing {label} to original...")

    # geometric distance
    hausdorff = hausdorff_distance(original, simplified)
    print(f"  hausdorff: {hausdorff['hausdorff']:.6f}")
    print(f"  rms error: {hausdorff['rms']:.6f}")

    # reduction ratios
    face_ratio = len(simplified.faces) / len(original.faces)
    vert_ratio = len(simplified.vertices) / len(original.vertices)

    print(f"  face reduction: {face_ratio:.1%}")
    print(f"  vertex reduction: {vert_ratio:.1%}")

    return {
        "label": label,
        "hausdorff": hausdorff,
        "face_ratio": face_ratio,
        "vertex_ratio": vert_ratio,
        "original": mesh_metrics(original),
        "simplified": mesh_metrics(simplified),
    }
