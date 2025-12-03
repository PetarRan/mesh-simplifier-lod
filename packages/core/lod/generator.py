import numpy as np
from pathlib import Path
from copy import deepcopy

from qem_simplifier import QEMSimplifier


def generate_lods(
    mesh, importance=None, target_ratios=None, alpha=1.0, output_dir=None
):
    """
    generate multiple lod levels for a mesh

    mesh: trimesh object (lod0 - original)
    importance: per-vertex importance array, or None
    target_ratios: list of target face ratios (e.g. [0.5, 0.2, 0.05])
    alpha: importance modulation weight
    output_dir: optional path to save lod meshes

    returns: list of meshes [lod0, lod1, lod2, ...]
    """
    if target_ratios is None:
        target_ratios = [0.5, 0.2, 0.05]

    lods = []

    # lod0 is original
    lod0 = deepcopy(mesh)
    lods.append(lod0)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        lod0.export(output_dir / "lod0.obj")
        print(f"saved lod0: {len(lod0.faces)} faces -> {output_dir / 'lod0.obj'}")

    # generate progressive lods
    # each lod is simplified from the original (not cascaded)
    for i, ratio in enumerate(target_ratios):
        print(f"\ngenerating lod{i+1} (target ratio: {ratio})...")

        # simplify from original mesh
        simplifier = QEMSimplifier(mesh, importance=importance, alpha=alpha)
        lod = simplifier.simplify(target_ratio=ratio)

        lods.append(lod)

        if output_dir:
            lod_path = output_dir / f"lod{i+1}.obj"
            lod.export(lod_path)
            print(f"saved lod{i+1}: {len(lod.faces)} faces -> {lod_path}")

    return lods


def compute_lod_metrics(lods):
    """
    compute metrics for generated lods
    returns dict with stats
    """
    metrics = []

    for i, lod in enumerate(lods):
        metric = {
            "level": i,
            "num_vertices": len(lod.vertices),
            "num_faces": len(lod.faces),
            "bounds": lod.bounds,
            "volume": lod.volume if lod.is_watertight else None,
        }

        # reduction ratio from lod0
        if i > 0:
            metric["face_reduction"] = len(lod.faces) / len(lods[0].faces)
            metric["vertex_reduction"] = len(lod.vertices) / len(lods[0].vertices)

        metrics.append(metric)

    return metrics
