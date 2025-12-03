import time
import json
from pathlib import Path

from core.preprocessing import load_mesh
from core.rendering import OffscreenRenderer
from core.ai_importance import SaliencyExtractor, project_importance_to_vertices
from core.qem_simplifier import QEMSimplifier
from core.evaluation import compare_meshes, mesh_metrics


def compare_qem_vs_ai(mesh, target_ratio=0.2, alpha=1.0, model="facebook/dinov2-small"):
    """compare standard qem vs ai-modulated qem"""
    print(f"\n=== comparing qem vs ai-qem at {target_ratio:.0%} ===\n")

    # standard qem
    print("running standard qem...")
    t0 = time.time()
    qem_simplifier = QEMSimplifier(mesh, importance=None, alpha=0)
    mesh_qem = qem_simplifier.simplify(target_ratio=target_ratio)
    time_qem = time.time() - t0
    print(f"  completed in {time_qem:.2f}s")

    # ai-modulated qem
    print("\nrunning ai-modulated qem...")
    t0 = time.time()

    renderer = OffscreenRenderer(resolution=256)
    views = renderer.render_views(mesh, num_views=6)
    extractor = SaliencyExtractor(model_name=model)
    saliency_maps = extractor.extract_multi_view_saliency(views)
    importance = project_importance_to_vertices(mesh, views, saliency_maps)
    renderer.cleanup()

    ai_simplifier = QEMSimplifier(mesh, importance=importance, alpha=alpha)
    mesh_ai = ai_simplifier.simplify(target_ratio=target_ratio)
    time_ai = time.time() - t0
    print(f"  completed in {time_ai:.2f}s")

    # compare
    print("\n=== evaluation ===\n")
    results_qem = compare_meshes(mesh, mesh_qem, label="standard qem")
    print()
    results_ai = compare_meshes(mesh, mesh_ai, label="ai-modulated")

    return {
        "target_ratio": target_ratio,
        "alpha": alpha,
        "model": model,
        "time_qem": time_qem,
        "time_ai": time_ai,
        "qem": results_qem,
        "ai": results_ai,
    }


def run_benchmark(mesh_path, output_dir, ratios=None, alpha=1.0):
    """run full benchmark on a mesh"""
    if ratios is None:
        ratios = [0.5, 0.2, 0.05]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"benchmark: {mesh_path}")
    print(f"{'='*60}")

    mesh = load_mesh(mesh_path)
    print(f"\nloaded: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    results = {
        "mesh": str(mesh_path),
        "original": mesh_metrics(mesh),
        "alpha": alpha,
        "comparisons": [],
    }

    for ratio in ratios:
        comparison = compare_qem_vs_ai(mesh, target_ratio=ratio, alpha=alpha)
        results["comparisons"].append(comparison)

    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ benchmark complete, saved to {results_path}")

    return results
