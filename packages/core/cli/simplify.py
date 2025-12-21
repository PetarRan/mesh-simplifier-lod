#!/usr/bin/env python3
"""cli tool for ai-lod mesh simplification"""

import argparse
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "packages"))
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent / "packages" / "benchmarks")
)

from core.preprocessing import load_mesh
from core.rendering import OffscreenRenderer
from core.ai_importance import SaliencyExtractor, project_importance_to_vertices
from core.lod import generate_lods, compute_lod_metrics


def main():
    parser = argparse.ArgumentParser(description="ai-modulated lod generation")

    parser.add_argument(
        "-i", "--input", required=True, help="input mesh (obj/ply/gltf)"
    )

    parser.add_argument("-o", "--out", required=True, help="output directory")

    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=1.0,
        help="importance weight (default: 1.0)",
    )

    parser.add_argument(
        "-r",
        "--ratios",
        default="0.5,0.2,0.05",
        help="lod ratios (default: 0.5,0.2,0.05)",
    )

    parser.add_argument(
        "-m", "--model", default="facebook/dinov2-small", help="saliency model"
    )

    parser.add_argument(
        "-v", "--views", type=int, default=6, help="render views (default: 6)"
    )

    parser.add_argument(
        "--resolution", type=int, default=256, help="render resolution (default: 256)"
    )

    parser.add_argument(
        "--no-ai", action="store_true", help="disable ai, use standard qem"
    )

    parser.add_argument("--compare", action="store_true", help="compare qem vs ai-qem")

    parser.add_argument(
        "--export-heatmap", action="store_true", help="export importance heatmap"
    )

    parser.add_argument("--benchmark", action="store_true", help="run full benchmark")

    args = parser.parse_args()

    # benchmark mode
    if args.benchmark:
        from benchmarks import run_benchmark

        ratios = [float(r) for r in args.ratios.split(",")]
        run_benchmark(args.input, args.out, ratios=ratios, alpha=args.alpha)
        return

    ratios = [float(r) for r in args.ratios.split(",")]

    print("=== ai-lod mesh simplification ===")
    print(f"input: {args.input}")
    print(f"output: {args.out}")
    print(f"ratios: {ratios}")
    print(f"alpha: {args.alpha}")
    print(f"mode: {'standard qem' if args.no_ai else 'ai-modulated'}\n")

    # load mesh
    print("loading mesh...")
    mesh = load_mesh(args.input)
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # ai importance pipeline
    importance = None

    if not args.no_ai:
        print("\n=== ai importance ===")

        # render
        print(f"rendering {args.views} views at {args.resolution}px...")
        renderer = OffscreenRenderer(resolution=args.resolution)
        views = renderer.render_views(mesh, num_views=args.views)
        print(f"  rendered {len(views)} views")

        # saliency
        print(f"loading model: {args.model}")
        extractor = SaliencyExtractor(model_name=args.model)
        print("extracting saliency...")
        saliency_maps = extractor.extract_multi_view_saliency(views)
        print(f"  extracted {len(saliency_maps)} maps")

        # project
        print("projecting to vertices...")
        importance = project_importance_to_vertices(mesh, views, saliency_maps)
        print(f"  range: [{importance.min():.3f}, {importance.max():.3f}]")

        renderer.cleanup()

        # save importance data for visualization
        import numpy as np

        Path(args.out).mkdir(parents=True, exist_ok=True)
        importance_path = Path(args.out) / "importance.npy"
        np.save(importance_path, importance)
        print(f"  saved importance data: {importance_path}")

        # export heatmap if requested
        if args.export_heatmap:
            from core.visualization import export_heatmap

            print("\n=== exporting heatmap ===")
            heatmap_path = Path(args.out) / "importance_heatmap"
            export_heatmap(mesh, importance, heatmap_path)

    # compare mode
    if args.compare:
        from benchmarks.runner import compare_qem_vs_ai

        result = compare_qem_vs_ai(
            mesh, target_ratio=ratios[0], alpha=args.alpha, model=args.model
        )

        # save comparison renders
        from core.visualization import render_comparison
        from core.qem_simplifier import QEMSimplifier

        mesh_qem = QEMSimplifier(mesh, importance=None).simplify(ratios[0])
        mesh_ai = QEMSimplifier(mesh, importance=importance, alpha=args.alpha).simplify(
            ratios[0]
        )

        comp_path = Path(args.out) / "comparison.png"
        render_comparison(
            [mesh, mesh_qem, mesh_ai], ["original", "qem", "ai-qem"], comp_path
        )

        return

    # generate lods
    print("\n=== generating lods ===")
    lods = generate_lods(
        mesh,
        importance=importance,
        target_ratios=ratios,
        alpha=args.alpha,
        output_dir=args.out,
    )

    # metrics
    print("\n=== metrics ===")
    metrics = compute_lod_metrics(lods)

    for m in metrics:
        if m["level"] == 0:
            print(f"lod{m['level']}: {m['num_faces']:6d} faces (original)")
        else:
            print(
                f"lod{m['level']}: {m['num_faces']:6d} faces ({m['face_reduction']:.1%} of original)"
            )

    print(f"\n✓ saved to {args.out}")


if __name__ == "__main__":
    main()
