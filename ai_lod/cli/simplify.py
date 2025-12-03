#!/usr/bin/env python3
"""cli tool for ai-lod mesh simplification"""

import argparse
from pathlib import Path

from ..preprocessing import load_mesh
from ..rendering import OffscreenRenderer
from ..ai_importance import SaliencyExtractor, project_importance_to_vertices
from ..lod import generate_lods, compute_lod_metrics


def main():
    parser = argparse.ArgumentParser(description="ai-modulated lod generation")

    parser.add_argument("-i", "--input", required=True, help="input mesh (obj/ply/gltf)")
    parser.add_argument("-o", "--out", required=True, help="output directory")
    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="importance weight (default: 1.0)")
    parser.add_argument("-r", "--ratios", default="0.5,0.2,0.05", help="lod ratios (default: 0.5,0.2,0.05)")
    parser.add_argument("-m", "--model", default="facebook/dinov2-small", help="saliency model")
    parser.add_argument("-v", "--views", type=int, default=6, help="render views (default: 6)")
    parser.add_argument("--resolution", type=int, default=256, help="render resolution (default: 256)")
    parser.add_argument("--no-ai", action="store_true", help="disable ai, use standard qem")

    args = parser.parse_args()
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

    # generate lods
    print("\n=== generating lods ===")
    lods = generate_lods(mesh, importance=importance, target_ratios=ratios, alpha=args.alpha, output_dir=args.out)

    # metrics
    print("\n=== metrics ===")
    metrics = compute_lod_metrics(lods)

    for m in metrics:
        if m["level"] == 0:
            print(f"lod{m['level']}: {m['num_faces']:6d} faces (original)")
        else:
            print(f"lod{m['level']}: {m['num_faces']:6d} faces ({m['face_reduction']:.1%} of original)")

    print(f"\n✓ saved to {args.out}")


if __name__ == "__main__":
    main()
