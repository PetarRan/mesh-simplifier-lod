import tempfile
import time
from pathlib import Path
import shutil

from core.preprocessing import load_mesh
from core.rendering import OffscreenRenderer
from core.ai_importance import SaliencyExtractor, project_importance_to_vertices
from core.qem_simplifier import QEMSimplifier
from core.evaluation import compare_meshes
from core.visualization import paint_importance_heatmap


class SimplificationPipeline:
    """wrapper for running simplification pipeline with caching"""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ailod_"))
        self.cache = {}

    def run(self, mesh_path, ratios, alpha=1.0, use_ai=True):
        """run simplification pipeline"""
        cache_key = f"{mesh_path}_{ratios}_{alpha}_{use_ai}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        print(f"running pipeline: alpha={alpha}, use_ai={use_ai}")
        start_time = time.time()

        mesh = load_mesh(mesh_path)

        # ai importance
        importance = None
        if use_ai:
            renderer = OffscreenRenderer(resolution=256)
            views = renderer.render_views(mesh, num_views=6)

            extractor = SaliencyExtractor(model_name="facebook/dinov2-small")
            saliency_maps = extractor.extract_multi_view_saliency(views)
            importance = project_importance_to_vertices(mesh, views, saliency_maps)

            renderer.cleanup()

        # generate lods
        lods = []
        lod_paths = []
        comparisons = []

        lods.append(mesh)
        orig_path = self.temp_dir / "lod0.obj"
        mesh.export(str(orig_path))
        lod_paths.append(orig_path)

        for i, ratio in enumerate(ratios):
            simplifier = QEMSimplifier(mesh, importance=importance, alpha=alpha)
            lod = simplifier.simplify(target_ratio=ratio)
            lods.append(lod)

            lod_path = self.temp_dir / f"lod{i+1}.obj"
            lod.export(str(lod_path))
            lod_paths.append(lod_path)

            comp = compare_meshes(mesh, lod, label=f"lod{i+1}")
            comparisons.append(comp)

        # heatmap
        heatmap_mesh = None
        if importance is not None:
            heatmap_mesh = paint_importance_heatmap(mesh, importance)
            heatmap_path = self.temp_dir / "heatmap.obj"
            heatmap_mesh.export(str(heatmap_path))

        elapsed = time.time() - start_time

        result = {
            "mesh": mesh,
            "lods": lods,
            "lod_paths": lod_paths,
            "importance": importance,
            "heatmap_mesh": heatmap_mesh,
            "comparisons": comparisons,
            "elapsed": elapsed,
        }

        self.cache[cache_key] = result
        return result

    def cleanup(self):
        """remove temp files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
