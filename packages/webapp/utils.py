import tempfile
import subprocess
import time
from pathlib import Path
import shutil
import trimesh
import numpy as np


class SimplificationPipeline:
    """wrapper for running simplification pipeline via CLI subprocess"""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ailod_"))
        self.cache = {}

    def run(self, mesh_path, ratios, alpha=1.0, use_ai=True):
        """run simplification pipeline by calling CLI"""
        cache_key = f"{mesh_path}_{ratios}_{alpha}_{use_ai}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        print(f"running pipeline via CLI: alpha={alpha}, use_ai={use_ai}")
        start_time = time.time()

        # Build CLI command
        output_dir = self.temp_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Build ratios string (comma-separated)
        ratios_str = ",".join(str(r) for r in ratios)

        cmd = [
            "uv", "run", "ai-lod",
            "-i", mesh_path,
            "-o", str(output_dir),
            "-a", str(alpha),
            "-r", ratios_str,
            "--export-heatmap",  # Always export heatmap for visualization
        ]

        # Add AI flag
        if not use_ai:
            cmd.append("--no-ai")

        # Run CLI as subprocess with real-time output
        print(f"Running: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Collect output
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            print(line.strip())  # Print to streamlit console

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"CLI failed: {''.join(output_lines)}")

        # Load results from output directory
        lods = []
        lod_paths = []

        # Load all generated LOD files
        lod_files = sorted(output_dir.glob("lod*.obj"))
        for lod_file in lod_files:
            mesh = trimesh.load(str(lod_file))
            lods.append(mesh)
            lod_paths.append(lod_file)

        # Load heatmap if exists
        heatmap_mesh = None
        heatmap_path = output_dir / "importance_heatmap.obj"
        if heatmap_path.exists():
            heatmap_mesh = trimesh.load(str(heatmap_path))

        # Load importance data if exists
        importance = None
        importance_path = output_dir / "importance.npy"
        if importance_path.exists():
            importance = np.load(str(importance_path))

        # Create basic comparisons (face counts)
        comparisons = []
        if len(lods) > 1:
            original = lods[0]
            for i, lod in enumerate(lods[1:], 1):
                comp = {
                    "simplified": {
                        "num_vertices": len(lod.vertices),
                        "num_faces": len(lod.faces),
                    },
                    "face_ratio": len(lod.faces) / len(original.faces),
                    # CLI doesn't compute this yet
                    "hausdorff": {"hausdorff": 0.0, "rms": 0.0},
                }
                comparisons.append(comp)

        elapsed = time.time() - start_time

        result = {
            "mesh": lods[0] if lods else None,
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
