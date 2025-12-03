# ai-guided lod mesh simplifier

a small, focused system for mesh simplification that mixes classical qem with a lightweight, local-importance model. the aim is straightforward: retain the shapes people notice, discard everything else.

<!-- [placeholder: pipeline diagram goes here] -->

## overview

the project builds a minimal but extensible path for perceptually-aware lod generation. it starts from the reliable baseline—qem edge collapses—then biases the collapse order with per-patch importance estimates. the model is intentionally small enough to run on an m1 laptop without drama.

<!-- [placeholder: before/after comparison] -->

## features

- clean qem implementation with pluggable cost terms
- optional ai importance weighting via dinov2/clip/vit
- multi-view saliency projection to vertices
- offscreen rendering (pyrender, m1 compatible)
- obj/ply/gltf support via trimesh
- simple python api and cli
- room to grow: better models, different collapse heuristics, web tools later

<!-- [placeholder: model diagram] -->

## how it works

1. load mesh (obj/ply/gltf)
2. compute curvature, normals, edge sharpness
3. render orbit views (6 by default, adjustable)
4. extract saliency maps from lightweight vision model
5. project 2d saliency to 3d vertices via depth-tested raycasting
6. qem collapse with modulated cost: `qem_cost × (1 + α × importance)`
7. generate progressive lods at target ratios
8. export meshes + metrics

<!-- [placeholder: patch extraction illustration] -->

## installation

using uv (recommended):

```bash
# install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# sync dependencies (creates .venv automatically)
uv sync

# run directly
uv run ai-lod --help
```

or activate the venv:

```bash
source .venv/bin/activate
python -m ai_lod.cli.simplify --help
```

## basic usage

standard qem (no ai):

```bash
uv run ai-lod -i mesh.obj -o output/ --no-ai
```

ai-modulated simplification:

```bash
uv run ai-lod -i mesh.obj -o output/ --alpha 1.0
```

custom lod ratios:

```bash
uv run ai-lod -i mesh.obj -o output/ --ratios 0.7,0.3,0.1
```

different saliency model:

```bash
uv run ai-lod -i mesh.obj -o output/ --model openai/clip-vit-base-patch32
```

all options:

```bash
uv run ai-lod --help
```

## cli options

| flag           | default               | description                       |
| -------------- | --------------------- | --------------------------------- |
| `-i, --input`  | required              | input mesh file (obj/ply/gltf)    |
| `-o, --out`    | required              | output directory for lod meshes   |
| `-a, --alpha`  | 1.0                   | importance modulation weight      |
| `-r, --ratios` | 0.5,0.2,0.05          | comma-separated lod target ratios |
| `-m, --model`  | facebook/dinov2-small | saliency model (dinov2/clip/vit)  |
| `-v, --views`  | 6                     | number of render views            |
| `--resolution` | 256                   | render resolution (px)            |
| `--no-ai`      | false                 | disable ai, use standard qem only |

## what's uv?

[uv](https://github.com/astral-sh/uv) is a blazing fast python package manager written in rust. it's like pip + venv + poetry combined, but way faster.

quick comparison:

- `pip install` → `uv sync` (10-100x faster)
- `pip install package` → `uv add package`
- `python -m venv` → automatic (uv creates .venv for you)

## python api

```python
from ai_lod.preprocessing import load_mesh, compute_features
from ai_lod.rendering import OffscreenRenderer
from ai_lod.ai_importance import SaliencyExtractor, project_importance_to_vertices
from ai_lod.lod import generate_lods

# load mesh
mesh = load_mesh("mesh.obj")
features = compute_features(mesh)

# render views
renderer = OffscreenRenderer(resolution=256)
views = renderer.render_views(mesh, num_views=6)

# extract saliency
extractor = SaliencyExtractor(model_name="facebook/dinov2-small")
saliency_maps = extractor.extract_multi_view_saliency(views)

# project to vertices
importance = project_importance_to_vertices(mesh, views, saliency_maps)

# generate lods
lods = generate_lods(
    mesh,
    importance=importance,
    target_ratios=[0.5, 0.2, 0.05],
    alpha=1.0,
    output_dir="output/"
)

# cleanup
renderer.cleanup()
```

<!-- [placeholder: training curves] -->

## notes

- runs on macbook air (cpu/mps, no cuda required)
- dinov2-small is ~20mb, very fast
- importance values are normalized [0,1]
- higher α = more aggressive preservation of salient regions
- render resolution can be lowered for speed (256px default)
- depth testing ensures only visible vertices get importance

## status

initial goal is a dependable reference implementation that is simple enough to read in one sitting yet flexible enough to experiment with.

## license

mit — see [LICENSE](LICENSE)
