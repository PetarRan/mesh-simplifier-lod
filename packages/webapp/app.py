#!/usr/bin/env python3
"""
streamlit web demo for ai-lod simplification
run: streamlit run ai_lod/webapp/app.py
"""

import streamlit as st
import tempfile
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import numpy as np

from utils import SimplificationPipeline

# page config
st.set_page_config(
    page_title="AI-LOD Mesh Simplifier",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# init session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = SimplificationPipeline()
if "results" not in st.session_state:
    st.session_state.results = None


def mesh_to_plotly(mesh, color="lightblue", opacity=1.0):
    """convert trimesh to plotly mesh3d"""
    vertices = mesh.vertices
    faces = mesh.faces

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=opacity,
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
    )

    return fig


def mesh_with_colors(mesh):
    """render mesh with vertex colors (for heatmap)"""
    vertices = mesh.vertices
    faces = mesh.faces

    # get vertex colors
    if hasattr(mesh.visual, "vertex_colors"):
        colors = mesh.visual.vertex_colors[:, :3] / 255.0
        # convert to vertex color for plotly
        intensities = colors.mean(axis=1)
    else:
        intensities = None

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=intensities,
                colorscale="Viridis",
                flatshading=True,
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
    )

    return fig


# sidebar controls
st.sidebar.title("🎲 AI-LOD Simplifier")
st.sidebar.markdown("upload a mesh and generate lods with ai-guided simplification")

# file upload
uploaded_file = st.sidebar.file_uploader(
    "upload mesh", type=["obj", "ply", "gltf", "glb"], help="upload obj, ply, or gltf mesh"
)

# parameters
st.sidebar.subheader("parameters")

use_ai = st.sidebar.checkbox("enable ai importance", value=True, help="use ai saliency for importance weighting")

alpha = st.sidebar.slider(
    "alpha (ai influence)",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="higher = more preservation of important regions",
    disabled=not use_ai,
)

st.sidebar.subheader("lod targets")
ratio_1 = st.sidebar.slider("lod1 reduction", 0.1, 0.9, 0.5, 0.05, format="%.0f%%", help="50% = half faces")
ratio_2 = st.sidebar.slider("lod2 reduction", 0.1, 0.9, 0.2, 0.05, format="%.0f%%")
ratio_3 = st.sidebar.slider("lod3 reduction", 0.01, 0.5, 0.05, 0.01, format="%.0f%%")

ratios = [ratio_1, ratio_2, ratio_3]

# run button
run_button = st.sidebar.button("🚀 generate lods", type="primary", disabled=uploaded_file is None)

# main area
st.title("AI-Guided LOD Mesh Simplification")
st.markdown("hybrid qem + ai importance for perceptually-aware mesh reduction")

if uploaded_file is None:
    st.info("👈 upload a mesh to get started")
    st.stop()

# save uploaded file
with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
    tmp.write(uploaded_file.read())
    mesh_path = tmp.name

# run pipeline
if run_button:
    with st.spinner("running simplification pipeline..."):
        try:
            results = st.session_state.pipeline.run(mesh_path, ratios, alpha=alpha, use_ai=use_ai)
            st.session_state.results = results
            st.success(f"✓ completed in {results['elapsed']:.1f}s")
        except Exception as e:
            st.error(f"error: {e}")
            st.stop()

# display results
if st.session_state.results is None:
    st.info("click 'generate lods' to run")
    st.stop()

results = st.session_state.results

# tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 lod comparison", "🎨 importance heatmap", "📈 metrics"])

with tab1:
    st.subheader("lod comparison")

    # show lods side by side
    cols = st.columns(4)

    labels = ["original", "lod1", "lod2", "lod3"]
    for i, (col, mesh, label) in enumerate(zip(cols, results["lods"], labels)):
        with col:
            st.markdown(f"**{label}**")
            st.markdown(f"`{len(mesh.faces)} faces`")
            fig = mesh_to_plotly(mesh, color=["lightblue", "lightgreen", "orange", "salmon"][i])
            st.plotly_chart(fig, use_container_width=True)

            # download link
            if i > 0:
                with open(results["lod_paths"][i], "rb") as f:
                    st.download_button(
                        f"⬇️ {label}.obj",
                        f,
                        file_name=f"{label}.obj",
                        mime="application/octet-stream",
                    )

with tab2:
    st.subheader("importance heatmap")

    if results["heatmap_mesh"] is not None:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig = mesh_with_colors(results["heatmap_mesh"])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**color scale**")
            st.markdown("- 🔴 red: high importance")
            st.markdown("- 🔵 blue: low importance")
            st.markdown("")
            st.markdown("ai model preserves high-importance regions during simplification")

            # download heatmap
            heatmap_path = st.session_state.pipeline.temp_dir / "heatmap.obj"
            if heatmap_path.exists():
                with open(heatmap_path, "rb") as f:
                    st.download_button(
                        "⬇️ download heatmap.obj",
                        f,
                        file_name="heatmap.obj",
                        mime="application/octet-stream",
                    )
    else:
        st.info("enable ai importance to see heatmap")

with tab3:
    st.subheader("quality metrics")

    # build metrics table
    rows = []
    for i, comp in enumerate(results["comparisons"]):
        rows.append(
            {
                "lod": f"lod{i+1}",
                "faces": comp["simplified"]["num_vertices"],
                "vertices": comp["simplified"]["num_vertices"],
                "reduction": f"{comp['face_ratio']:.1%}",
                "hausdorff": f"{comp['hausdorff']['hausdorff']:.6f}",
                "rms_error": f"{comp['hausdorff']['rms']:.6f}",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("**metrics explanation:**")
    st.markdown("- **hausdorff distance**: max geometric error (lower is better)")
    st.markdown("- **rms error**: root mean square error (lower is better)")
    st.markdown("- **reduction**: target face count percentage")

# footer
st.sidebar.markdown("---")
st.sidebar.markdown("built with [streamlit](https://streamlit.io)")
