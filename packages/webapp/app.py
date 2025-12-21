#!/usr/bin/env python3
"""
streamlit web demo for ai-lod simplification
run: streamlit run ai_lod/webapp/app.py
"""

import streamlit as st
import tempfile
from pathlib import Path

from .utils import SimplificationPipeline
from .ui_components import (
    render_lod_comparison_tab,
    render_heatmap_tab,
    render_metrics_tab,
)

# page config
st.set_page_config(
    page_title="AI-LOD Mesh Simplifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# kind of sucks but CSS for better layout
st.markdown(
    """
<style>
    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-height: 100vh;
    }

    /* Make main content area use full height */
    .main {
        height: 100vh;
        overflow-y: auto;
    }

    /* Sidebar - ultra compact */
    [data-testid="stSidebar"] {
        max-height: 100vh;
        overflow-y: auto;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem;
    }

    /* Minimal spacing for all sidebar elements */
    [data-testid="stSidebar"] h1 {
        font-size: 1.3rem;
        margin-top: 0;
        margin-bottom: 0.3rem;
        padding-top: 0;
    }

    [data-testid="stSidebar"] h2 {
        font-size: 1.1rem;
        margin-top: 0.3rem;
        margin-bottom: 0.2rem;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 0.95rem;
        margin-top: 0.2rem;
        margin-bottom: 0.1rem;
    }

    [data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0.2rem;
    }

    [data-testid="stSidebar"] p {
        margin-bottom: 0.2rem;
    }

    /* Ultra compact sliders */
    [data-testid="stSidebar"] .stSlider {
        margin-top: 0;
        margin-bottom: 0.3rem;
        padding-top: 0;
        padding-bottom: 0;
    }

    [data-testid="stSidebar"] .stSlider label {
        margin-bottom: 0.1rem;
    }

    /* Compact toggle */
    [data-testid="stSidebar"] .stCheckbox {
        margin-bottom: 0.3rem;
    }

    /* Compact file uploader */
    [data-testid="stSidebar"] .stFileUploader {
        margin-bottom: 0.3rem;
    }

    /* Bigger buttons */
    .stButton > button {
        width: 100%;
        height: 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* Reduce title spacing */
    h1 {
        margin-top: 0;
        padding-top: 0;
    }

    /* Remove extra divider spacing */
    [data-testid="stSidebar"] hr {
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# init session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = SimplificationPipeline()
if "results" not in st.session_state:
    st.session_state.results = None


# sidebar controls
st.sidebar.title("AI-LOD Simplifier")

# file upload
uploaded_file = st.sidebar.file_uploader(
    "upload mesh", type=["obj", "ply", "gltf", "glb"]
)

# parameters
st.sidebar.subheader("parameters")

use_ai = st.sidebar.toggle("Enable AI Importance", value=True, key="ai_toggle")

alpha = st.sidebar.slider(
    "alpha",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.1,
    disabled=not use_ai,
)

st.sidebar.subheader("lod targets")
ratio_1 = st.sidebar.slider("lod1", 0.0, 1.0, 0.5, 0.05, format="%.2f")
ratio_2 = st.sidebar.slider("lod2", 0.0, 1.0, 0.2, 0.05, format="%.2f")
ratio_3 = st.sidebar.slider("lod3", 0.0, 1.0, 0.05, 0.01, format="%.2f")

ratios = [ratio_1, ratio_2, ratio_3]

# run button
run_button = st.sidebar.button(
    "generate lods", type="primary", disabled=uploaded_file is None
)

# main area
st.title("AI-Guided LOD Mesh Simplification")
st.markdown("hybrid qem + ai importance for perceptually-aware mesh reduction")

if uploaded_file is None:
    st.info("upload a mesh to get started")
    st.stop()

# save uploaded file
with tempfile.NamedTemporaryFile(
    delete=False, suffix=Path(uploaded_file.name).suffix
) as tmp:
    tmp.write(uploaded_file.read())
    mesh_path = tmp.name

# run pipeline
if run_button:
    progress_text = st.empty()
    progress_bar = st.progress(0)

    try:
        # Show progress steps
        progress_text.text("Loading mesh...")
        progress_bar.progress(10)

        progress_text.text("Rendering orbital views...")
        progress_bar.progress(20)

        progress_text.text("Extracting AI saliency (this may take a while)...")
        progress_bar.progress(40)

        progress_text.text("Projecting importance to vertices...")
        progress_bar.progress(60)

        progress_text.text("Generating LODs with QEM simplification...")
        progress_bar.progress(80)

        # Run the pipeline (this blocks until done)
        results = st.session_state.pipeline.run(
            mesh_path, ratios, alpha=alpha, use_ai=use_ai
        )
        st.session_state.results = results

        progress_bar.progress(100)
        progress_text.text("")
        st.success(f"completed in {results['elapsed']:.1f}s")
    except Exception as e:
        progress_text.text("")
        progress_bar.empty()
        st.error(f"error: {e}")
        st.stop()
    finally:
        # Clear progress indicators
        import time

        time.sleep(1)
        progress_text.empty()
        progress_bar.empty()

# display results
if st.session_state.results is None:
    st.info("click 'generate lods' to run")
    st.stop()

results = st.session_state.results

# tabs for different views
tab1, tab2, tab3 = st.tabs(["lod comparison", "importance heatmap", "metrics"])

with tab1:
    render_lod_comparison_tab(results)

with tab2:
    render_heatmap_tab(results, st.session_state.pipeline)

with tab3:
    render_metrics_tab(results)

# footer
st.sidebar.markdown("---")
st.sidebar.markdown("built with [streamlit](https://streamlit.io)")
