"""UI component rendering for Streamlit webapp"""

import streamlit as st
import pandas as pd
from visualization import mesh_to_plotly, mesh_with_colors


def render_lod_comparison_tab(results):
    """Render LOD comparison tab"""
    st.subheader("lod comparison")

    # Summary stats
    original_faces = len(results["lods"][0].faces)
    st.markdown(
        f"**Original:** {original_faces:,} faces | **LOD Levels:** {len(results['lods']) - 1}")

    # show lods side by side
    cols = st.columns(4)

    labels = ["original", "lod1", "lod2", "lod3"]
    for i, (col, mesh, label) in enumerate(zip(cols, results["lods"], labels)):
        with col:
            st.markdown(f"**{label.upper()}**")

            # Stats
            faces = len(mesh.faces)
            verts = len(mesh.vertices)
            reduction = (1 - faces / original_faces) * 100 if i > 0 else 0

            st.markdown(f"**Faces:** {faces:,}")
            st.markdown(f"**Vertices:** {verts:,}")
            if i > 0:
                st.markdown(f"**Reduction:** {reduction:.1f}%")

            # Visualization
            fig = mesh_to_plotly(
                mesh, color=["lightblue", "lightgreen", "orange", "salmon"][i])
            st.plotly_chart(fig, use_container_width=True)

            # download link
            if i > 0:
                with open(results["lod_paths"][i], "rb") as f:
                    st.download_button(
                        f"download {label}.obj",
                        f,
                        file_name=f"{label}.obj",
                        mime="application/octet-stream",
                    )


def render_heatmap_tab(results, pipeline):
    """Render importance heatmap tab"""
    st.subheader("importance heatmap")

    if results["heatmap_mesh"] is not None:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig = mesh_with_colors(results["heatmap_mesh"])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Importance Stats")

            if results["importance"] is not None:
                importance = results["importance"]
                st.markdown(f"**Min:** {importance.min():.3f}")
                st.markdown(f"**Max:** {importance.max():.3f}")
                st.markdown(f"**Mean:** {importance.mean():.3f}")
                st.markdown(f"**Std:** {importance.std():.3f}")
                st.markdown("")

            st.markdown("### Color Scale")
            st.markdown("- **Warm (red/yellow):** High importance")
            st.markdown("- **Cool (blue/purple):** Low importance")
            st.markdown("")
            st.markdown(
                "**How it works:** DINOv2 extracts saliency from 6 orbital views. Regions with high saliency are preserved during QEM simplification.")

            # download heatmap
            heatmap_path = pipeline.temp_dir / "heatmap.obj"
            if heatmap_path.exists():
                st.markdown("")
                with open(heatmap_path, "rb") as f:
                    st.download_button(
                        "download heatmap.obj",
                        f,
                        file_name="heatmap.obj",
                        mime="application/octet-stream",
                    )
    else:
        st.info("enable ai importance to see heatmap")


def render_metrics_tab(results):
    """Render quality metrics tab"""
    st.subheader("quality metrics")

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total LOD Levels", len(results["comparisons"]))
    with col2:
        st.metric("AI Enabled", "Yes" if results.get(
            "importance") is not None else "No")
    with col3:
        st.metric("Processing Time", f"{results['elapsed']:.1f}s")

    st.markdown("---")

    # build metrics table
    rows = []
    for i, comp in enumerate(results["comparisons"]):
        original_faces = len(results["lods"][0].faces)
        simplified_faces = comp["simplified"]["num_faces"]
        reduction_pct = (1 - simplified_faces / original_faces) * 100

        rows.append(
            {
                "LOD": f"LOD{i+1}",
                "Faces": f"{simplified_faces:,}",
                "Vertices": f"{comp['simplified']['num_vertices']:,}",
                "Reduction": f"{reduction_pct:.1f}%",
                "Target Ratio": f"{comp['face_ratio']:.2f}",
                "Tri/Vert Ratio": f"{simplified_faces / comp['simplified']['num_vertices']:.2f}",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("")
    st.markdown("### Metrics Explanation")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Reduction:** Percentage of faces removed from original")
        st.markdown("**Target Ratio:** Desired face count ratio")
    with col2:
        st.markdown(
            "**Tri/Vert Ratio:** Average triangles per vertex (quality indicator)")
        st.markdown("**Faces/Vertices:** Mesh complexity indicators")
