"""Mesh visualization utilities for Streamlit"""

import plotly.graph_objects as go
import numpy as np


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
