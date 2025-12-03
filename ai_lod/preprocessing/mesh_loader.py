import numpy as np
import trimesh


def load_mesh(path):
    """load mesh and ensure normals exist"""
    mesh = trimesh.load(str(path), force="mesh")

    if not hasattr(mesh, "vertex_normals") or np.allclose(mesh.vertex_normals, 0):
        mesh.fix_normals()

    return mesh
