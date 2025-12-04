import numpy as np
import pyrender
from copy import deepcopy


class OffscreenRenderer:
    """offscreen rendering, pyrender wrapper"""

    def __init__(self, resolution=256):
        self.resolution = resolution
        self.renderer = pyrender.OffscreenRenderer(resolution, resolution)

    def render_views(self, mesh, num_views=6, distance_scale=2.5):
        """render orbit views around mesh"""
        # CENTER MESH AT ORIGIN
        mesh_centered = deepcopy(mesh)
        original_centroid = mesh.centroid
        mesh_centered.vertices -= original_centroid

        # setup scene
        mesh_pr = pyrender.Mesh.from_trimesh(mesh_centered, smooth=True)
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        scene.add(mesh_pr)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [2, 5, 2]
        scene.add(light, pose=light_pose)

        # camera setup - fixed distance that works
        distance = 1.0
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        views = []

        # orbit around mesh (now centered at origin)
        for i in range(num_views):
            angle = 2 * np.pi * i / num_views

            # Rotate the base camera pose (which points down -Z) around Y axis
            # Base pose: camera at (0, 0, distance) looking at origin
            rotation_y = np.array([
                [np.cos(angle), 0, -np.sin(angle), 0],
                [0, 1, 0, 0],
                [np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1]
            ])

            # Base camera pose (looking down -Z axis from +Z position)
            base_pose = np.eye(4)
            base_pose[2, 3] = distance

            # Rotate base pose around origin
            cam_pose = rotation_y @ base_pose

            cam_node = scene.add(camera, pose=cam_pose)

            # render
            color, depth = self.renderer.render(scene)

            # ttransform pose back to original coordinates for projection
            pose_original = cam_pose.copy()
            pose_original[:3, 3] += original_centroid

            views.append(
                {"image": color, "depth": depth, "pose": pose_original})

            scene.remove_node(cam_node)

        return views

    def cleanup(self):
        """release resources"""
        self.renderer.delete()
