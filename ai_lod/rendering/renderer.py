import numpy as np
import pyrender


class OffscreenRenderer:
    """offscreen rendering, pyrender wrapper (m1 compatible)"""

    def __init__(self, resolution=256):
        self.resolution = resolution
        self.renderer = pyrender.OffscreenRenderer(resolution, resolution)

    def render_views(self, mesh, num_views=6, distance_scale=2.5):
        """render orbit views around mesh"""
        # setup scene
        mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        scene.add(mesh_pr)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=self._light_pose())

        # camera setup
        centroid = mesh.centroid
        distance = np.max(mesh.bounds[1] - mesh.bounds[0]) * distance_scale
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        views = []

        # orbit around mesh
        for i in range(num_views):
            angle = 2 * np.pi * i / num_views

            # camera position
            cam_pos = np.array([
                centroid[0] + distance * np.cos(angle),
                centroid[1] + distance * 0.3,
                centroid[2] + distance * np.sin(angle),
            ])

            camera_pose = self._look_at(cam_pos, centroid, np.array([0, 1, 0]))
            cam_node = scene.add(camera, pose=camera_pose)

            # render
            color, depth = self.renderer.render(scene)
            views.append({"image": color, "depth": depth, "pose": camera_pose})

            scene.remove_node(cam_node)

        return views

    def _look_at(self, eye, center, up):
        """create look-at matrix"""
        f = (center - eye) / np.linalg.norm(center - eye)
        s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
        u = np.cross(s, f)

        m = np.eye(4)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        m[:3, 3] = eye
        return m

    def _light_pose(self):
        """light position"""
        pose = np.eye(4)
        pose[:3, 3] = [2, 5, 2]
        return pose

    def cleanup(self):
        """release resources"""
        self.renderer.delete()
