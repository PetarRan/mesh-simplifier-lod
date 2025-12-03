import numpy as np


def project_importance_to_vertices(mesh, views, saliency_maps):
    """
    project 2d saliency onto 3d vertices via depth testing
    returns importance array [0,1] per vertex
    """
    num_verts = len(mesh.vertices)
    importance_sum = np.zeros(num_verts)
    view_count = np.zeros(num_verts)

    fov = np.pi / 3.0  # matches renderer

    for view, saliency in zip(views, saliency_maps):
        h, w = saliency.shape
        focal = (h / 2.0) / np.tan(fov / 2.0)

        # transform vertices to camera space
        cam_inv = np.linalg.inv(view["pose"])
        verts_cam = _transform_points(mesh.vertices, cam_inv)

        # project to image plane
        z = -verts_cam[:, 2]
        u = (verts_cam[:, 0] / (z + 1e-8) * focal + w / 2).astype(int)
        v = (-verts_cam[:, 1] / (z + 1e-8) * focal + h / 2).astype(int)

        # valid projections
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)

        # depth test + accumulate saliency
        for i in np.where(valid)[0]:
            depth_rendered = view["depth"][v[i], u[i]]
            if depth_rendered > 0 and abs(z[i] - depth_rendered) < 0.1:
                importance_sum[i] += saliency[v[i], u[i]]
                view_count[i] += 1

    # average over views
    importance = np.zeros(num_verts)
    seen = view_count > 0
    importance[seen] = importance_sum[seen] / view_count[seen]

    # normalize
    if importance.max() > 0:
        importance /= importance.max()

    return importance


def _transform_points(points, transform):
    """apply 4x4 transform to 3d points"""
    ones = np.ones((len(points), 1))
    points_h = np.hstack([points, ones])
    return (transform @ points_h.T).T[:, :3]