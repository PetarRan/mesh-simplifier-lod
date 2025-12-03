import numpy as np
import trimesh
import heapq
from copy import deepcopy


class QEMSimplifier:
    """qem mesh simplification with optional ai importance modulation"""

    def __init__(self, mesh, importance=None, alpha=1.0):
        self.original_mesh = mesh
        self.mesh = deepcopy(mesh)
        self.alpha = alpha
        self.importance = (
            np.ones(len(mesh.vertices)) if importance is None else np.array(importance)
        )

        self.quadrics = self._init_quadrics()
        self.heap = []
        self._build_heap()

    def simplify(self, target_ratio=0.5):
        """simplify to target_ratio of original faces"""
        target = int(len(self.original_mesh.faces) * target_ratio)
        current = len(self.mesh.faces)

        print(f"simplifying {current} → {target} faces...")

        collapses = 0
        while current > target and self.heap:
            cost, edge_key, v1, v2, new_pos = heapq.heappop(self.heap)

            if not self._valid_edge(v1, v2):
                continue

            self._collapse(v1, v2, new_pos)
            collapses += 1
            current = len(self.mesh.faces)

            if collapses % 100 == 0:
                print(f"  {collapses} collapses, {current} faces left")

        print(f"done: {collapses} collapses, {current} faces")

        self.mesh.remove_degenerate_faces()
        self.mesh.remove_unreferenced_vertices()

        return self.mesh

    def _init_quadrics(self):
        """compute per-vertex quadric error matrices"""
        quadrics = [np.zeros((4, 4)) for _ in range(len(self.mesh.vertices))]

        for face_idx, face in enumerate(self.mesh.faces):
            # plane equation: ax + by + cz + d = 0
            normal = self.mesh.face_normals[face_idx]
            d = -np.dot(normal, self.mesh.vertices[face[0]])
            plane = np.append(normal, d)

            # quadric Q = pp^T
            Q = np.outer(plane, plane)

            for v_idx in face:
                quadrics[v_idx] += Q

        return quadrics

    def _build_heap(self):
        """build priority queue of edge collapses"""
        for v1, v2 in self.mesh.edges_unique:
            cost, pos = self._collapse_cost(v1, v2)
            heapq.heappush(self.heap, (cost, tuple(sorted([v1, v2])), v1, v2, pos))

    def _collapse_cost(self, v1, v2):
        """compute cost of collapsing edge (v1, v2)"""
        Q = self.quadrics[v1] + self.quadrics[v2]

        # use midpoint (could solve Q*v=0 for optimal, but midpoint works fine)
        v_new = (self.mesh.vertices[v1] + self.mesh.vertices[v2]) / 2.0
        v_h = np.append(v_new, 1.0)

        # qem cost
        qem = v_h @ Q @ v_h

        # ai modulation: higher importance → higher cost → preserve
        importance = (self.importance[v1] + self.importance[v2]) / 2.0
        cost = qem * (1.0 + self.alpha * importance)

        return cost, v_new

    def _valid_edge(self, v1, v2):
        """check if vertices still exist in mesh"""
        return (
            np.any(np.any(self.mesh.faces == v1, axis=1))
            and np.any(np.any(self.mesh.faces == v2, axis=1))
        )

    def _collapse(self, v1, v2, new_pos):
        """collapse edge by merging v2 into v1"""
        self.mesh.vertices[v1] = new_pos
        self.quadrics[v1] += self.quadrics[v2]
        self.importance[v1] = max(self.importance[v1], self.importance[v2])
        self.mesh.faces[self.mesh.faces == v2] = v1
