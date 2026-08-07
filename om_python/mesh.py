"""Minimal Wavefront OBJ loader for the OpenManipulator-X chain meshes.

The mesh files (see meshes/*.obj) contain plain triangulated `v`/`f` data with
no normals or texture coordinates, so the loader only needs to handle that
subset. Per-triangle (flat) normals are computed on load since the source
files don't include any.
"""
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_FLOAT,
    GL_NORMAL_ARRAY,
    GL_TRIANGLES,
    GL_VERTEX_ARRAY,
    glDisableClientState,
    glDrawArrays,
    glEnableClientState,
    glNormalPointer,
    glVertexPointer,
)


class Mesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        triangles = vertices[faces]  # (M, 3, 3)
        self.vertex_data = triangles.reshape(-1, 3).astype(np.float32)

        v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        face_normals = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        face_normals /= lengths
        self.normal_data = np.repeat(face_normals, 3, axis=0).astype(np.float32)

        self.vertex_count = self.vertex_data.shape[0]

    def draw(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.vertex_data)
        glNormalPointer(GL_FLOAT, 0, self.normal_data)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)


def load_obj(path: "str | Path") -> Mesh:
    vertices = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for i in range(1, len(idx) - 1):  # fan-triangulate defensively
                    faces.append((idx[0], idx[i], idx[i + 1]))

    return Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )
