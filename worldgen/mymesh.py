import numpy as np
from typing import List, Tuple


class Mesh:
    def __init__(self, vertices: np.ndarray, faces: List[Tuple[int]]):
        self.vertices = vertices  # the coordinates of each vertex
        self.faces = faces  # a tuple of the indices of vertices that make each triangle

    # Based on https://github.com/lazysquirrellabs/sphere_generator/blob/main/Assets/Lazy%20Squirrel%20Labs/SphereGenerator/Runtime/Generators/IcosphereGenerator.cs
    @staticmethod
    def icosphere(order: int = 3):
        phi = (1 + np.sqrt(5)) / 2
        primary_vertices = np.array([
            [phi, 1, 0],
            [0, phi, -1],
            [0, phi, 1],
            [1, 0, -phi],
            [1, 0, phi],
            [phi, -1, 0],
            [-1, 0, -phi],
            [-phi, 1, 0],
            [-1, 0, phi],
            [0, -phi, -1],
            [0, -phi, 1],
            [-phi, -1, 0]
        ])
        primary_faces = [
            (0, 1, 2),
            (0, 3, 1),
            (0, 2, 4),
            (3, 0, 5),
            (0, 4, 5),
            (1, 3, 6),
            (1, 7, 2),
            (7, 1, 6),
            (4, 2, 8),
            (7, 8, 2),
            (9, 3, 5),
            (6, 3, 9),
            (5, 4, 10),
            (4, 8, 10),
            (9, 5, 10),
            (7, 6, 11),
            (7, 11, 8),
            (11, 6, 9),
            (8, 11, 10),
            (10, 11, 9)
        ]
        return Mesh(primary_vertices, primary_faces)

    @staticmethod
    def triangle():
        v = np.array([[-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0]])
        f = [(0, 1, 2)]
        return Mesh(v, f)
