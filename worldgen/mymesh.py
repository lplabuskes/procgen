import numpy as np
from typing import List, Tuple


class Mesh:
    def __init__(self, vertices: np.ndarray, faces: List[Tuple[int]]):
        self.vertices = vertices  # the coordinates of each vertex
        self.faces = faces  # a tuple of the indices of vertices that make each triangle


def generate_mesh():
    v = np.array([[-1.0, -1.0, 0.0],
                  [1.0, -1.0, 0.0],
                  [0.0, 1.0, 0.0]])
    f = [(0, 1, 2)]
    return Mesh(v, f)
