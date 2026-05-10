import random
from typing import List, Tuple, Dict, FrozenSet

import numpy as np

class Mesh:
    def __init__(self, vertices: np.ndarray, faces: List[Tuple[int, int, int]]):
        tmp = np.argsort(vertices[:,2])  # Sort along z axis
        mapping = np.argsort(tmp)
        self.vertices = vertices[tmp].astype(np.float32)  # the coordinates of each vertex
        self.faces = sorted([(mapping[x], mapping[y], mapping[z]) for x, y, z in faces])
        self.index_array = np.array(self.faces, dtype=np.uint32)  # all faces are 3-tuples so just directly convert to numpy array

        n_faces = len(self.faces)
        face_colors = np.random.rand(n_faces, 4).astype(np.float32)
        face_colors[:, 3] = 1.0
        flat_indices = self.index_array.flatten()
        self.render_vertices = self.vertices[flat_indices]
        self.render_colors = np.repeat(face_colors, 3, axis=0)

    def deregularize(self, n_swaps: int = 1000, n_relax: int = 10,
                     stiffness: float = 0.1, outer_iters: int = 3) -> 'Mesh':
        """Edge-swap + spring-relax loop to break icosphere regularity."""
        verts = self.vertices.astype(np.float64)
        faces: List[Tuple[int, int, int]] = list(self.faces)
        n_verts = len(verts)

        # adj[v] = set of neighboring vertex indices
        adj: List[set] = [set() for _ in range(n_verts)]
        # vertex_faces[v] = list of face indices that contain v
        vertex_faces: List[List[int]] = [[] for _ in range(n_verts)]
        # edge_faces[frozenset{a,b}] = list of face indices sharing that edge
        edge_faces: Dict[FrozenSet, List[int]] = {}

        for fi, (a, b, c) in enumerate(faces):
            vertex_faces[a].append(fi)
            vertex_faces[b].append(fi)
            vertex_faces[c].append(fi)
            adj[a].update([b, c])
            adj[b].update([a, c])
            adj[c].update([a, b])
            for e in (frozenset([a, b]), frozenset([a, c]), frozenset([b, c])):
                edge_faces.setdefault(e, []).append(fi)

        def ensure_ccw(p0: int, p1: int, p2: int) -> Tuple[int, int, int]:
            centroid = (verts[p0] + verts[p1] + verts[p2]) / 3.0
            normal = np.cross(verts[p1] - verts[p0], verts[p2] - verts[p0])
            return (p0, p1, p2) if np.dot(normal, centroid) >= 0 else (p0, p2, p1)

        def remove_face(fi: int) -> None:
            f = faces[fi]
            a, b, c = f
            for v in f:
                vertex_faces[v].remove(fi)
            for e in (frozenset([a, b]), frozenset([a, c]), frozenset([b, c])):
                edge_faces[e].remove(fi)
                if not edge_faces[e]:
                    del edge_faces[e]

        def add_face(fi: int, f: Tuple[int, int, int]) -> None:
            faces[fi] = f
            a, b, c = f
            for v in f:
                vertex_faces[v].append(fi)
            for e in (frozenset([a, b]), frozenset([a, c]), frozenset([b, c])):
                edge_faces.setdefault(e, []).append(fi)

        for _ in range(outer_iters):
            # --- Edge swap phase ---
            for _ in range(n_swaps):
                fi_pick = random.randrange(len(faces))
                f_pick = faces[fi_pick]
                ei = random.randrange(3)
                bv, cv = f_pick[ei], f_pick[(ei + 1) % 3]
                edge = frozenset([bv, cv])
                ef = edge_faces.get(edge)
                if ef is None or len(ef) != 2:
                    continue

                fi1, fi2 = ef[0], ef[1]
                f1, f2 = faces[fi1], faces[fi2]
                av = next(v for v in f1 if v != bv and v != cv)
                dv = next(v for v in f2 if v != bv and v != cv)

                # av and dv gain a degree. avoid degree >=7 to avoid octagons
                if len(adj[av]) >= 7 or len(adj[dv]) >= 7:
                    continue
                # bv and cv lose a degree. avoid degree <=5 to avoid squares
                if len(adj[bv]) <= 5 or len(adj[cv]) <= 5:
                    continue

                # ensure_ccw only swaps p1/p2, so result is (av, bv, dv) or (av, dv, bv)
                new_f1 = ensure_ccw(av, bv, dv)
                p0, p1, p2 = new_f1
                # Derive new_f2 so the shared AD edge runs opposite to new_f1
                if (p0 == av and p1 == dv) or (p1 == av and p2 == dv) or (p2 == av and p0 == dv):
                    new_f2 = (dv, av, cv)  # a→d in new_f1 → d→a in new_f2
                else:
                    new_f2 = (av, dv, cv)  # d→a in new_f1 → a→d in new_f2

                remove_face(fi1)
                remove_face(fi2)
                adj[bv].discard(cv)
                adj[cv].discard(bv)
                adj[av].add(dv)
                adj[dv].add(av)
                add_face(fi1, new_f1)
                add_face(fi2, new_f2)

            # --- Spring relaxation phase ---
            # Build directed edge arrays for vectorized mean-neighbor computation
            src_list, dst_list = [], []
            for v in range(n_verts):
                for u in adj[v]:
                    src_list.append(v)
                    dst_list.append(u)
            src_arr = np.array(src_list)
            dst_arr = np.array(dst_list)
            degree = np.bincount(src_arr, minlength=n_verts).astype(np.float64)

            for _ in range(n_relax):
                neighbor_sum = np.zeros((n_verts, 3), dtype=np.float64)
                np.add.at(neighbor_sum, src_arr, verts[dst_arr])
                target = neighbor_sum / degree[:, np.newaxis]
                verts += stiffness * (target - verts)
                verts /= np.linalg.norm(verts, axis=1, keepdims=True)

        # Commit results
        self.vertices = verts.astype(np.float32)
        self.faces = faces
        self.index_array = np.array(faces, dtype=np.uint32)
        n_faces = len(faces)
        face_colors = np.random.rand(n_faces, 4).astype(np.float32)
        face_colors[:, 3] = 1.0
        flat_indices = self.index_array.flatten()
        self.render_vertices = self.vertices[flat_indices]
        self.render_colors = np.repeat(face_colors, 3, axis=0)
        return self

    # Took vertices from https://github.com/lazysquirrellabs/sphere_generator/blob/main/Assets/Lazy%20Squirrel%20Labs/SphereGenerator/Runtime/Generators/IcosphereGenerator.cs
    @staticmethod
    def icosphere(depth: int = 0):
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
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
        faces = [
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

        for _ in range(depth):
            # Subdivide each triangle into 4 smaller triangles
            new_faces = []
            new_vertices = []  # temporary cache of added vertices to cut down on numpy appends
            edge_indices = {}  # given two vertices, what's the index of their midpoint

            for face in faces:
                e0 = (face[0], face[1]) if face[0]<face[1] else (face[1], face[0])
                e1 = (face[0], face[2]) if face[0]<face[2] else (face[2], face[0])
                e2 = (face[1], face[2]) if face[1]<face[2] else (face[2], face[1])

                if e0 in edge_indices:
                    v3 = edge_indices[e0]
                else:
                    pt3 = (vertices[face[0]] + vertices[face[1]])/2
                    v3 = vertices.shape[0] + len(new_vertices)
                    edge_indices[e0] = v3
                    new_vertices.append(pt3)

                if e1 in edge_indices:
                    v4 = edge_indices[e1]
                else:
                    pt4 = (vertices[face[0]] + vertices[face[2]])/2
                    v4 = vertices.shape[0] + len(new_vertices)
                    edge_indices[e1] = v4
                    new_vertices.append(pt4)

                if e2 in edge_indices:
                    v5 = edge_indices[e2]
                else:
                    pt5 = (vertices[face[1]] + vertices[face[2]])/2
                    v5 = vertices.shape[0] + len(new_vertices)
                    edge_indices[e2] = v5
                    new_vertices.append(pt5)

                # at this point, any new vertices have been added to the new vertices list
                # the indices of interest are face[0], face[1], face[2], v3, v4, v5
                new_faces.append((face[0], v3, v4))
                new_faces.append((v3, face[1], v5))
                new_faces.append((v3, v5, v4))
                new_faces.append((v4, v5, face[2]))

            faces = new_faces
            vertices = np.vstack([vertices] + new_vertices)

        # We have a subdivided icosahedron, normalize vertices to get icosphere
        vertices /= np.linalg.norm(vertices, axis=1)[..., np.newaxis]

        return Mesh(vertices, faces)

    @staticmethod
    def triangle():
        v = np.array([[-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0]])
        f = [(0, 1, 2)]
        return Mesh(v, f)


# A general polytope that can have non-trriangular sides. Mainly used for representing the dual of a mesh
class Polytope:
    def __init__(self, vertices: np.ndarray, faces: List[Tuple[int]]):
        self.vertices = vertices.astype(np.float32)
        self.faces = faces

        n_faces = len(faces)
        face_colors = np.random.rand(n_faces, 4).astype(np.float32)
        face_colors[:, 3] = 1.0

        triangles = []
        triangle_face_indices = []
        for face_idx, face in enumerate(faces):
            for i in range(1, len(face)-1):
                triangles.append((face[0], face[i], face[i+1]))
                triangle_face_indices.append(face_idx)
        self.index_array = np.array(triangles, dtype=np.uint32)

        flat_indices = self.index_array.flatten()
        self.render_vertices = self.vertices[flat_indices]
        triangle_colors = face_colors[np.array(triangle_face_indices)]
        self.render_colors = np.repeat(triangle_colors, 3, axis=0)

    @staticmethod
    def mesh_dual(mesh: Mesh):
        # First compute the centroids of each face
        dual_vertices = []
        for face in mesh.faces:
            pts = mesh.vertices[np.array(face)]
            centroid = np.mean(pts, axis=0)
            centroid /= np.linalg.norm(centroid)
            dual_vertices.append(centroid)
        dual_vertices = np.array(dual_vertices)

        # Now, for each vertex in the original mesh, find all faces that include it
        dual_faces_dict: Dict[int, list[int]] = {}
        for face_idx, face in enumerate(mesh.faces):
            for v in face:
                if v not in dual_faces_dict:
                    dual_faces_dict[v] = [face_idx]
                else:
                    dual_faces_dict[v].append(face_idx)

        # Reorder the face indices in each dual face to be CCW when viewed from outside.
        # We project each face centroid onto the tangent plane at the vertex and
        # sort by angle in that local 2D frame. Using the global XY projection
        # (arctan2 of the raw x/y components) fails for equatorial vertices because
        # it ignores the z-component of the displacement vectors.
        dual_faces = []
        for v_idx, face_indices in dual_faces_dict.items():
            v = mesh.vertices[v_idx]
            n = v  # unit outward normal (v is already on the unit sphere)
            ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u_axis = np.cross(ref, n)
            u_axis /= np.linalg.norm(u_axis)
            w_axis = np.cross(n, u_axis)  # u_axis × w_axis = n (right-hand frame)

            def angle_from_center(face_idx, u=u_axis, w=w_axis, center=v):
                vec = dual_vertices[face_idx] - center
                return np.arctan2(np.dot(vec, w), np.dot(vec, u))

            face_indices.sort(key=angle_from_center)
            dual_faces.append(tuple(face_indices))

        return Polytope(dual_vertices, dual_faces)
