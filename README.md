# procgen

Eventually a super-project for all things procedural generation

## Setup

Requires Python 3.13+ and [Homebrew](https://brew.sh).

```bash
brew install python@3.13
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After the first time, reactivate with:

```bash
source .venv/bin/activate
```

## `worldgen/mesh`

### Overview

The pipeline is: build a triangular `Mesh` → deregularize it → compute its dual `PolygonMesh` → display it.

```
Mesh.icosphere(depth)
    └─ .deregularize(...)   # edge-swap + spring relax
        └─ .mesh_dual()     # → PolygonMesh
            └─ MainWindow(mesh=dual)
```

### `mymesh.py`

**`PolygonMesh(vertices, faces)`**

Base class for any polygon mesh. Faces are tuples of vertex indices of arbitrary length.
On construction it fan-triangulates every face and pre-builds flat `render_vertices` and `render_colors` arrays (one random color per face) ready for direct VBO upload.

**`Mesh(vertices, faces)`** — subclass of `PolygonMesh`

Triangular mesh only (faces are 3-tuples). Vertices are re-sorted along the Z-axis at construction for spatial locality.

- `Mesh.icosphere(depth)` — generates a subdivided icosahedron, projecting all vertices onto the unit sphere. Each subdivision level quadruples the face count: depth 0 gives 20 triangles, depth `n` gives `20 × 4ⁿ`.
- `.deregularize(n_swaps, n_relax, stiffness, outer_iters)` — breaks the icosphere's regularity by alternating random edge-swaps with spring relaxation. Edge-swaps are rejected if they would create a vertex with degree ≤ 5 (squares in the dual) or ≥ 7 (octagons). Spring relaxation pulls each vertex toward the mean position of its neighbors and re-projects onto the unit sphere.
- `.mesh_dual()` — builds the dual `PolygonMesh`. Each original face becomes a vertex (its centroid, normalized to the unit sphere). Each original vertex becomes a face whose vertices are the centroids of its adjacent faces, ordered CCW by projecting onto the tangent plane at that vertex.

**`PolygonMesh.set_face_metadata(face_idx, **kwargs)`**

Attaches arbitrary key-value data to a face for downstream use.

### `gui.py`

**`MeshViewer(QOpenGLWidget)`**

Renders a `PolygonMesh` via two VBOs (vertices + colors) using `glDrawArrays(GL_TRIANGLES, ...)`. Rotates the mesh 0.5°/frame driven by a 16 ms `QTimer` (~60 FPS). OpenGL 3.3 compatibility profile.

**`MainWindow(QMainWindow)`**

Hosts `MeshViewer` in an 800×600 window. Call `set_mesh(mesh)` to swap the displayed mesh at runtime.

### `test/test_mymesh.py`

Covers `Mesh` (vertex/face counts at various subdivision depths, unit-sphere property, outward face normals) and `PolygonMesh.mesh_dual` (dual topology counts, face sizes, outward winding of all fan triangles, convexity of dual faces on the sphere).
