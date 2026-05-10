# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
pip install -r requirements.txt
python3 worldgen/mesh/main.py
```

No build step or linter is configured. Run the test suite with:

```bash
cd worldgen && python3 -m pytest mesh/test/test_mymesh.py -v
```

Tests live in `worldgen/mesh/test/test_mymesh.py` and cover `Mesh` (icosphere topology, unit-sphere property, outward face normals) and `Polytope.mesh_dual` (dual topology counts, face sizes, and correct CCW winding of all fan triangles).

## Architecture

This is a procedural mesh generation and 3D visualization project. The pipeline is: generate a mesh → compute its dual → display it.

### `worldgen/mesh/mymesh.py` — Mesh data structures

- **`Mesh`**: Triangular mesh (vertices + triangular faces as numpy uint32 arrays). `icosphere(depth)` generates a subdivided icosahedron; vertices are sorted along the Z-axis for spatial locality. Pre-computes `render_vertices` and `render_colors` (flat per-vertex arrays) in `__init__` for direct VBO upload.
- **`Polytope`**: General polytope with non-triangular faces. `mesh_dual()` computes the dual of a `Mesh` by placing new vertices at face centroids (normalized to unit sphere), then for each original vertex builds a face from the centroids of its adjacent faces, ordered by angle for correct winding. Also pre-computes `render_vertices`/`render_colors` by fan-triangulating each face.

### `worldgen/mesh/gui.py` — OpenGL/Qt renderer

- **`MeshViewer`** (`QOpenGLWidget`): Renders a `Mesh` or `Polytope` using VBOs and `glDrawArrays` (OpenGL 3.3 compatibility). Drives ~60 FPS via a timer and rotates the mesh 0.5°/frame.
- **`MainWindow`** (`QMainWindow`): Hosts `MeshViewer`; exposes `set_mesh(mesh)` to swap the displayed mesh at runtime.

### `worldgen/mesh/main.py` — Entry point

Creates an `icosphere(depth=5)`, computes its dual, and opens the viewer.

## Key Design Notes

- `Mesh` stores only triangular faces; `Polytope` handles arbitrary polygon faces needed for duals (e.g., a depth-1 icosphere dual has 12 pentagons + 30 hexagons).
- The dual construction normalizes centroids to the unit sphere so the dual lives on the same sphere as the original, not scaled by face area.
- Both `Mesh` and `Polytope` pre-expand geometry into flat `render_vertices`/`render_colors` arrays at construction time; the renderer calls `glDrawArrays` with no index buffer.
- OpenGL context is set to 3.3 compatibility profile (not core) so legacy calls remain available during development.
