# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
pip install -r requirements.txt
python3 worldgen/main.py
```

No build step, test suite, or linter is configured.

## Architecture

This is a procedural mesh generation and 3D visualization project. The pipeline is: generate a mesh → compute its dual → display it.

### `worldgen/mymesh.py` — Mesh data structures

- **`Mesh`**: Triangular mesh (vertices + triangular faces as numpy uint32 arrays). `icosphere(depth)` generates a subdivided icosahedron; vertices are sorted along the Z-axis for spatial locality.
- **`Polytope`**: General polytope with non-triangular faces. `mesh_dual()` computes the dual of a `Mesh` by placing new vertices at face centroids (normalized to unit sphere), then for each original vertex builds a face from the centroids of its adjacent faces, ordered by angle for correct winding.

### `worldgen/gui.py` — OpenGL/Qt renderer

- **`MeshViewer`** (`QOpenGLWidget`): Renders a `Polytope` using VBOs (OpenGL 3.3 compatibility). Drives ~60 FPS via a timer and rotates the mesh 0.5°/frame. Supports configurable blend modes for semi-transparent rendering.
- **`MainWindow`** (`QMainWindow`): Hosts `MeshViewer`; exposes `set_mesh(polytope)` to swap the displayed mesh at runtime.

### `worldgen/main.py` — Entry point

Creates an `icosphere(depth=3)`, computes its dual, and opens the viewer.

## Key Design Notes

- `Mesh` stores only triangular faces; `Polytope` handles arbitrary polygon faces needed for duals (e.g., a depth-3 icosphere's dual has pentagonal and hexagonal faces).
- The dual construction normalizes centroids to the unit sphere so the dual lives on the same sphere as the original, not scaled by face area.
- OpenGL context is set to 3.3 compatibility profile (not core) so legacy calls remain available during development.
