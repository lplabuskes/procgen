"""Polity border lines for a region or local map, contoured from the refined fill.

The world's border polylines follow the world rivers; a region re-routes its
rivers within a corridor, and the Polities fill there snaps to the drawn ones
(``RIVER_SEGMENTS_KEY``). These lines are traced from that same field — the
world ``PolityLabelField`` with the view's rivers swapped in — on a triangle
lattice over the view's own ``(s, t)`` frame, with water from the refined masks,
so they bound the fill the view draws. Same tracer as the world
(:func:`realmgen.refine.borders.trace_lattice_borders`).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from realmgen.refine.borders import _unique_edges, trace_lattice_borders
from realmgen.refine.fill import WATER, PolityLabelField
from region.upsample import RegionFields


def _square_lattice(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(st (V, 2), tris (T, 3))``: an ``(n+1)²`` grid over ``[0, 1]²``, two triangles per square."""
    g = np.linspace(0.0, 1.0, n + 1)
    ss, tt = np.meshgrid(g, g, indexing="xy")
    st = np.column_stack([ss.ravel(), tt.ravel()])
    idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([np.stack([a, b, d], axis=1), np.stack([a, d, c], axis=1)])
    return st, tris


def view_polity_borders(fields: RegionFields, field: PolityLabelField, polities: List,
                        river_segments: Optional[np.ndarray], *, iterations: int = 2,
                        lattice_texels: float = 1.5, bisect_steps: int = 5) -> dict:
    """``{depth: (M, 6)}`` border segments over this view, snapped to ``river_segments``.

    The lattice spacing is ``lattice_texels`` refined texels; points off the cell
    count as water (lines end there); past a local map's hexagon the view clips.
    """
    if not polities:
        return {}
    view = field.with_rivers(river_segments) if river_segments is not None else field
    geom, grid, tex = fields.geometry, fields.grid, int(fields.tex_size)
    width = float(np.linalg.norm(
        geom.local_to_ecef(np.array([0.0]), np.array([0.5]))[0]
        - geom.local_to_ecef(np.array([1.0]), np.array([0.5]))[0]))
    spacing = max(grid.texel_spacing_rad(tex) * float(lattice_texels), 1e-9)
    n = max(8, int(np.ceil(width / spacing)))
    st, tris = _square_lattice(n)
    verts = geom.local_to_ecef(st[:, 0], st[:, 1])

    hexes, rows, cols, inside = grid.texels_at(st[:, 0], st[:, 1], tex)
    water = ~inside
    for key in ("ocean_mask", "lake_mask"):
        mask = fields.textures.get(key)
        if mask is not None:
            water |= mask[hexes, rows, cols, 0] > 0.5
    labels = view.land_labels(verts)
    labels[water] = WATER

    edges, tri_edges = _unique_edges(tris, len(verts))
    return trace_lattice_borders(verts, tris, edges, tri_edges, labels, polities,
                                 iterations=iterations, label_fn=view.land_labels,
                                 bisect_steps=bisect_steps)
