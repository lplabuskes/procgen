"""Lots as boundaries: a block's ground and the lines dividing it, the lots derived.

A :class:`~settlegen.layout.types.Block` stores its ground (``polygon``), the lines
between its lots and along their backs (``dividers``, inside the ground) and a
point in each lot (``seeds``).  The lots are the faces of the ground cut by the
dividers that hold a seed (:func:`derive_lots`): neighbours share their edges and
nothing overlaps.  Faces without a seed are open ground — a village's gaps and
paddocks, a deep block's yards.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry

# Divider pieces shorter than this (m) are overlay noise (a real piece between two cuts can be mm).
_MIN_LINE_M = 1e-6
# Lots are derived on a grid this fine (m).
_GRID_M = 1e-6


def as_lines(geom: BaseGeometry) -> MultiLineString:
    """The line parts of ``geom`` (longer than overlay noise), as one MultiLineString."""
    parts = [g for g in shapely.get_parts(shapely.line_merge(geom) if not geom.is_empty else geom)
             if isinstance(g, LineString) and g.length > _MIN_LINE_M]
    return MultiLineString(parts)


def dividers_of(polygon: Polygon, lots: Sequence[Polygon] | np.ndarray) -> MultiLineString:
    """The lines between ``lots`` (which tile part of ``polygon``) and round their open
    sides: their outlines less the ground's own."""
    if not len(lots):
        return MultiLineString()
    lines = shapely.union_all(shapely.boundary(np.asarray(lots, dtype=object)))
    return as_lines(lines.difference(polygon.boundary))


def derive_lots(polygon: Polygon, dividers: MultiLineString, seeds: np.ndarray) -> np.ndarray:
    """``(k,)`` lot polygons, one per seed: the face of ``polygon`` cut by ``dividers``
    holding it (the whole ground with no dividers)."""
    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 2)
    if len(seeds) == 0:
        return np.zeros(0, dtype=object)
    if dividers.is_empty:
        return np.asarray([polygon] * len(seeds), dtype=object)
    # Noded by snap-rounding: dividers end on the ground's edge only to rounding.
    lines = shapely.union_all([polygon.boundary, dividers], grid_size=_GRID_M)
    faces = np.asarray(shapely.get_parts(shapely.polygonize(shapely.get_parts(lines))), dtype=object)
    pts = np.asarray(shapely.points(seeds), dtype=object)
    out = np.empty(len(seeds), dtype=object)
    tree = shapely.STRtree(faces)
    hit_seed, hit_face = tree.query(pts, predicate="within")
    out[hit_seed] = faces[hit_face]
    for i in np.flatnonzero(np.asarray([o is None for o in out])):
        out[i] = faces[int(tree.nearest(pts[i]))]                       # on an edge: the nearest face
    return out


def seeds_of(lots: Sequence[Polygon] | np.ndarray) -> np.ndarray:
    """A point inside each lot."""
    if not len(lots):
        return np.zeros((0, 2))
    return shapely.get_coordinates(shapely.point_on_surface(np.asarray(lots, dtype=object))).reshape(-1, 2)


def merge_slivers(lots: np.ndarray, min_width: float) -> np.ndarray:
    """``lots`` with each sliver (under ``min_width`` across on average: its area over its
    minimum rectangle's long side — a wedge's taper counts) joined to
    the neighbour it shares the most edge with — the divider between them dropped — or,
    touching none, left out (open ground).  Lots with nothing to merge come back as they were."""
    lots = np.asarray(lots, dtype=object)
    if not len(lots):
        return lots
    with np.errstate(divide="ignore", invalid="ignore"):
        env = shapely.oriented_envelope(lots)
    xy, idx = shapely.get_coordinates(shapely.get_exterior_ring(env), return_index=True)
    first = np.searchsorted(idx, np.arange(len(lots)))
    along = np.maximum(np.hypot(*(xy[first + 1] - xy[first]).T), np.hypot(*(xy[first + 2] - xy[first + 1]).T))
    across = shapely.area(lots) / np.maximum(along, 1e-9)
    thin = np.flatnonzero((across < min_width) | (shapely.get_type_id(env) != 3))
    if not len(thin):
        return lots
    tree = shapely.STRtree(lots)
    edges = shapely.boundary(lots)
    into = np.full(len(lots), -1)
    skip = set(thin.tolist())
    for i in thin.tolist():
        near = [int(j) for j in tree.query(lots[i], predicate="dwithin", distance=1e-3) if int(j) not in skip]
        shared = [shapely.intersection(edges[i], edges[j].buffer(1e-3)).length for j in near]
        if shared and max(shared) > 0.5:
            into[i] = near[int(np.argmax(shared))]
    out = []
    for j in range(len(lots)):
        if j in skip:
            continue
        parts = [lots[j], *lots[into == j]]
        out.append(lots[j] if len(parts) == 1 else _one(shapely.union_all(parts, grid_size=_GRID_M)))
    return np.asarray(out, dtype=object)


def _one(g: BaseGeometry) -> Polygon:
    """``g``'s largest polygon."""
    return max((p for p in shapely.get_parts(g) if isinstance(p, Polygon)), key=lambda p: p.area)
