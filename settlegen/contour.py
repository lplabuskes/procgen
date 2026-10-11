"""Vector outlines from the settlement terrain raster: marching squares, then polygons.

The site's polygons are traced from smooth fields, not from 0/1 masks: a mask
traces as a staircase of texels, a signed field (e.g. depth under a water
surface) at its zero crossing as a smooth shore.  Contours that leave the frame
are closed along its edge (:func:`regions_below`).
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union


def contour_segments(field: np.ndarray, xy: np.ndarray, level: float = 0.0) -> np.ndarray:
    """``(K, 2, 2)`` segments of ``field == level`` by marching squares.

    ``field`` ``(T, T)`` sampled at ``xy`` ``(T, T, 2)`` (texel centres).  Each
    cell between four samples contributes 0, 1 or 2 segments between its edge
    crossings (linearly interpolated); a saddle is split by the cell's mean.
    """
    f = np.asarray(field, dtype=np.float64) - float(level)
    f = np.where(f == 0.0, 1e-12, f)                     # nothing exactly on the level
    above = f > 0.0
    # Corners of each cell: 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left.
    v = [f[:-1, :-1], f[:-1, 1:], f[1:, 1:], f[1:, :-1]]
    p = [xy[:-1, :-1], xy[:-1, 1:], xy[1:, 1:], xy[1:, :-1]]
    a = [above[:-1, :-1], above[:-1, 1:], above[1:, 1:], above[1:, :-1]]
    case = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8

    def crossing(i: int, j: int) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):     # cells it doesn't cross aren't used
            t = v[i] / (v[i] - v[j])
            return p[i] + t[..., None] * (p[j] - p[i])

    # Edges 0 top, 1 right, 2 bottom, 3 left — each interpolated left → right or
    # top → bottom, so the two cells sharing an edge compute the same point exactly.
    edges = [crossing(0, 1), crossing(1, 2), crossing(3, 2), crossing(0, 3)]
    # Per case, the edge pairs to join (saddles 5 and 10 resolved by the centre).
    table = {
        1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)],
        9: [(0, 2)], 11: [(1, 2)], 12: [(1, 3)], 13: [(0, 1)], 14: [(3, 0)],
    }
    centre = (v[0] + v[1] + v[2] + v[3]) > 0.0
    out = []
    for c, pairs in table.items():
        m = case == c
        for e0, e1 in pairs:
            out.append(np.stack([edges[e0][m], edges[e1][m]], axis=1))
    for c, joined, split in ((5, [(3, 2), (1, 0)], [(3, 0), (1, 2)]), (10, [(0, 3), (2, 1)], [(0, 1), (2, 3)])):
        for flag, pairs in ((True, joined), (False, split)):
            m = (case == c) & (centre == flag)
            for e0, e1 in pairs:
                out.append(np.stack([edges[e0][m], edges[e1][m]], axis=1))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2, 2))


def regions_below(field: np.ndarray, xy: np.ndarray, frame: Polygon, level: float = 0.0) -> MultiPolygon:
    """The parts of ``frame`` where ``field`` < ``level`` (bilinear between samples),
    as polygons: the level's contours closed along the frame's edge.

    Samples are texel centres, so the outermost half texel is decided by its
    nearest sample's sign.
    """
    f = np.asarray(field, dtype=np.float64)
    if (f >= level).all():
        return MultiPolygon()
    if (f < level).all():
        return MultiPolygon([frame])
    # Pad by one ring outside the frame so contours reach its edge.
    step = xy[0, 1, 0] - xy[0, 0, 0]
    fp = np.pad(f, 1, mode="edge")
    T = f.shape[0]
    c = xy[0, :, 0]
    cx = np.concatenate([[c[0] - step], c, [c[-1] + step]])
    cy = cx[::-1]
    xx, yy = np.meshgrid(cx, cy)
    segs = contour_segments(fp, np.stack([xx, yy], axis=-1), level)
    lines = [LineString(s) for s in segs]
    lines.append(frame.exterior)
    faces = list(polygonize(unary_union(lines)))
    if not faces:
        return MultiPolygon()
    # Each face is wholly above or below the level: read the field (bilinear) at a point inside it.
    pts = np.array([face.representative_point().coords[0] for face in faces])
    fx = np.clip((pts[:, 0] - cx[0]) / step, 0.0, T + 1 - 1e-9)
    fy = np.clip((cy[0] - pts[:, 1]) / step, 0.0, T + 1 - 1e-9)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    x1, y1 = np.minimum(x0 + 1, T + 1), np.minimum(y0 + 1, T + 1)
    tx, ty = fx - x0, fy - y0
    value = ((1 - tx) * (1 - ty) * fp[y0, x0] + tx * (1 - ty) * fp[y0, x1]
             + (1 - tx) * ty * fp[y1, x0] + tx * ty * fp[y1, x1])
    inside = frame.buffer(1e-6)
    below = [face for face, val in zip(faces, value) if val < level and face.within(inside)]
    merged = shapely.union_all(below) if below else MultiPolygon()
    if isinstance(merged, Polygon):
        return MultiPolygon([merged])
    return merged if isinstance(merged, MultiPolygon) else MultiPolygon(
        [g for g in getattr(merged, "geoms", []) if isinstance(g, Polygon)])
