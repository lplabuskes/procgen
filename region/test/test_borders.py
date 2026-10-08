"""Region / local border lines: contoured from the view's own fill, snapped to the rivers it draws."""
from __future__ import annotations

import numpy as np
import pytest

from realmgen.polity.types import Polity
from realmgen.refine.barriers import pack_river_segments
from realmgen.refine.fill import PolityLabelField
from region.borders import _square_lattice, view_polity_borders
from region.upsample import RegionUpsampleParams, build_region_fields
from worldgen.cost.surfaces import EARTH_RADIUS_KM

REGION_ID = 40   # has land (the conftest lake sits on its highland)


def _pol(pid):
    return Polity(polity_id=pid, capital_face=pid, capital_position=np.array([1.0, 0.0, 0.0]),
                  power=10.0, n_faces=1, member_settlement_faces=[pid])


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    return build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16))


@pytest.fixture(scope="module")
def split(synthetic_globe, fields):
    """Two polities split by the great circle through the cell centre, east–west; that circle's axis."""
    mesh = synthetic_globe.mesh
    centre = fields.geometry.center_ecef()
    east = fields.geometry.local_to_ecef(np.array([0.6]), np.array([0.5]))[0] - centre
    axis = east - (east @ centre) * centre
    axis /= np.linalg.norm(axis)
    c = np.asarray(mesh.face_centroids, dtype=np.float64)
    pof = np.where(c @ axis > 0.0, 0, 1)
    n = len(pof)
    # Snapping on but no world rivers: plain until a view swaps its rivers in.
    field = PolityLabelField(mesh, pof, np.zeros(n, bool), np.ones(n),
                             river_reach_km=0.25 * EARTH_RADIUS_KM, river_snap_fraction=1.0)
    return field, axis, centre


def _river(axis, centre, offset):
    """A big river along the small circle ``x · axis = offset`` through the cell."""
    north = np.cross(axis, centre)
    ang = np.linspace(-0.6, 0.6, 400)
    pts = (np.cos(ang)[:, None] * centre + np.sin(ang)[:, None] * north) * np.sqrt(1 - offset ** 2) \
        + offset * axis
    return (pts / np.linalg.norm(pts, axis=1, keepdims=True), np.full(ang.size - 1, 20000.0))


def test_square_lattice():
    st, tris = _square_lattice(4)
    assert st.shape == (25, 2) and tris.shape == (32, 3)
    assert st.min() == 0.0 and st.max() == 1.0


def test_lines_bound_the_fill(fields, split):
    field, _, _ = split
    borders = view_polity_borders(fields, field, [_pol(0), _pol(1)], None)
    assert set(borders) == {0}
    segs = borders[0]
    a, b = segs[:, :3].astype(np.float64), segs[:, 3:].astype(np.float64)
    mid = a + b
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    normal = np.cross(mid, b - a)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    step = 3.0 * fields.grid.texel_spacing_rad(fields.tex_size)
    differ = field.land_labels(mid + step * normal) != field.land_labels(mid - step * normal)
    assert differ.mean() > 0.9


def test_lines_follow_the_rivers_the_view_draws(fields, split):
    field, axis, centre = split
    tol = 2.0 * fields.grid.texel_spacing_rad(fields.tex_size)

    def share_on_river(segs):
        segs = segs.astype(np.float64)
        mid = segs[:, :3] + segs[:, 3:]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        length = np.linalg.norm(segs[:, 3:] - segs[:, :3], axis=1)
        return length[np.abs(mid @ axis - 0.03) < tol].sum() / length.sum()

    river = _river(axis, centre, 0.03)
    snapped = view_polity_borders(fields, field, [_pol(0), _pol(1)], pack_river_segments([river]))
    plain = view_polity_borders(fields, field, [_pol(0), _pol(1)], None)
    assert share_on_river(snapped[0]) > 0.9
    assert share_on_river(plain[0]) < 0.5
