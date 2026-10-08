"""Region / local border lines: contoured from the view's own fill, following the rivers it draws."""
from __future__ import annotations

import numpy as np
import pytest

from realmgen.polity.types import Polity
from realmgen.refine.barriers import pack_river_segments
from realmgen.refine.borders import _icosphere_lattice, trace_lattice_chains
from realmgen.refine.fill import PolityLabelField
from realmgen.refine.river_follow import RiverFollowParams, build_river_swaps
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
    field = PolityLabelField(mesh, pof, np.zeros(n, bool), np.ones(n))
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
    """A border moved onto a world river follows the view's re-routed drawing of it."""
    field, axis, centre = split
    tol = 2.0 * fields.grid.texel_spacing_rad(fields.tex_size)

    def share_on(segs, offset):
        segs = segs.astype(np.float64)
        mid = segs[:, :3] + segs[:, 3:]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        length = np.linalg.norm(segs[:, 3:] - segs[:, :3], axis=1)
        return length[np.abs(mid @ axis - offset) < tol].sum() / length.sum()

    pols = [_pol(0), _pol(1)]
    verts, tris, edges, tri_edges = _icosphere_lattice(5)
    chains = trace_lattice_chains(verts, tris, edges, tri_edges, field.plain_land_labels(verts), pols,
                                  label_fn=field.plain_land_labels)
    # Coarse world (faces ~1000 km): reach and sampling scaled up to match.
    params = RiverFollowParams(reach_km=0.15 * EARTH_RADIUS_KM, min_flow=1000.0, full_flow=8000.0,
                               min_run_km=200.0, sample_km=20.0, border_step_km=10.0,
                               view_reach_km=0.08 * EARTH_RADIUS_KM)
    swaps = build_river_swaps(chains, [_river(axis, centre, 0.03)], field.plain_land_labels, field.is_water_at,
                              np.zeros((0, 3)), field.centroids, params)
    moved = field.with_swaps(swaps)
    drawn = _river(axis, centre, 0.07)                       # the view's drawing of world river 0
    ids = [np.zeros(len(drawn[0]), dtype=np.int64)]
    on_drawn = view_polity_borders(fields, moved, pols, pack_river_segments([drawn], river_ids=ids))
    on_world = view_polity_borders(fields, moved, pols, None)
    assert share_on(on_drawn[0], 0.07) > 0.9
    assert share_on(on_world[0], 0.03) > 0.9
    assert share_on(on_world[0], 0.07) < 0.5
