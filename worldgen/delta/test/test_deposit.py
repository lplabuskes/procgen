"""Delta deposition on a synthetic globe: a sloping hemisphere of land, a shelf
deepening offshore, and one hand-made river running straight down to the sea."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from core.mesh import Mesh
from worldgen.delta.builder import DeltaParams
from worldgen.delta.deposit import REFERENCE_FLOW, TexelIndex, build_deltas
from worldgen.hydrology.types import River, RiverNode, RiverSegment, StreamNetwork

TEX = 16
# The coast is the great circle ⟂ to UP; land rises inland, the sea deepens offshore.
UP = np.array([0.0, 0.0, 1.0])
EAST = np.array([1.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def mesh():
    return Mesh.icosphere(4).mesh_dual()


@pytest.fixture(scope="module")
def index(mesh):
    return TexelIndex(mesh, TEX, TEX)


def _textures(mesh, land_km: float = 0.05):
    pos = mesh.face_texture_positions(TEX, TEX).astype(np.float64)
    h = pos @ UP
    elev = np.where(h >= 0.0, land_km + 0.5 * h, -0.02 + 1.5 * h).astype(np.float32)[..., None]
    ocean = (elev < 0.0).astype(np.uint8)
    lake = np.zeros_like(ocean)
    filled = elev.copy()
    return elev, ocean, lake, filled


def _river(index: TexelIndex, ocean: np.ndarray, flow: float) -> River:
    """Straight down the slope along the EAST meridian, ending on the first sea texel."""
    pts = []
    for lat in np.linspace(0.5, -0.2, 200):
        p = np.cos(lat) * EAST + np.sin(lat) * UP
        pts.append(p)
        flat = int(index.nearest_inside(p[None, :])[0])
        if ocean.reshape(-1)[flat]:
            break
    nodes = []
    for p in pts:
        flat = int(index.nearest_inside(p[None, :])[0])
        face, u, v = index.face_uv(flat)
        nodes.append(RiverNode(position=p.astype(np.float32), elevation=0.0, face_id=face, uv=(u, v)))
    segs = [RiverSegment(start=a, end=b, flow_rate=flow) for a, b in zip(nodes[:-1], nodes[1:])]
    return River(segments=segs, source=nodes[0], mouth=nodes[-1])


def _streams(index: TexelIndex, river: River) -> StreamNetwork:
    """A two-node stream network: the river's last land texel draining into its mouth texel."""
    pts = np.stack(river.as_polyline()[-2:]).astype(np.float32)
    flats = index.nearest_inside(pts)
    return StreamNetwork(
        nodes=flats.astype(np.int64), positions=pts, receiver=np.array([1, -1], dtype=np.int64),
        flow=np.full(2, 5000.0, dtype=np.float32), channel=np.array([True, False]),
        stop=np.array([False, True]), resolution=TEX,
    )


def _run(mesh, index, flow=5000.0, params=None, land_km=0.05, streams=True):
    elev, ocean, lake, filled = _textures(mesh, land_km)
    before = elev.copy(), ocean.copy()
    river = _river(index, ocean, flow)
    original = copy.deepcopy(river)
    net = _streams(index, river) if streams else None
    result = build_deltas(mesh, [river], elev, ocean, lake, filled, net, params or DeltaParams())
    return result, before, (elev, ocean, filled), river, original, net


def test_a_delta_builds_land_into_the_sea(mesh, index):
    result, (elev0, ocean0), (elev, ocean, filled), *_ = _run(mesh, index)
    assert len(result.fans) == 1
    built = (ocean0 > 0) & (ocean == 0)
    assert built.sum() > 0
    plain = result.delta[..., :1] > 0
    # The plain is dry, low and flat — between the front and apex heights.
    p = DeltaParams()
    assert np.all(ocean[plain] == 0)
    assert elev[plain].min() >= p.front_height_km - 1e-6
    assert elev[plain].max() <= p.apex_height_km + 1e-6
    assert np.allclose(filled[plain], elev[plain])
    # Everything new is on the plain; nothing else changed sides.
    assert np.all(plain[built])
    assert np.all(ocean[~plain] == ocean0[~plain])


def test_the_prodelta_only_shoals(mesh, index):
    _, (elev0, ocean0), (elev, ocean, _), *_ = _run(mesh, index)
    sea = (ocean > 0) & (ocean0 > 0)
    assert np.all(elev[sea] >= elev0[sea] - 1e-7)
    assert np.any(elev[sea] > elev0[sea])
    assert np.all(elev[sea] < 0.0)


def test_area_grows_with_flow(mesh, index):
    small = _run(mesh, index, flow=REFERENCE_FLOW)[0].fans[0]
    big = _run(mesh, index, flow=30 * REFERENCE_FLOW)[0].fans[0]
    assert big.area_km2 > 3.0 * small.area_km2
    assert big.radius_km > small.radius_km
    # The budget is met to within a texel or so.
    p = DeltaParams()
    texel = float(np.mean(index.texel_area_km2[index.ids]))
    assert abs(small.area_km2 - p.area_km2) < 2.0 * texel


def test_hills_are_not_flattened(mesh, index):
    # Land everywhere far above the flattening limit: the plain is all built land.
    result, (elev0, ocean0), (elev, ocean, _), *_ = _run(mesh, index, land_km=1.0)
    plain = result.delta[..., :1] > 0
    assert plain.any()
    assert np.all(ocean0[plain] > 0)


def test_the_river_is_carried_to_the_new_front(mesh, index):
    result, _, (elev, ocean, _), river, original, _ = _run(mesh, index)
    n0 = len(original.segments)
    assert len(river.segments) > n0
    for a, b in zip(original.as_polyline(), river.as_polyline()[: n0 + 1]):
        assert np.allclose(a, b)
    end = np.asarray(river.as_polyline()[-1])[None, :]
    assert ocean.reshape(-1)[int(index.nearest_inside(end)[0])] > 0
    assert np.allclose(result.fans[0].mouth, end[0], atol=1e-6)
    assert all(s.flow_rate == original.segments[-1].flow_rate for s in river.segments[n0:])


def test_the_stream_network_reaches_the_new_front(mesh, index):
    *_, (elev, ocean, _), river, _, net = _run(mesh, index)
    assert net is not None
    # The old mouth is no longer an outlet; following it downstream ends at sea.
    assert not net.stop[1] and net.channel[1]
    k, steps = 1, 0
    while net.receiver[k] >= 0:
        k = int(net.receiver[k])
        steps += 1
    assert steps > 0 and net.stop[k]
    assert ocean.reshape(-1)[int(net.nodes[k])] > 0
    assert len(net.nodes) == len(net.positions) == len(net.flow) == len(net.channel) == len(net.stop)


def test_overhanging_texels_copy_the_texel_they_overlap(mesh, index):
    # Big enough to spread over several faces' over-hangs.
    result, _, (elev, ocean, _), *_ = _run(mesh, index, flow=300 * REFERENCE_FLOW)
    delta = result.delta[..., 0].reshape(-1)
    apex = np.asarray(result.fans[0].apex, dtype=np.float64)
    near = index.near(apex, 2.0 * result.fans[0].radius_km)
    overhang = near[~index.inside[near]]
    assert (delta[overhang] > 0).any()
    role = index.nearest_inside(index.positions[overhang])
    assert np.array_equal(delta[overhang], delta[role])
    assert np.array_equal(ocean.reshape(-1)[overhang], ocean.reshape(-1)[role])


def test_face_fractions(mesh, index):
    result, *_ = _run(mesh, index)
    plain, marsh = result.plain_fraction, result.marsh_fraction
    inside = mesh.face_texture_in_polygon(TEX, TEX)
    expected = (result.delta[..., 0] * inside).sum(axis=(1, 2)) / inside.sum(axis=(1, 2))
    assert np.allclose(plain, expected)
    assert plain.max() > 0 and np.all((plain >= 0) & (plain <= 1))
    assert np.all(marsh <= plain + 1e-9)


def test_small_or_short_rivers_get_no_delta(mesh, index):
    result, (elev0, ocean0), (elev, ocean, _), *_ = _run(
        mesh, index, params=DeltaParams(min_flow=1e9), streams=False)
    assert result.fans == []
    assert np.array_equal(elev, elev0) and np.array_equal(ocean, ocean0)


def test_builder_round_trips_its_fans(mesh, index):
    from core.pipeline import BuildContext
    from core.world import World
    from worldgen.delta.builder import DeltaBuilder

    result, *_ = _run(mesh, index)
    builder = DeltaBuilder()
    builder.fans = result.fans
    arrays = builder.serialize()
    world = World.from_mesh(mesh)
    loaded = DeltaBuilder()
    loaded.load(arrays, BuildContext(world=world))
    assert len(loaded.fans) == len(result.fans)
    for a, b in zip(loaded.fans, result.fans):
        assert np.allclose(a.apex, b.apex) and np.allclose(a.mouth, b.mouth) and np.allclose(a.axis, b.axis)
        assert np.isclose(a.flow, b.flow) and np.isclose(a.radius_km, b.radius_km)
    assert np.allclose(world.get_feature("deltas"), np.stack([f.mouth for f in result.fans]))
    assert world.get_feature("delta_fans") is loaded.fans
