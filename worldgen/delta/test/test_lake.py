"""Lake deltas on the synthetic globe of test_deposit: a round lake on the land,
the river running through it and on down to the sea."""
from __future__ import annotations

import copy

import numpy as np

from worldgen.delta.builder import DeltaParams
from worldgen.delta.deposit import TexelIndex, build_deltas, lake_mouths
from worldgen.delta.shape import WaveField
from worldgen.delta.test.test_deposit import EAST, TEX, UP, _textures
from worldgen.hydrology.types import River, RiverNode, RiverSegment, StreamNetwork

FLOW = 5000.0
# The lake: a cap of this angular radius, centred this far north of the coast.
LAKE_LAT, LAKE_RADIUS = 0.2, 0.08
LEVEL, BED = 0.3, 0.27


def _lake_textures(mesh):
    elev, ocean, lake, filled = _textures(mesh)
    pos = mesh.face_texture_positions(TEX, TEX).astype(np.float64)
    centre = np.cos(LAKE_LAT) * EAST + np.sin(LAKE_LAT) * UP
    wet = np.arccos(np.clip(pos @ centre, -1.0, 1.0)) < LAKE_RADIUS
    # Raise the land around the lake to hold it, and flood it to LEVEL.
    near = np.arccos(np.clip(pos @ centre, -1.0, 1.0)) < 2.0 * LAKE_RADIUS
    elev[..., 0] = np.where(near & ~wet & (ocean[..., 0] == 0), np.maximum(elev[..., 0], LEVEL + 0.01), elev[..., 0])
    elev[..., 0] = np.where(wet, BED, elev[..., 0])
    lake[..., 0] = wet.astype(np.uint8)
    filled[..., 0] = np.where(wet, LEVEL, elev[..., 0])
    return elev, ocean, lake, filled


def _river(index: TexelIndex, ocean: np.ndarray, lake: np.ndarray, flow: float) -> River:
    """Down the EAST meridian, through the lake, ending on the first sea texel."""
    pts, wet = [], []
    for lat in np.linspace(0.5, -0.2, 300):
        p = np.cos(lat) * EAST + np.sin(lat) * UP
        flat = int(index.nearest_inside(p[None, :])[0])
        pts.append(p)
        wet.append(bool(lake.reshape(-1)[flat]))
        if ocean.reshape(-1)[flat]:
            break
    nodes = []
    for p in pts:
        face, u, v = index.face_uv(int(index.nearest_inside(p[None, :])[0]))
        nodes.append(RiverNode(position=p.astype(np.float32), elevation=0.0, face_id=face, uv=(u, v)))
    segs = [RiverSegment(start=a, end=b, flow_rate=flow, in_lake=wet[i] or wet[i + 1])
            for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:]))]
    return River(segments=segs, source=nodes[0], mouth=nodes[-1])


def _streams(index: TexelIndex, river: River, flow: float = FLOW) -> StreamNetwork:
    """The river's own texels as one chain, ending on its sea texel."""
    flats = index.nearest_inside(np.stack(river.as_polyline()).astype(np.float64))
    keep = np.concatenate([[True], flats[1:] != flats[:-1]])
    flats = flats[keep]
    _, first = np.unique(flats, return_index=True)
    flats = flats[np.sort(first)]
    n = len(flats)
    receiver = np.append(np.arange(1, n), -1).astype(np.int64)
    return StreamNetwork(
        nodes=flats.astype(np.int64), positions=index.positions[flats].astype(np.float32), receiver=receiver,
        flow=np.full(n, flow, dtype=np.float32), channel=np.append(np.ones(n - 1, bool), False),
        stop=np.append(np.zeros(n - 1, bool), True), resolution=TEX,
    )


def _run(mesh, index, params=None, streams=False, flow=FLOW):
    elev, ocean, lake, filled = _lake_textures(mesh)
    before = elev.copy(), ocean.copy(), lake.copy()
    river = _river(index, ocean, lake, flow)
    original = copy.deepcopy(river)
    net = _streams(index, river, flow) if streams else None
    result = build_deltas(mesh, [river], elev, ocean, lake, filled, net, params or DeltaParams())
    return result, before, (elev, ocean, lake, filled), river, original, net


def test_a_river_entering_a_lake_is_a_lake_mouth(mesh, index):
    _, ocean, lake, _ = _lake_textures(mesh)
    river = _river(index, ocean, lake, FLOW)
    mouths = lake_mouths([river], DeltaParams())
    assert len(mouths) == 1 and mouths[0].lake
    k = mouths[0].end()
    assert river.segments[k - 1].in_lake and not river.segments[k - 2].in_lake
    assert lake_mouths([river], DeltaParams(min_flow=2 * FLOW)) == []


def test_a_delta_is_built_into_the_lake(mesh, index):
    result, (elev0, ocean0, lake0), (elev, ocean, lake, filled), *_ = _run(mesh, index)
    lake_fans = [f for f in result.fans if f.lake]
    sea_fans = [f for f in result.fans if not f.lake]
    assert len(lake_fans) == 1 and len(sea_fans) == 1
    plain = result.delta[..., 0] > 0
    filled_in = (lake0[..., 0] > 0) & (lake[..., 0] == 0)
    assert filled_in.sum() > 0
    assert np.all(plain[filled_in])
    # The lake plain stands on the lake's surface, not sea level.
    p = DeltaParams()
    on_lake = plain & (np.abs(elev0[..., 0] - BED) < 1e-6)
    assert on_lake.any()
    assert elev[..., 0][on_lake].min() >= LEVEL + p.front_height_km - 1e-4
    assert elev[..., 0][on_lake].max() <= LEVEL + p.apex_height_km + 1e-4
    assert np.allclose(filled[..., 0][on_lake], elev[..., 0][on_lake])
    # Lake texels never become sea, and the lake's open water only shoals.
    assert np.all(ocean[lake0 > 0] == 0)
    still = (lake > 0)[..., 0]
    assert np.all(elev[..., 0][still] >= elev0[..., 0][still] - 1e-7)
    assert np.all(elev[..., 0][still] < LEVEL)


def test_the_lake_is_never_filled_past_its_share(mesh, index):
    inside = mesh.face_texture_in_polygon(TEX, TEX)
    for share in (0.05, 0.25):
        result, (_, _, lake0), (_, _, lake, _), *_ = _run(mesh, index, DeltaParams(max_lake_share=share))
        before = int(((lake0[..., 0] > 0) & inside).sum())
        lost = int(((lake0[..., 0] > 0) & (lake[..., 0] == 0) & inside).sum())
        assert 0 < lost <= share * before + 2
    # No room at all: no lake delta.
    result, *_ = _run(mesh, index, DeltaParams(max_lake_share=0.0))
    assert not any(f.lake for f in result.fans)


def test_the_river_crosses_the_lake_plain_and_runs_on(mesh, index):
    result, _, (_, ocean, lake, _), river, original, _ = _run(mesh, index)
    k = lake_mouths([original], DeltaParams())[0].end()
    # Upstream of the lake nothing changed.
    for a, b in zip(original.as_polyline()[:k + 1], river.as_polyline()[:k + 1]):
        assert np.allclose(a, b)
    pts = np.stack(river.as_polyline()).astype(np.float64)
    # Continuous, and still ending at sea.
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert steps.max() < 3.0 * LAKE_RADIUS
    assert ocean.reshape(-1)[int(index.nearest_inside(pts[-1:])[0])] > 0
    # Its reach over the lake plain is drawn (not hidden as in-lake), and what
    # it still runs through of the lake is.
    plain = result.delta[..., 0].reshape(-1) > 0
    flats = index.nearest_inside(pts)
    seg_plain = plain[flats[1:]] & plain[flats[:-1]] & (lake.reshape(-1)[flats[1:]] == 0)
    in_lake = np.array([s.in_lake for s in river.segments])
    lake_fan = next(f for f in result.fans if f.lake)
    near = np.arccos(np.clip(pts[1:] @ np.asarray(lake_fan.apex, dtype=np.float64), -1, 1)) * 6371.0 \
        < lake_fan.radius_km
    assert (seg_plain & near & ~in_lake).any()
    assert in_lake.any()


def test_the_stream_network_crosses_the_lake_plain(mesh, index):
    # Big enough that the plain reaches well out over the river's old course in the lake.
    big = 20 * FLOW
    result, (_, _, lake0), (_, ocean, _, _), _, _, net = _run(mesh, index, streams=True, flow=big)
    assert net is not None
    assert len(net.nodes) == len(net.positions) == len(net.flow) == len(net.channel) == len(net.stop)
    assert np.all(net.receiver < len(net.nodes))
    # From the headwater the chain still reaches the sea, across the lake plain.
    k, seen = 0, [0]
    while net.receiver[k] >= 0:
        k = int(net.receiver[k])
        assert k not in seen
        seen.append(k)
    assert net.stop[k] and ocean.reshape(-1)[int(net.nodes[k])] > 0
    lake_plain = (result.delta[..., 0] > 0).reshape(-1) & (lake0.reshape(-1) > 0)
    assert lake_plain[net.nodes[seen]].any()
    # The river's old course over what was lake is gone: no other node there
    # still carries its flow (it would out-rank the new reach as the main stem).
    orphans = np.setdiff1d(np.flatnonzero(lake_plain[net.nodes] & (net.flow >= big)), seen)
    assert orphans.size == 0


def test_lake_waves_have_no_swell_and_short_fetch():
    # Random points on the sphere: a small "lake" (label 1) around the mouth, land beyond.
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(20000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    mouth = np.array([0.0, 0.0, 1.0])
    ang = np.arccos(np.clip(pts @ mouth, -1, 1))
    labels = np.where(ang < 0.15, 1, -1)
    spacing = 0.03   # about the sample points' own spacing
    p = DeltaParams()
    calm = WaveField(pts, labels, None, spacing, step_rad=spacing, start_rad=spacing)
    assert calm.power(mouth, np.array([1.0, 0.0, 0.0]), p, body=1, swell=False) == 0.0
    swell = calm.power(mouth, np.array([1.0, 0.0, 0.0]), p, body=1, swell=True)
    sea = WaveField(pts, np.where(ang < 1.0, 1, -1), None, spacing, step_rad=spacing, start_rad=spacing)
    assert 0.0 < swell < sea.power(mouth, np.array([1.0, 0.0, 0.0]), p, body=1, swell=True)
