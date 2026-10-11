"""Regional roads: the world's trade routes re-routed over the refined terrain."""
from dataclasses import replace
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from region.rivers import build_region_rivers
from realmgen.trade.network import ROUTE_LAND, ROUTE_TRANSFER, ROUTE_WATER
from region.roads import RegionRoadParams, _chains, _pull_string, build_region_roads
from region.sampling import GlobalSampler
from region.upsample import RegionUpsampleParams, build_region_fields

REGION_ID = 40   # coast, a lake and a peak (see conftest)
HIGH, BRANCH, PORT, LOW = 10.0, 6.0, 8.0, 1.0


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    return build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16))


def _bfs(mesh, allowed: set, a: int, b: int) -> list[int]:
    """Shortest face path from ``a`` to ``b`` through ``allowed`` faces."""
    prev = {a: -1}
    queue = [a]
    for f in queue:
        if f == b:
            break
        for g in mesh.face_adjacency[f]:
            if g in allowed and g not in prev:
                prev[g] = f
                queue.append(g)
    path = [b]
    while prev[path[-1]] >= 0:
        path.append(prev[path[-1]])
    return path[::-1]


@pytest.fixture(scope="module")
def world(fields, synthetic_globe):
    """A trunk road across the region's land faces to a coastal face, a branch,
    a port hop onto the sea, and weak hops elsewhere that set the drawing
    threshold."""
    globe = synthetic_globe
    mesh = globe.mesh
    c = _unit(mesh.face_centroids)
    ocean = mesh.face_textures["ocean_mask"][..., 0].mean(axis=(1, 2)) > 0.5
    lake = mesh.face_textures["lake_mask"][..., 0].mean(axis=(1, 2)) > 0.1
    land = ~ocean & ~lake
    s, t, valid = fields.geometry.ecef_to_local(c)
    inside = valid & (s > 0.05) & (s < 0.95) & (t > 0.05) & (t < 0.95)
    inland = {int(f) for f in np.flatnonzero(inside & land)}
    # A coastal face with a sea neighbour in the cell, and the furthest inland face.
    coast = next(f for f in sorted(inland) if any(ocean[g] and inside[g] for g in mesh.face_adjacency[f]))
    sea = next(g for g in mesh.face_adjacency[coast] if ocean[g] and inside[g])
    start = max(inland, key=lambda f: np.linalg.norm(c[f] - c[coast]))
    trunk = _bfs(mesh, inland, start, coast)
    assert len(trunk) >= 3
    start = coast
    routes = [(c[a], c[b], HIGH) for a, b in zip(trunk, trunk[1:])]
    branch = [g for g in mesh.face_adjacency[trunk[1]] if g in inland and g not in trunk]
    if branch:
        routes.append((c[trunk[1]], c[branch[0]], BRANCH))
    routes.append((c[coast], c[sea], PORT))
    far = np.flatnonzero(c @ c[start] < -0.5)[:40]
    routes += [(c[a], c[b], LOW) for a, b in zip(far[:-1], far[1:])]
    globe.add_feature("trade_routes", routes)
    globe.add_feature("cost_surfaces", SimpleNamespace(land_passable=~(ocean | lake)))
    yield SimpleNamespace(trunk=trunk, start=start, sea=sea, routes=routes, c=c)
    globe.remove_feature("trade_routes")
    globe.remove_feature("cost_surfaces")


@pytest.fixture(scope="module")
def roads(fields, synthetic_globe, world):
    rivers = build_region_rivers(fields, synthetic_globe)
    return build_region_roads(fields, synthetic_globe, rivers)


def _refined_water(fields, pts: np.ndarray) -> np.ndarray:
    s, t, valid = fields.geometry.ecef_to_local(pts)
    hexes, rows, cols, inside = fields.grid.texels_at(s, t, fields.tex_size)
    wet = (fields.textures["ocean_mask"][..., 0] > 0.5) | (fields.textures["lake_mask"][..., 0] > 0.5)
    return valid & inside & wet[hexes, rows, cols]


def test_the_drawn_routes_are_rerouted(roads, world):
    assert roads.polylines
    assert all(len(p) > 2 for p in roads.polylines)
    flows = {round(float(f), 6) for _, _, f in roads.segments}
    assert flows <= {HIGH, BRANCH, PORT}
    assert HIGH in flows and PORT in flows


def test_weak_routes_are_kept_straight_for_the_towns(roads):
    assert all(f == LOW for _, _, f in roads.undrawn)


def test_roads_stay_on_land(fields, roads):
    for path in roads.polylines:
        # The ends may be a settlement the view moves off the water itself.
        assert _refined_water(fields, path[1:-1]).mean() < 0.05


def test_roads_end_on_their_world_endpoints(roads, world):
    ends = np.array([p[k] for p in roads.polylines for k in (0, -1)])
    trunk_end = world.c[world.trunk[0]]
    assert np.linalg.norm(ends - trunk_end, axis=1).min() < 1e-6


def test_roads_stay_in_their_corridor(synthetic_globe, roads, world):
    face = float(np.sqrt(np.mean(synthetic_globe.mesh.face_areas)))
    corridor = RegionRoadParams().corridor_face_fraction * face
    line = np.concatenate([
        np.linspace(a, b, 50) for a, b, f in world.routes if f > LOW
    ])
    line = _unit(line)
    for path in roads.polylines:
        d = np.linalg.norm(path[:, None, :] - line[None, :, :], axis=-1).min(axis=1)
        assert d.max() < 1.5 * corridor


def test_the_port_hop_is_split_at_the_refined_coast(fields, roads, world):
    """The land part is routed; a straight sea leg runs from the coast to the
    water face."""
    legs = [(a, b) for a, b, f in roads.sea if f == PORT]
    assert len(legs) == 1
    coast, water = (np.asarray(p, dtype=np.float64) for p in legs[0])
    assert np.linalg.norm(water - world.c[world.sea]) < 1e-6
    assert not _refined_water(fields, coast[None])[0]


def test_roads_avoid_danger(fields, synthetic_globe, world):
    """A dangerous patch on the trunk's straight line pushes the road off it."""
    mid = _unit(world.c[world.trunk[1]] + world.c[world.trunk[2]])
    radius = 0.06

    def patch(pts: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - np.linalg.norm(pts - mid, axis=-1) / radius, 0.0, 1.0)

    with_danger = dict(fields.textures, danger=patch(fields.positions)[..., None].astype(np.float32))
    calm = build_region_roads(fields, synthetic_globe, None)
    wary = build_region_roads(replace(fields, textures=with_danger), synthetic_globe, None, danger_scale=50.0)

    def exposure(r) -> float:
        return float(patch(np.concatenate(r.polylines)).mean())

    assert exposure(calm) > 0.0
    assert exposure(wary) < 0.5 * exposure(calm)


def test_generation_is_deterministic(fields, synthetic_globe, world, roads):
    again = build_region_roads(fields, synthetic_globe, build_region_rivers(fields, synthetic_globe))
    assert len(again.segments) == len(roads.segments)
    for (a, b, f), (c, d, g) in zip(again.segments, roads.segments):
        assert np.array_equal(a, c) and np.array_equal(b, d) and f == g


def test_no_routes_no_roads(fields, synthetic_globe, world):
    synthetic_globe.remove_feature("trade_routes")
    try:
        assert build_region_roads(fields, synthetic_globe).segments == []
    finally:
        synthetic_globe.add_feature("trade_routes", world.routes)


def test_chains_cut_at_junctions_and_loops():
    # A Y: 0-1-2 with a branch 1-3, plus a separate triangle 4-5-6.
    edges = [(0, 1, 1.0), (1, 2, 2.0), (1, 3, 3.0), (4, 5, 4.0), (5, 6, 5.0), (6, 4, 6.0)]
    chains = _chains(edges, 7)
    got = sorted(tuple(sorted(f)) for _, f in chains)
    assert got == [(1.0,), (2.0,), (3.0,), (4.0, 5.0, 6.0)]
    loop = next(n for n, f in chains if len(f) == 3)
    assert loop[0] == loop[-1]


@contextmanager
def _features(globe, **features):
    """Temporarily replace ``globe``'s features (None removes one)."""
    old = {k: globe.get_feature(k) for k in features}
    try:
        for k, v in features.items():
            globe.remove_feature(k) if v is None else globe.add_feature(k, v)
        yield
    finally:
        for k, v in old.items():
            globe.remove_feature(k) if v is None else globe.add_feature(k, v)


def test_known_layers_split_the_port_hop_the_same(fields, synthetic_globe, world):
    """With the world's layers the trunk is road and the port hop onto the sea is
    split at the coast, as when they are guessed from the faces."""
    layers = np.array([ROUTE_TRANSFER if f == PORT else ROUTE_WATER if f == LOW else ROUTE_LAND
                       for _, _, f in world.routes], dtype=np.int8)
    with _features(synthetic_globe, trade_route_layers=layers):
        roads = build_region_roads(fields, synthetic_globe, build_region_rivers(fields, synthetic_globe))
    assert roads.polylines and not roads.water_polylines
    legs = [(a, b) for a, b, f in roads.sea if f == PORT]
    assert len(legs) == 1
    assert np.linalg.norm(np.asarray(legs[0][1], dtype=np.float64) - world.c[world.sea]) < 1e-6


@pytest.fixture(scope="module")
def river_hops(fields, synthetic_globe):
    """World hops face to face down the region's longest regional river."""
    rivers = build_region_rivers(fields, synthetic_globe)
    path = max((np.asarray(p, dtype=np.float64) for p, _, _ in rivers.paths), key=len)
    faces, _, _ = GlobalSampler(synthetic_globe.mesh).locate(path)
    seq = [int(f) for k, f in enumerate(faces) if k == 0 or f != faces[k - 1]]
    assert len(seq) >= 3
    c = _unit(synthetic_globe.mesh.face_centroids)
    routes = [(c[a], c[b], HIGH) for a, b in zip(seq, seq[1:])]
    return SimpleNamespace(rivers=rivers, path=path, routes=routes)


def _river_roads(fields, synthetic_globe, river_hops, layer: int):
    mesh = synthetic_globe.mesh
    ocean = mesh.face_textures["ocean_mask"][..., 0].mean(axis=(1, 2)) > 0.5
    lake = mesh.face_textures["lake_mask"][..., 0].mean(axis=(1, 2)) > 0.1
    with _features(synthetic_globe, trade_routes=river_hops.routes,
                   trade_route_layers=np.full(len(river_hops.routes), layer, dtype=np.int8),
                   cost_surfaces=SimpleNamespace(land_passable=~(ocean | lake))):
        return build_region_roads(fields, synthetic_globe, river_hops.rivers)


def test_river_hops_follow_the_river_not_a_road(fields, synthetic_globe, river_hops):
    roads = _river_roads(fields, synthetic_globe, river_hops, ROUTE_WATER)
    assert roads.water_polylines and not roads.polylines
    assert roads.sea == []
    assert len(roads.water_pins) == len(roads.water_polylines)
    pts = np.concatenate([p[1:-1] for p in roads.water_polylines])
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    off_river = np.linalg.norm(pts[:, None, :] - river_hops.path[None, :, :], axis=-1).min(axis=1)
    # Smoothing pulls a little off the raster nodes on the river (two texels apart).
    on = (off_river < 3.0 * spacing) | _refined_water(fields, pts)
    assert on.mean() > 0.9
    assert len(roads.segments) == sum(len(p) - 1 for p in roads.water_polylines)


def test_the_same_hops_by_land_are_a_road(fields, synthetic_globe, river_hops):
    roads = _river_roads(fields, synthetic_globe, river_hops, ROUTE_LAND)
    assert roads.polylines and not roads.water_polylines


def test_water_legs_sail_straight_across_open_water_but_keep_river_bends():
    n = 12
    raster: Any = SimpleNamespace(n=n)          # only its size is read
    # A staircase (as a grid path crosses water) from (0, 0) to (6, 11).
    rows = [0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6]
    stair = np.array([r * n + c for r, c in zip(rows, range(n))])
    open_water = np.ones(n * n, dtype=bool)
    pulled = _pull_string(raster, stair, open_water)
    assert pulled[0] == stair[0] and pulled[-1] == stair[-1]
    r, c = np.divmod(pulled, n)
    # Straight from end to end (the staircase strays 2.7 rows from the line).
    assert np.abs(r - c * 6 / 11).max() < 1.0
    # Only the path itself is water (a river): nothing to cut.
    river = np.zeros(n * n, dtype=bool)
    river[stair] = True
    assert np.array_equal(_pull_string(raster, stair, river), stair)


def test_chains_also_end_at_stops():
    edges = [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)]
    stops = np.array([False, False, True, False])
    assert sorted(n for n, _ in _chains(edges, 4, stops)) == [[0, 1, 2], [2, 3]]


def test_roads_pass_through_their_settlements(fields, synthetic_globe, world):
    """A settlement mid-chain (two drawn hops) is a chain end, so the road lands on it."""
    town = world.c[world.trunk[1]]
    routes = [r for r in world.routes if r[2] != BRANCH]      # trunk[1] has just its two trunk hops
    with _features(synthetic_globe, trade_routes=routes, settlements=[SimpleNamespace(position=town)]):
        roads = build_region_roads(fields, synthetic_globe, None)
    ends = np.array([p[k] for p in roads.polylines for k in (0, -1)])
    assert np.linalg.norm(ends - town, axis=1).min() < 1e-6


def test_a_port_on_the_water_boards_at_once(fields, synthetic_globe, world):
    """A port standing on refined water (as a re-sited city can) puts out from the
    shore: no road is drawn out over the water."""
    wet = (fields.textures["ocean_mask"][..., 0] > 0.5) & fields.domain
    sea_pts = fields.positions[wet]
    coast = world.c[world.trunk[-1]]
    port = sea_pts[np.argmin(np.linalg.norm(sea_pts - coast, axis=1))]
    routes = [r if r[2] != PORT else (port, r[1], PORT) for r in world.routes]
    with _features(synthetic_globe, trade_routes=routes, settlements=[SimpleNamespace(position=port)]):
        roads = build_region_roads(fields, synthetic_globe, None)
    legs = [(a, b) for a, b, f in roads.sea if f == PORT]
    assert len(legs) == 1
    assert not _refined_water(fields, np.asarray(legs[0][0], dtype=np.float64)[None])[0]
    step = RegionRoadParams().raster_texels * fields.grid.texel_spacing_rad(fields.tex_size)
    for path, flows in zip(roads.polylines, roads.polyline_flows):
        assert len(path) <= 2 or _refined_water(fields, path[1:-1]).mean() < 0.05
        if np.all(flows == PORT):     # at most a step or two from the port ashore to the shore
            assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < 3.0 * step
