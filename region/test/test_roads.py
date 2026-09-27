"""Regional roads: the world's trade routes re-routed over the refined terrain."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from region.rivers import build_region_rivers
from region.roads import RegionRoadParams, _chains, build_region_roads
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
    # A coastal face with a sea neighbour in the cell, and the inland face
    # furthest from it.
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
