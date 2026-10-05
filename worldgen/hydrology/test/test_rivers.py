"""River generation tests: crossing_pairs unit tests and end-to-end properties of generate()."""
from __future__ import annotations

import numpy as np
import pytest

from worldgen.geology.elevation import compute_elevation_textures
from worldgen.hydrology.crossings import _arc_param, crossing_pairs
from worldgen.hydrology.rivers import RiverGenerator
from worldgen.hydrology.types import River
from core.mesh import Mesh


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _arcs_cross(p1_a: np.ndarray, p2_a: np.ndarray,
                p1_b: np.ndarray, p2_b: np.ndarray) -> bool:
    """True if great-circle arc A (p1_a→p2_a) crosses arc B (candidate point ±n_a × n_b)."""
    n_a = np.cross(p1_a, p2_a)
    n_b = np.cross(p1_b, p2_b)
    d = np.cross(n_a, n_b)
    d_len = float(np.linalg.norm(d))
    if d_len < 1e-10:
        return False
    for sign in (1.0, -1.0):
        c = sign * d / d_len
        if (np.dot(np.cross(p1_a, c), n_a) >= -1e-6
                and np.dot(np.cross(c, p2_a), n_a) >= -1e-6
                and np.dot(np.cross(p1_b, c), n_b) >= -1e-6
                and np.dot(np.cross(c, p2_b), n_b) >= -1e-6):
            return True
    return False


def _river_segments(river: River) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized (p1, p2) arrays for every segment of a river."""
    p1 = np.array([s.start.position for s in river.segments], dtype=np.float64)
    p2 = np.array([s.end.position for s in river.segments], dtype=np.float64)
    p1 /= np.maximum(np.linalg.norm(p1, axis=1, keepdims=True), 1e-12)
    p2 /= np.maximum(np.linalg.norm(p2, axis=1, keepdims=True), 1e-12)
    return p1, p2


def _aabb(p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.vstack([p1, p2])
    return pts.min(axis=0), pts.max(axis=0)


_ENDPOINT_TOL = 0.01   # crossings within this fraction of an arc endpoint are junctions


def _crossing_pairs_np(
    p1_a: np.ndarray, p2_a: np.ndarray,
    p1_b: np.ndarray, p2_b: np.ndarray,
) -> list[tuple[int, int]]:
    """Vectorized great-circle arc crossing test; returns (si_a, si_b) hit pairs.

    Endpoint touches (within _ENDPOINT_TOL) are junctions and excluded.
    """
    n_a = np.cross(p1_a, p2_a)
    n_b = np.cross(p1_b, p2_b)
    d = np.cross(n_a[:, None, :], n_b[None, :, :])   # (Na, Nb, 3)
    d_len = np.linalg.norm(d, axis=2)                 # (Na, Nb)
    valid = d_len > 1e-10
    hits: list[tuple[int, int]] = []
    for sign in (1.0, -1.0):
        safe = np.where(valid, d_len, 1.0)
        c = sign * d / safe[:, :, None]
        on_a = (
            np.sum(np.cross(p1_a[:, None, :], c) * n_a[:, None, :], axis=2) >= -1e-6
        ) & (
            np.sum(np.cross(c, p2_a[:, None, :]) * n_a[:, None, :], axis=2) >= -1e-6
        )
        on_b = (
            np.sum(np.cross(p1_b[None, :, :], c) * n_b[None, :, :], axis=2) >= -1e-6
        ) & (
            np.sum(np.cross(c, p2_b[None, :, :]) * n_b[None, :, :], axis=2) >= -1e-6
        )
        mask = valid & on_a & on_b
        for i, j in zip(*np.where(mask)):
            i, j = int(i), int(j)
            cp = c[i, j]
            t_a = _arc_param(p1_a[i], p2_a[i], cp)
            t_b = _arc_param(p1_b[j], p2_b[j], cp)
            if (_ENDPOINT_TOL < t_a < 1 - _ENDPOINT_TOL
                    and _ENDPOINT_TOL < t_b < 1 - _ENDPOINT_TOL):
                hits.append((i, j))
    return hits


def _inter_river_crossings(rivers: list[River]) -> list[tuple]:
    """(ri_a, ri_b, si_a, si_b) for every crossing between distinct rivers (AABB pre-filtered)."""
    segs = [_river_segments(r) for r in rivers]
    bounds = [_aabb(p1, p2) for p1, p2 in segs]
    crossings = []
    for ri_a in range(len(rivers)):
        p1_a, p2_a = segs[ri_a]
        lo_a, hi_a = bounds[ri_a]
        for ri_b in range(ri_a + 1, len(rivers)):
            lo_b, hi_b = bounds[ri_b]
            if np.any(lo_a > hi_b) or np.any(lo_b > hi_a):
                continue
            p1_b, p2_b = segs[ri_b]
            for si_a, si_b in _crossing_pairs_np(p1_a, p2_a, p1_b, p2_b):
                crossings.append((ri_a, ri_b, si_a, si_b))
    return crossings


def _intra_river_loops(river: River) -> list[tuple]:
    """Return (si, sj) for self-crossing non-adjacent segment pairs within a river."""
    p1, p2 = _river_segments(river)
    loops = []
    for i in range(len(p1)):
        for j in range(i + 2, len(p1)):  # skip adjacent segments
            if _arcs_cross(p1[i], p2[i], p1[j], p2[j]):
                loops.append((i, j))
    return loops


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rivers_for_seed(seed: int) -> list[River]:
    """Run a minimal but realistic pipeline and return generated rivers."""
    mesh = Mesh.icosphere(3).mesh_dual()
    rng = np.random.default_rng(seed)
    # z-gradient plus noise: rivers drain from the north toward the equator.
    centroids = mesh.face_centroids            # (N, 3)
    z = centroids[:, 2]                        # in [-1, 1]
    elevations = 4.0 * z + 0.5 * rng.standard_normal(len(mesh.faces))
    textures = compute_elevation_textures(mesh, elevations, tex_size=8)
    rg = RiverGenerator(mesh, textures, min_flow=100.0, sea_level=0.0)
    return rg.generate()


# Inter-river crossings are accepted: rivers are smoothed independently (flow-based
# truncation was dropped — it chopped long modest-flow rivers). Per-seed ratchet:
# fails only if overlaps get worse.
_MAX_INTER_RIVER_CROSSINGS = {0: 6, 7: 2, 42: 3}


@pytest.fixture(scope="module", params=[0, 7, 42])
def generated_rivers(request) -> tuple[int, list[River]]:
    return request.param, _make_rivers_for_seed(request.param)


# ---------------------------------------------------------------------------
# TestCrossingPairs
# ---------------------------------------------------------------------------

class TestCrossingPairs:
    """Unit tests for the crossing_pairs primitive using synthetic arc data."""

    # Diagonals of a small spherical square, corners 10° from the north pole.
    _a = 10.0    # angular distance from pole in degrees

    @classmethod
    def _pole_corner(cls, az_deg: float) -> np.ndarray:
        """Unit-sphere point cls._a degrees from the north pole at azimuth az_deg (east of +X)."""
        ar = np.deg2rad(cls._a)
        az = np.deg2rad(az_deg)
        return _unit(np.array([
            np.sin(ar) * np.cos(az),
            np.sin(ar) * np.sin(az),
            np.cos(ar),
        ]))

    def _run(self, p1_a, p2_a, p1_b, p2_b):
        a1 = np.array([p1_a], dtype=np.float64)
        a2 = np.array([p2_a], dtype=np.float64)
        b1 = np.array([p1_b], dtype=np.float64)
        b2 = np.array([p2_b], dtype=np.float64)
        return crossing_pairs(a1, a2, b1, b2)

    def test_crossing_diagonals_detected(self):
        # NE↔SW diagonal and NW↔SE diagonal of a small spherical square.
        ne = self._pole_corner(45)
        sw = self._pole_corner(225)
        nw = self._pole_corner(135)
        se = self._pole_corner(315)
        hits = self._run(ne, sw, nw, se)
        assert len(hits) >= 1, "diagonals of spherical square must cross"

    def test_crossing_point_lies_on_both_arcs(self):
        ne = self._pole_corner(45)
        sw = self._pole_corner(225)
        nw = self._pole_corner(135)
        se = self._pole_corner(315)
        hits = self._run(ne, sw, nw, se)
        si_a, si_b, t_a, t_b, cp = hits[0]
        assert 0.0 <= t_a <= 1.0
        assert 0.0 <= t_b <= 1.0
        # cp must be unit-length and lie between the arc endpoints
        assert abs(float(np.linalg.norm(cp)) - 1.0) < 1e-8

    def test_parallel_arcs_not_detected(self):
        # Two arcs at the same latitude (different great circles far apart).
        lat = np.deg2rad(60)
        p1 = _unit(np.array([np.cos(lat), 0.0, np.sin(lat)]))
        p2 = _unit(np.array([0.0, np.cos(lat), np.sin(lat)]))
        # Second arc at the same latitude circle but offset ~90° — parallel, no crossing.
        q1 = _unit(np.array([-np.cos(lat), 0.0, np.sin(lat)]))
        q2 = _unit(np.array([0.0, -np.cos(lat), np.sin(lat)]))
        hits = self._run(p1, p2, q1, q2)
        assert len(hits) == 0, "non-crossing arcs should produce no hits"

    def test_adjacent_arcs_sharing_endpoint_not_reported(self):
        # Two arcs that share an endpoint should not produce an interior crossing.
        p1 = _unit(np.array([1.0, 0.0, 0.0]))
        mid = _unit(np.array([0.0, 1.0, 0.0]))
        p2 = _unit(np.array([-1.0, 0.0, 0.3]))
        # Arc A: p1 → mid; Arc B: mid → p2; they share `mid` but don't cross.
        hits = self._run(p1, mid, mid, p2)
        # If a hit is detected, it must be at t≈0 or t≈1 (endpoint touch), not interior.
        for _, _, t_a, t_b, _ in hits:
            assert t_a < 1e-4 or t_b < 1e-4 or t_a > 1 - 1e-4 or t_b > 1 - 1e-4

    def test_vectorized_multiple_arcs(self):
        # Pass two segments for river A and two for river B; exactly one crossing pair.
        ne = self._pole_corner(45)
        sw = self._pole_corner(225)
        nw = self._pole_corner(135)
        se = self._pole_corner(315)
        a1 = np.array([ne, _unit(ne + np.array([0.01, 0.0, 0.0]))], dtype=np.float64)
        a2 = np.array([sw, _unit(sw + np.array([0.01, 0.0, 0.0]))], dtype=np.float64)
        b1 = np.array([nw, _unit(nw + np.array([0.0, 0.01, 0.0]))], dtype=np.float64)
        b2 = np.array([se, _unit(se + np.array([0.0, 0.01, 0.0]))], dtype=np.float64)

        hits = crossing_pairs(a1, a2, b1, b2)
        # The first segment pair (ne→sw) × (nw→se) must be in the results.
        crossing_indices = {(h[0], h[1]) for h in hits}
        assert (0, 0) in crossing_indices


# ---------------------------------------------------------------------------
# TestRiverProperties — end-to-end checks over a few seeds
# ---------------------------------------------------------------------------

class TestRiverProperties:
    """No river loops on itself; inter-river crossings stay within ``_MAX_INTER_RIVER_CROSSINGS``."""

    def test_rivers_produced(self, generated_rivers):
        """Guards against the property checks being vacuous."""
        _seed, rivers = generated_rivers
        assert len(rivers) > 0, (
            "no rivers generated — the property tests below would be vacuous"
        )

    def test_inter_river_crossings_within_baseline(self, generated_rivers):
        seed, rivers = generated_rivers
        crossings = _inter_river_crossings(rivers)
        allowed = _MAX_INTER_RIVER_CROSSINGS[seed]
        assert len(crossings) <= allowed, (
            f"{len(crossings)} inter-river crossing(s) for seed {seed} exceeds "
            f"baseline {allowed}: "
            + "; ".join(f"rivers {a}&{b} segs {sa}&{sb}"
                        for a, b, sa, sb in crossings[:5])
        )

    def test_no_intra_river_loops(self, generated_rivers):
        _seed, rivers = generated_rivers
        for ri, river in enumerate(rivers):
            loops = _intra_river_loops(river)
            assert loops == [], (
                f"river {ri} self-intersects at segment pairs {loops[:5]}"
            )


# ---------------------------------------------------------------------------
# TestRiverMouthTermination — every mouth must reach a physical destination
# ---------------------------------------------------------------------------

def _min_chord_dist_to_polyline(q: np.ndarray, river: River) -> float:
    """Minimum chord distance from unit-sphere point q to river (error < 0.002 for arcs < ~5°)."""
    pts = river.as_polyline()
    if len(pts) < 2:
        return float('inf')
    arr = np.array(pts, dtype=np.float64)
    p1 = arr[:-1]
    p2 = arr[1:]
    d = p2 - p1
    dlen2 = np.einsum('ni,ni->n', d, d).clip(min=1e-30)
    t = np.clip(np.einsum('ni,ni->n', q[None] - p1, d) / dlen2, 0.0, 1.0)
    closest = p1 + t[:, None] * d
    return float(np.linalg.norm(closest - q[None], axis=1).min())


def _classify_river_mouth(river: River, all_rivers: list, gen: RiverGenerator) -> str:
    """Return 'ocean', 'lake', 'junction', or 'unknown' for a river's mouth.

    - ocean: nearest node is ocean, or within OCEAN_TOL of one.
    - lake: nearest node is endorheic.
    - junction: mouth lies on another river within JUNCTION_TOL (tributary confluence).
    - unknown: a stall, loop, or geometry failure.
    """
    assert river.mouth is not None
    q = river.mouth.position.astype(np.float64)
    nearest = int(np.argmin(np.linalg.norm(gen.positions - q[None], axis=1)))

    if gen._ocean_mask[nearest]:
        return 'ocean'
    if gen._endorheic_mask[nearest]:
        return 'lake'

    # resolve_self_crossings can move river.mouth to a smoothed node slightly
    # inland; accept ~1-2 pixel widths.
    OCEAN_TOL = 0.025
    ocean_idxs = np.where(gen._ocean_mask)[0]
    if ocean_idxs.size > 0:
        if np.linalg.norm(gen.positions[ocean_idxs] - q[None], axis=1).min() < OCEAN_TOL:
            return 'ocean'

    # Tolerance also absorbs smoothing tributary and stem independently.
    JUNCTION_TOL = 0.03
    for other in all_rivers:
        if other is river:
            continue
        if _min_chord_dist_to_polyline(q, other) < JUNCTION_TOL:
            return 'junction'

    return 'unknown'


def _make_gen_and_rivers_mouth_test(seed: int):
    mesh = Mesh.icosphere(3).mesh_dual()
    rng = np.random.default_rng(seed)
    centroids = mesh.face_centroids
    z = centroids[:, 2]
    elevations = 4.0 * z + 0.5 * rng.standard_normal(len(mesh.faces))
    textures = compute_elevation_textures(mesh, elevations, tex_size=8)
    rg = RiverGenerator(mesh, textures, min_flow=100.0, sea_level=0.0)
    rivers = rg.generate()
    return rg, rivers


@pytest.fixture(scope="module", params=[0, 7, 42], ids=["seed0", "seed7", "seed42"])
def gen_and_rivers_mouths(request):
    return _make_gen_and_rivers_mouth_test(request.param)


class TestRiverMouthTermination:
    """Every mouth ends at the ocean, an endorheic lake, or a confluence with its main stem."""

    def test_rivers_generated(self, gen_and_rivers_mouths):
        _, rivers = gen_and_rivers_mouths
        assert len(rivers) > 0, "no rivers generated — mouth tests would be vacuous"

    def test_all_mouths_have_valid_termination(self, gen_and_rivers_mouths):
        gen, rivers = gen_and_rivers_mouths
        breakdown: dict[str, list[int]] = {
            'ocean': [], 'lake': [], 'junction': [], 'unknown': [],
        }
        for i, river in enumerate(rivers):
            cat = _classify_river_mouth(river, rivers, gen)
            breakdown[cat].append(i)

        unknown = breakdown['unknown']
        assert not unknown, (
            f"{len(unknown)}/{len(rivers)} river(s) have unclassified mouth termination "
            f"(not ocean, lake, or junction). "
            f"ocean={len(breakdown['ocean'])}, "
            f"lake={len(breakdown['lake'])}, "
            f"junction={len(breakdown['junction'])}, "
            f"unknown={len(unknown)}. "
            f"Unknown river indices: {unknown[:10]}"
        )


# ---------------------------------------------------------------------------
# Stream network
# ---------------------------------------------------------------------------

_STREAM_THRESHOLD = 20.0


@pytest.fixture(scope="module")
def gen_with_streams():
    mesh = Mesh.icosphere(3).mesh_dual()
    rng = np.random.default_rng(3)
    z = mesh.face_centroids[:, 2]
    textures = compute_elevation_textures(mesh, 4.0 * z + 0.5 * rng.standard_normal(len(mesh.faces)), tex_size=8)
    gen = RiverGenerator(mesh, textures, min_flow=100.0, sea_level=0.0)
    rivers = gen.generate()
    return gen, rivers, gen.stream_network(_STREAM_THRESHOLD)


class TestStreamNetwork:
    def test_channels_carry_the_threshold(self, gen_with_streams):
        _, _, streams = gen_with_streams
        assert streams.channel.sum() > 0
        assert np.all(streams.flow[streams.channel] >= _STREAM_THRESHOLD)

    def test_every_channel_drains_to_an_end(self, gen_with_streams):
        _, _, streams = gen_with_streams
        for start in np.flatnonzero(streams.channel)[::7].tolist():
            node, steps = start, 0
            while streams.receiver[node] >= 0 and not streams.stop[node]:
                nxt = int(streams.receiver[node])
                assert streams.flow[nxt] >= streams.flow[node] - 1e-3     # flow never drops downstream
                node, steps = nxt, steps + 1
                assert steps < len(streams)

    def test_rivers_run_along_the_network(self, gen_with_streams):
        """The rivers are the network's high-flow part: every river source is a channel."""
        gen, rivers, streams = gen_with_streams
        res = streams.resolution
        for river in rivers:
            u, v = river.source.uv
            node = river.source.face_id * res * res + round(v * (res - 1)) * res + round(u * (res - 1))
            k = np.searchsorted(streams.nodes, node)
            assert streams.nodes[k] == node and streams.channel[k]

    def test_positions_are_the_texel_centres(self, gen_with_streams):
        gen, _, streams = gen_with_streams
        np.testing.assert_allclose(streams.positions, gen.positions[streams.nodes], atol=1e-6)

    def test_pack_round_trip(self, gen_with_streams):
        from worldgen.hydrology.types import StreamNetwork
        _, _, streams = gen_with_streams
        back = StreamNetwork.unpack(streams.pack())
        for name in ("nodes", "receiver", "flow", "channel", "stop"):
            np.testing.assert_array_equal(getattr(back, name), getattr(streams, name))
        assert back.resolution == streams.resolution
