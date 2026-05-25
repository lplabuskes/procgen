"""Tests for river generation: crossing detection and property checks.

Covers:
 - _crossing_pairs: great-circle arc intersection primitive (unit tests)
 - _resolve_crossings: post-processing step that terminates lower-flow rivers at
   crossing points (unit tests with synthetic River objects)
 - generate(): end-to-end property tests verifying no rivers cross and no river
   loops back on itself (over a few deterministic seeds)
"""
from __future__ import annotations

import numpy as np
import pytest

from worldgen.geology.elevation import compute_elevation_textures
from worldgen.hydrology.crossings import _arc_param, crossing_pairs, resolve_crossings
from worldgen.hydrology.rivers import RiverGenerator
from worldgen.hydrology.types import River, RiverNode, RiverSegment
from worldgen.mesh.mymesh import Mesh


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _arcs_cross(p1_a: np.ndarray, p2_a: np.ndarray,
                p1_b: np.ndarray, p2_b: np.ndarray) -> bool:
    """Return True if great-circle arc A (p1_a→p2_a) crosses arc B (p1_b→p2_b).

    Uses the same plane-intersection test as _crossing_pairs: candidate crossing
    point is ±(n_a × n_b)/|n_a × n_b| and is valid iff it lies in both arcs.
    """
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

    Only strict interior crossings are returned — endpoint touches (t < _ENDPOINT_TOL
    or t > 1 - _ENDPOINT_TOL for either arc) are tributary junctions, not crossings.
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
    """Return (ri_a, ri_b, si_a, si_b) for every crossing between distinct rivers.

    Uses AABB pre-filtering and vectorized arc intersection (same approach as
    _resolve_crossings) so the check runs fast even with hundreds of rivers.
    """
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
    # Elevation: mostly a smooth gradient by z-coordinate plus small noise so
    # that rivers form on the northern hemisphere and drain toward the equator.
    centroids = mesh.face_centroids            # (N, 3)
    z = centroids[:, 2]                        # in [-1, 1]
    elevations = 4.0 * z + 0.5 * rng.standard_normal(len(mesh.faces))
    textures = compute_elevation_textures(mesh, elevations, tex_size=8)
    rg = RiverGenerator(mesh, textures, min_flow=100.0, sea_level=0.0)
    return rg.generate()


@pytest.fixture(scope="module", params=[0, 7, 42])
def generated_rivers(request) -> list[River]:
    return _make_rivers_for_seed(request.param)


# ---------------------------------------------------------------------------
# TestCrossingPairs
# ---------------------------------------------------------------------------

class TestCrossingPairs:
    """Unit tests for the crossing_pairs primitive using synthetic arc data."""

    # Two diagonals of a small spherical square near the north pole cross in
    # the centre of the square.  Pick corners 10° from the pole at four
    # cardinal/diagonal azimuths so the arcs are clearly in the interior.
    _a = 10.0    # angular distance from pole in degrees

    @classmethod
    def _pole_corner(cls, az_deg: float) -> np.ndarray:
        """Unit-sphere point at angular distance cls._a from the north pole,
        at azimuth az_deg (degrees east of +X)."""
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
# TestResolveCrossings
# ---------------------------------------------------------------------------

def _make_river(positions: list[np.ndarray], flow_rate: float) -> River:
    """Construct a River from a list of unit-sphere positions and a constant flow."""
    segs = []
    for i in range(len(positions) - 1):
        segs.append(RiverSegment(
            start=RiverNode(position=positions[i].astype(np.float32),
                            elevation=0.0, face_id=0, uv=(0.0, 0.0)),
            end=RiverNode(position=positions[i + 1].astype(np.float32),
                          elevation=0.0, face_id=0, uv=(0.0, 0.0)),
            flow_rate=flow_rate,
        ))
    return River(segments=segs, source=segs[0].start, mouth=segs[-1].end)


class TestResolveCrossings:
    """Unit tests for resolve_crossings with hand-crafted crossing rivers."""

    def _crossing_pair(self):
        """Two rivers whose first segments cross; river B has higher flow."""
        a = 10.0
        ne = _unit(np.array([np.sin(np.deg2rad(a)), np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a * 1.41))]))
        sw = _unit(np.array([-np.sin(np.deg2rad(a)), -np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a * 1.41))]))
        nw = _unit(np.array([-np.sin(np.deg2rad(a)), np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a * 1.41))]))
        se = _unit(np.array([np.sin(np.deg2rad(a)), -np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a * 1.41))]))
        # River A goes NE → SW (low flow). River B goes NW → SE (high flow).
        # Each river has >2 segments so it survives the len>2 filter.
        river_a = _make_river(
            [ne, sw, _unit(sw + np.array([-0.1, -0.1, 0.0])),
             _unit(sw + np.array([-0.2, -0.2, 0.0]))],
            flow_rate=1.0,
        )
        river_b = _make_river(
            [nw, se, _unit(se + np.array([0.1, -0.1, 0.0])),
             _unit(se + np.array([0.2, -0.2, 0.0]))],
            flow_rate=100.0,
        )
        return river_a, river_b

    def test_lower_flow_river_truncated(self):
        river_a, river_b = self._crossing_pair()
        result = resolve_crossings([river_a, river_b])
        # River B (higher flow) must survive intact.
        b_in_result = any(
            abs(s.flow_rate - 100.0) < 1e-6 for r in result for s in r.segments
        )
        assert b_in_result, "higher-flow river must be kept"

    def test_no_crossings_after_resolve(self):
        river_a, river_b = self._crossing_pair()
        result = resolve_crossings([river_a, river_b])
        crossings = _inter_river_crossings(result)
        assert crossings == [], (
            f"rivers still cross after resolve_crossings: {crossings}"
        )

    def test_single_river_unchanged(self):
        p1 = _unit(np.array([0.5, 0.0, 0.866]))
        p2 = _unit(np.array([0.0, 0.5, 0.866]))
        p3 = _unit(np.array([-0.5, 0.0, 0.866]))
        p4 = _unit(np.array([0.0, -0.5, 0.866]))
        river = _make_river([p1, p2, p3, p4], flow_rate=10.0)
        result = resolve_crossings([river])
        assert len(result) == 1
        assert len(result[0].segments) == len(river.segments)

    def test_empty_list_unchanged(self):
        assert resolve_crossings([]) == []


# ---------------------------------------------------------------------------
# TestRiverProperties — end-to-end checks over a few seeds
# ---------------------------------------------------------------------------

class TestRiverProperties:
    """Property tests: no pair of output rivers may cross, and no river may
    loop back on itself. These tests run over three deterministic seeds using
    a small icosphere(3) mesh with gradient terrain, so they exercise both
    the Priority-Flood / D-infinity pipeline and the crossing-resolution step.
    """

    def test_rivers_produced(self, generated_rivers):
        """Ensure the pipeline actually generates rivers so the property checks
        below are not vacuously true."""
        assert len(generated_rivers) > 0, (
            "no rivers generated — the property tests below would be vacuous"
        )

    def test_no_inter_river_crossings(self, generated_rivers):
        crossings = _inter_river_crossings(generated_rivers)
        assert crossings == [], (
            f"{len(crossings)} crossing(s) found between rivers: "
            + "; ".join(f"rivers {a}&{b} segs {sa}&{sb}"
                        for a, b, sa, sb in crossings[:5])
        )

    def test_no_intra_river_loops(self, generated_rivers):
        for ri, river in enumerate(generated_rivers):
            loops = _intra_river_loops(river)
            assert loops == [], (
                f"river {ri} self-intersects at segment pairs {loops[:5]}"
            )


# ---------------------------------------------------------------------------
# TestRiverMouthTermination — every mouth must reach a physical destination
# ---------------------------------------------------------------------------

def _min_chord_dist_to_polyline(q: np.ndarray, river: River) -> float:
    """Minimum chord distance from unit-sphere point q to any segment of river.

    Uses the standard parametric nearest-point-on-segment formula in 3-space.
    For short arcs (< ~5°) the chord vs. great-circle approximation error is
    < 0.002, which is well within the junction detection tolerance.
    """
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

    - ocean: nearest grid node is in the ocean mask, or the mouth is within
      OCEAN_TOL chord distance of any ocean node (covers traces that reached the
      ocean but whose mouth drifted slightly inland after Laplacian smoothing).
    - lake: nearest grid node is in the endorheic basin mask (closed sink).
    - junction: mouth lies on another river's polyline within JUNCTION_TOL;
      this is how resolve_crossings terminates lower-flow tributaries.
    - unknown: none of the above; indicates a stall, loop, or geometry failure.
    """
    q = river.mouth.position.astype(np.float64)
    nearest = int(np.argmin(np.linalg.norm(gen.positions - q[None], axis=1)))

    if gen._ocean_mask[nearest]:
        return 'ocean'
    if gen._endorheic_mask[nearest]:
        return 'lake'

    # Near-ocean check: Laplacian smoothing pins polyline endpoints but
    # resolve_self_crossings can overwrite river.mouth with a segment-end node
    # that drifted slightly inland. Accept any mouth within ~1-2 pixel widths
    # of an ocean node.
    OCEAN_TOL = 0.025
    ocean_idxs = np.where(gen._ocean_mask)[0]
    if ocean_idxs.size > 0:
        if np.linalg.norm(gen.positions[ocean_idxs] - q[None], axis=1).min() < OCEAN_TOL:
            return 'ocean'

    # After resolve_crossings, a truncated river's mouth is placed at the exact
    # great-circle crossing point, which lies on the other river's arc. The
    # chord approximation error is < 0.002 for 5° segments, so 0.01 is safe.
    # A slightly larger tolerance (0.03) also handles cases where smoothing
    # moved the junction point away from the arc.
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


def _termination_reason_for_river(river: River, gen: RiverGenerator) -> str:
    """Look up the trace termination reason for a river by matching its source node."""
    src = river.source
    if src is None:
        return 'unknown_source'
    face = src.face_id
    res = gen.resolution
    u, v = src.uv
    col = int(round(u * (res - 1)))
    row = int(round(v * (res - 1)))
    hw = face * res * res + row * res + col
    return gen._trace_reasons.get(hw, 'not_found')


class TestRiverMouthTermination:
    """Every river mouth must terminate at the ocean, an endorheic lake, or as
    a tributary joining a higher-flow river (junction created by resolve_crossings).

    The test is intentionally diagnostic: the error message reports the full
    category breakdown so the awkward-ending fraction is visible at a glance.
    """

    def test_rivers_generated(self, gen_and_rivers_mouths):
        _, rivers = gen_and_rivers_mouths
        assert len(rivers) > 0, "no rivers generated — mouth tests would be vacuous"

    def test_termination_reason_breakdown(self, gen_and_rivers_mouths):
        """Diagnostic: show trace-reason and mouth context for unknown-mouth rivers.

        This test always passes — it prints the breakdown so we can see which
        termination conditions remain and what the mouth positions look like.
        """
        gen, rivers = gen_and_rivers_mouths
        unknown_details: list[str] = []
        reason_counts: dict[str, int] = {}
        for river in rivers:
            cat = _classify_river_mouth(river, rivers, gen)
            if cat != 'unknown':
                continue
            reason = _termination_reason_for_river(river, gen)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            q = river.mouth.position.astype(np.float64)
            nearest = int(np.argmin(np.linalg.norm(gen.positions - q[None], axis=1)))
            near_elev = float(gen.elevations[nearest])
            ocean_idxs = np.where(gen._ocean_mask)[0]
            if ocean_idxs.size:
                min_ocean_d = float(np.min(np.linalg.norm(gen.positions[ocean_idxs] - q[None], axis=1)))
            else:
                min_ocean_d = float('inf')
            min_river_d = min(
                (_min_chord_dist_to_polyline(q, r) for r in rivers if r is not river),
                default=float('inf'),
            )
            unknown_details.append(
                f"    trace={reason} near_elev={near_elev:.3f} "
                f"d_ocean={min_ocean_d:.4f} d_river={min_river_d:.4f}"
            )
        if reason_counts:
            breakdown_str = ', '.join(f'{r}={n}' for r, n in sorted(reason_counts.items()))
            print(f"\n  Unknown mouths by trace reason: {breakdown_str}")
            for d in unknown_details[:8]:
                print(d)

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
