"""River banks: settlements beside a channel, texels under it, fields on one bank."""
import dataclasses

import numpy as np
import pytest

from localmap.banks import Channels, covered_texels, crossing_pairs, onto_banks
from localmap.cities import city_footprints
from localmap.farmland import FARM_VILLAGE, FarmlandParams, place_farmland
from localmap.test.conftest import local_params
from localmap.test.test_farmland import _people, _water, sites  # noqa: F401
from localmap.upsample import build_local_fields
from region.upsample import EARTH_RADIUS_KM


@pytest.fixture(scope="module")
def local(globe, region, lake_hex):
    return build_local_fields(globe, region, lake_hex, local_params())


def _line(local, s0: float, width_km: float) -> Channels:
    """A straight north–south channel at local ``s = s0`` across the whole map."""
    t = np.linspace(-0.1, 1.1, 241)
    pts = local.geometry.local_to_ecef(np.full_like(t, s0), t)
    return Channels(a=pts[:-1], b=pts[1:], width_km=np.full(len(t) - 1, float(width_km)))


def _s(local, points) -> np.ndarray:
    return local.geometry.ecef_to_local(np.asarray(points, dtype=np.float64).reshape(-1, 3))[0]


def _texel_km(local) -> float:
    return local.grid.texel_spacing_rad(local.tex_size) * EARTH_RADIUS_KM


@pytest.fixture(scope="module")
def river(local, sites):  # noqa: F811
    """Three texels wide, straight through the first site."""
    return _line(local, float(_s(local, sites[:1])[0]), 3.0 * _texel_km(local))


def _edge_km(channels: Channels, points: np.ndarray) -> np.ndarray:
    """Distance from each point to the channel's bank (negative inside)."""
    ab = channels.b - channels.a
    out = []
    for p in points:
        u = np.clip(((p - channels.a) * ab).sum(1) / (ab * ab).sum(1), 0.0, 1.0)
        d = np.linalg.norm(channels.a + u[:, None] * ab - p, axis=1) - channels.half_width
        out.append(d.min())
    return np.array(out) * EARTH_RADIUS_KM


def test_settlements_move_just_past_the_bank_on_their_side(local, river, sites):  # noqa: F811
    s0 = float(_s(local, sites[:1])[0])
    texel = local.grid.radius * 2.0 / local.tex_size
    t = np.array([0.4, 0.5, 0.6, 0.5])
    s = np.array([s0, s0 - 0.5 * texel, s0 + 0.5 * texel, s0 + 10.0 * texel])
    points = local.geometry.local_to_ecef(s, t)
    moved = onto_banks(river, points, clearance_km=0.1)
    np.testing.assert_allclose(_edge_km(river, moved[:3]), 0.1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(moved, axis=1), 1.0)
    side = np.sign(_s(local, moved) - s0)
    assert side[1] < 0 and side[2] > 0                  # each stays on its own bank
    np.testing.assert_array_equal(moved[3], points[3])  # clear of the river: unchanged
    np.testing.assert_allclose(onto_banks(river, moved, clearance_km=0.1), moved, atol=1e-12)


def test_covered_texels_are_those_under_the_channel(local, river):
    covered = covered_texels(local, river)
    edge = _edge_km(river, local.positions.reshape(-1, 3)).reshape(covered.shape)
    assert covered.any()
    np.testing.assert_array_equal(covered, edge < 0.0)


def test_crossing_pairs(local, river, sites):  # noqa: F811
    s0 = float(_s(local, sites[:1])[0])
    texel = local.grid.radius * 2.0 / local.tex_size
    origin = local.geometry.local_to_ecef(np.array([s0 - 3 * texel]), np.array([0.5]))
    targets = local.geometry.local_to_ecef(np.array([s0 - 6 * texel, s0 + 3 * texel, s0 - texel]),
                                           np.array([0.5, 0.55, 0.45]))
    cross = crossing_pairs(river, origin, np.zeros(3, dtype=np.int64), targets, np.array([10 * texel]))
    np.testing.assert_array_equal(cross, [False, True, False])


def test_city_footprints_skip_the_river(local, river, sites):  # noqa: F811
    covered = covered_texels(local, river)
    fp = city_footprints(local, sites[:1], np.array([60_000.0]), seed=2, river_covered=covered)
    assert (fp.owner >= 0).any()
    assert not ((fp.owner >= 0) & covered).any()


def test_village_fields_keep_to_their_bank_of_a_wide_river(local, river, sites):  # noqa: F811
    covered = covered_texels(local, river)
    village = onto_banks(river, sites[:1], clearance_km=0.1)
    s0 = float(_s(local, sites[:1])[0])
    side = np.sign(_s(local, village)[0] - s0)

    def fields(params):
        fm = place_farmland(local, village, _people(local, [40]), np.zeros((0, 3)), np.zeros(0), None, 8, params,
                            channels=river, river_covered=covered)
        farmed = fm.kind == FARM_VILLAGE
        assert farmed.any() and not (farmed & covered).any()
        return np.sign(_s(local, local.positions[farmed]) - s0)

    assert (fields(FarmlandParams()) == side).all()
    # A river narrower than the barrier width is fordable: fields reach across.
    fordable = dataclasses.replace(FarmlandParams(), river_barrier_width_m=1e6)
    assert (fields(fordable) == -side).any()
