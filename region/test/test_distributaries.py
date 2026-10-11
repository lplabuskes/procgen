"""Delta distributaries: a delta river fanning out over its plain on a region."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from region.distributaries import DistributaryParams, mouth_count
from region.rivers import RegionRiverParams, build_region_rivers
from region.sampling import GlobalSampler
from region.upsample import EARTH_RADIUS_KM, RegionUpsampleParams, build_region_fields
from worldgen.delta.deposit import DeltaFan
from worldgen.delta.shape import BIRDFOOT, CUSPATE

REGION_ID = 40   # coast, a lake and a peak (see conftest)
# The synthetic world's flows are tiny, so fix the mouth count instead of scaling it.
PARAMS = RegionRiverParams(distributaries=DistributaryParams(base_count=4, per_decade=0.0))


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    return build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16))


@pytest.fixture(scope="module")
def plain_rivers(fields, synthetic_globe):
    off = dataclasses.replace(PARAMS, distributaries=DistributaryParams(enabled=False))
    return build_region_rivers(fields, synthetic_globe, off)


@pytest.fixture(scope="module")
def delta_setup(fields, synthetic_globe, plain_rivers):
    """A delta at the biggest river mouth: a fan, and its plain on the region's textures."""
    i = int(np.argmax([float(f[-1]) if raster_ends_at_sea else -1.0
                       for (p, f, _), raster_ends_at_sea in _ends_at_sea(plain_rivers, fields)]))
    pts = np.asarray(plain_rivers.paths[i][0], dtype=np.float64)
    apex = pts[max(len(pts) - 1 - max(len(pts) // 3, 4), 0)]
    mouth = pts[-1]
    radius_km = 1.6 * float(np.linalg.norm(mouth - apex)) * EARTH_RADIUS_KM
    fan = DeltaFan(apex=apex, mouth=mouth, axis=mouth - apex, radius_km=radius_km,
                   flow=float(plain_rivers.paths[i][1][-1]), area_km2=0.0, head=mouth.copy())
    pos = fields.positions.reshape(-1, 3)
    near = np.linalg.norm(pos - apex[None, :], axis=1) * EARTH_RADIUS_KM < radius_km
    plain = np.zeros((pos.shape[0], 2), dtype=np.float32)
    plain[near, 0] = 1.0
    with_delta = dataclasses.replace(
        fields, textures={**fields.textures, "delta": plain.reshape(*fields.positions.shape[:3], 2)})
    return with_delta, fan, i


def _ends_at_sea(rivers, fields):
    ocean = fields.positions[fields.textures["ocean_mask"][..., 0] > 0.5]
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    for path in rivers.paths:
        end = np.asarray(path[0][-1], dtype=np.float64)
        yield path, bool(np.linalg.norm(ocean - end[None, :], axis=1).min() < 2.0 * spacing)


@pytest.fixture(scope="module")
def branched(delta_setup, synthetic_globe):
    fields, fan, _ = delta_setup
    synthetic_globe.add_feature("delta_fans", [fan])
    try:
        yield build_region_rivers(fields, synthetic_globe, PARAMS), build_region_rivers(
            fields, synthetic_globe, PARAMS)
    finally:
        synthetic_globe.remove_feature("delta_fans")


def test_the_delta_river_fans_out(branched, plain_rivers):
    rivers, _ = branched
    extra = rivers.paths[len(plain_rivers.paths):]
    assert len(extra) >= 1
    assert len(rivers.mouths) == len(plain_rivers.mouths) + len(extra)
    # The original paths are untouched.
    for a, b in zip(plain_rivers.paths, rivers.paths):
        assert np.array_equal(a[0], b[0])


def test_branches_leave_a_channel_and_reach_the_sea(branched, plain_rivers, delta_setup, synthetic_globe):
    rivers, _ = branched
    fields, _, stem = delta_setup
    extra = rivers.paths[len(plain_rivers.paths):]
    lines = [np.asarray(rivers.paths[stem][0], dtype=np.float64)]
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    ocean = fields.positions[fields.textures["ocean_mask"][..., 0] > 0.5]
    sampler = GlobalSampler(synthetic_globe.mesh)
    for positions, flows, _ in extra:
        pts = np.asarray(positions, dtype=np.float64)
        # Starts exactly on the stem or an earlier branch.
        assert min(np.linalg.norm(line - pts[0][None, :], axis=1).min() for line in lines) < 1e-6
        # Ends on the sea as the router sees it: the refined sea on the region's textures,
        # the globe's past them (sample_raster), where a branch off the cell's edge may end.
        s, t, _ = fields.geometry.ecef_to_local(pts[-1:])
        if fields.grid.nearest_texels(s, t, fields.tex_size)[3][0]:
            assert np.linalg.norm(ocean - pts[-1][None, :], axis=1).min() < 2.0 * spacing
        else:
            faces, u, v = sampler.locate(pts[-1:])
            assert sampler.sample(synthetic_globe.mesh.face_textures["ocean_mask"], faces, u, v)[0, 0] > 0.5
        lines.append(pts)


def test_branch_flows_are_shares_of_the_stem(branched, plain_rivers, delta_setup):
    rivers, _ = branched
    _, _, stem = delta_setup
    stem_flow = float(np.max(rivers.paths[stem][1]))
    for _, flows, _ in rivers.paths[len(plain_rivers.paths):]:
        assert np.all(flows > 0) and np.all(flows < stem_flow)
        assert np.ptp(flows) == 0.0


def test_distributaries_are_deterministic(branched):
    a, b = branched
    assert len(a.paths) == len(b.paths)
    for pa, pb in zip(a.paths, b.paths):
        assert np.array_equal(pa[0], pb[0])


def _with_fan(globe, fields, fan):
    globe.add_feature("delta_fans", [fan])
    try:
        return build_region_rivers(fields, globe, PARAMS)
    finally:
        globe.remove_feature("delta_fans")


def test_a_cuspate_delta_keeps_few_mouths(delta_setup, synthetic_globe, plain_rivers):
    fields, fan, _ = delta_setup
    rivers = _with_fan(synthetic_globe, fields, dataclasses.replace(fan, shape=CUSPATE))
    extra = len(rivers.paths) - len(plain_rivers.paths)
    assert 0 <= extra <= PARAMS.distributaries.wave_mouths - 1


def test_a_birdfoot_delta_has_a_mouth_at_each_finger(branched, delta_setup, synthetic_globe, plain_rivers):
    arcuate, _ = branched
    fields, fan, _ = delta_setup
    # Fingers where the arcuate branches reached the sea.
    tips = np.array([p[0][-1] for p in arcuate.paths[len(plain_rivers.paths):]], dtype=np.float64)
    rivers = _with_fan(synthetic_globe, fields, dataclasses.replace(fan, shape=BIRDFOOT, fingers=tips))
    ends = np.array([p[0][-1] for p in rivers.paths[len(plain_rivers.paths):]], dtype=np.float64)
    assert len(ends) >= 1
    spacing = fields.grid.texel_spacing_rad(fields.tex_size)
    gap = np.linalg.norm(ends[:, None, :] - tips[None, :, :], axis=2).min(axis=1)
    assert np.all(gap < 4.0 * spacing)


def test_no_delta_no_distributaries(fields, synthetic_globe, plain_rivers):
    rivers = build_region_rivers(fields, synthetic_globe, PARAMS)
    assert len(rivers.paths) == len(plain_rivers.paths)


def test_mouth_count_grows_with_flow():
    p = DistributaryParams()
    counts = [mouth_count(q, p) for q in (500.0, 1000.0, 10_000.0, 100_000.0, 1e7)]
    assert counts == sorted(counts)
    assert counts[1] == p.base_count
    assert counts[-1] == p.max_count
