"""Coast styling and shore reshaping: the shore changes, the invariants do not."""
import numpy as np
import pytest

from region.sampling import anchor_means
from region.upsample import (
    SEA_LEVEL_KM,
    RegionUpsampleParams,
    _normalized_fbm,
    _varying_persistence_fbm,
    build_region_fields,
    conservation_report,
    reshape_to_target,
)

REGION_ID = 40   # coast *and* a lake (see conftest)
TEX_SIZE = 16


def _params(**overrides) -> RegionUpsampleParams:
    base: dict = dict(n_cols=6, tex_size=TEX_SIZE)
    base.update(overrides)
    return RegionUpsampleParams(**base)


def _build(globe, smoothness, **overrides):
    """Region fields with a uniform ``coast_smoothness`` on the globe (None: no field)."""
    tex = globe.mesh.face_textures
    shape = tex["elevation"].shape
    if smoothness is not None:
        tex["coast_smoothness"] = np.full(shape, smoothness, dtype=np.float32)
    try:
        return build_region_fields(globe, REGION_ID, _params(**overrides))
    finally:
        tex.pop("coast_smoothness", None)


def _shore_edges(fields) -> int:
    water = (fields.textures["ocean_mask"][..., 0] > 0.5) | (fields.textures["lake_mask"][..., 0] > 0.5)
    inside = fields.in_hex
    rows = (water[:, 1:] != water[:, :-1]) & inside[:, 1:] & inside[:, :-1]
    cols = (water[:, :, 1:] != water[:, :, :-1]) & inside[:, :, 1:] & inside[:, :, :-1]
    return int(rows.sum() + cols.sum())


# ── reshape_to_target ────────────────────────────────────────────────────────


def _random_groups(seed: int = 0, n_anchors: int = 40, per: int = 25):
    rng = np.random.default_rng(seed)
    anchors = np.repeat(np.arange(n_anchors), per)
    elevation = rng.normal(0.0, 0.05, anchors.size)
    target = elevation + rng.normal(0.0, 0.05, anchors.size)
    weights = rng.uniform(0.5, 1.0, anchors.size)
    return elevation, target, anchors, weights, n_anchors


def test_reshape_keeps_every_anchor_mean():
    e, t, anchors, w, n = _random_groups()
    out = reshape_to_target(e, t, anchors, w, n, max_scale=10.0)
    before, _ = anchor_means(e[:, None], anchors, w, n)
    after, _ = anchor_means(out[:, None], anchors, w, n)
    assert np.abs(after - before).max() < 1e-12


def test_reshape_matches_the_target_side_where_it_edits():
    e, t, anchors, w, n = _random_groups()
    out = reshape_to_target(e, t, anchors, w, n, max_scale=10.0)
    edited = np.unique(anchors[out != e])
    assert edited.size > 0
    in_edited = np.isin(anchors, edited)
    # In every anchor it touched, each texel is on the target's side now.
    assert np.array_equal((out <= 0.0)[in_edited], (t <= 0.0)[in_edited])


def test_reshape_leaves_an_anchor_alone_when_it_cannot_absorb_the_change():
    e, t, anchors, w, n = _random_groups()
    out = reshape_to_target(e, t, anchors, w, n, max_scale=0.0)
    assert np.array_equal(out, e)


def test_reshape_respects_a_per_texel_level_and_the_active_mask():
    e, t, anchors, w, n = _random_groups()
    level = np.full(e.size, 0.02)
    level[: e.size // 2] = np.inf                         # no water surface known here
    active = np.ones(e.size, dtype=bool)
    active[e.size // 2: 3 * e.size // 4] = False
    out = reshape_to_target(e + 0.02, t + 0.02, anchors, w, n, level=level, active=active, max_scale=10.0)
    untouched = ~np.isfinite(level) | ~active
    assert np.array_equal(out[untouched], (e + 0.02)[untouched])
    changed = ~untouched & (out != e + 0.02)
    assert changed.any()
    assert np.array_equal((out <= 0.02)[changed], (t + 0.02 <= 0.02)[changed])


def test_constant_persistence_matches_plain_fbm():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(200, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    freqs = np.array([20.0, 40.0, 80.0])
    plain = _normalized_fbm(points, 5, freqs, 0.5 ** np.arange(3))
    varying = _varying_persistence_fbm(points, 5, freqs, np.full(len(points), 0.5))
    assert np.allclose(plain, varying)


# ── refine_fields with a coast field ─────────────────────────────────────────


@pytest.mark.parametrize("smoothness", [0.0, 1.0])
def test_conservation_and_the_water_invariants_hold(synthetic_globe, smoothness):
    fields = _build(synthetic_globe, smoothness)
    assert conservation_report(fields, synthetic_globe)["elevation"]["max"] < 1e-3
    elev = fields.textures["elevation"].reshape(-1)
    ocean = fields.textures["ocean_mask"].reshape(-1) > 0.5
    lake = fields.textures["lake_mask"].reshape(-1) > 0.5
    assert np.all(elev[ocean] <= SEA_LEVEL_KM)
    assert not (ocean & lake).any()
    assert lake.sum() > 50


def test_the_field_is_passed_down(synthetic_globe):
    fields = _build(synthetic_globe, 0.3)
    assert np.allclose(fields.textures["coast_smoothness"], 0.3)
    assert "coast_smoothness" not in _build(synthetic_globe, None).textures


def test_without_a_field_nothing_changes(synthetic_globe):
    a = _build(synthetic_globe, None)
    b = _build(synthetic_globe, None, reshape_coast=False)
    assert np.array_equal(a.textures["elevation"], b.textures["elevation"])


def test_smooth_shores_are_shorter_than_rugged_ones(synthetic_globe):
    assert _shore_edges(_build(synthetic_globe, 1.0)) < _shore_edges(_build(synthetic_globe, 0.0))


def test_reshaping_moves_the_shore_and_only_near_water(synthetic_globe):
    on = _build(synthetic_globe, 0.5)
    off = _build(synthetic_globe, 0.5, reshape_coast=False)
    changed = (on.textures["elevation"] != off.textures["elevation"]).reshape(-1)
    assert changed.any()
    # Edits stay inside the anchors (global texels) the shore band reaches.
    from region.source import GlobalSource
    source = GlobalSource(synthetic_globe)
    loc = source.locate(on.positions.reshape(-1, 3))
    water = np.clip(source.coarse("ocean_mask", loc)[:, 0] + source.coarse("lake_mask", loc)[:, 0], 0.0, 1.0)
    band_anchors = np.unique(loc.anchors[(water > 0.0) & (water < 1.0)])
    assert np.all(np.isin(loc.anchors[changed], band_anchors))
