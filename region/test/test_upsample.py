"""The critical invariant: a region refines the globe, it never replaces it."""
import numpy as np
import pytest

from region.upsample import (
    RegionUpsampleParams,
    build_region_fields,
    conservation_report,
    global_slope,
)

REGION_ID = 20
TEX_SIZE = 16


def _params(**overrides) -> RegionUpsampleParams:
    base = dict(n_cols=6, tex_size=TEX_SIZE)
    base.update(overrides)
    return RegionUpsampleParams(**base)


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    return build_region_fields(synthetic_globe, REGION_ID, _params())


def test_shapes_and_keys(fields):
    m, t = fields.n_hexes, fields.tex_size
    assert t == TEX_SIZE and m > 0
    assert fields.positions.shape == (m, t, t, 3)
    assert fields.in_hex.shape == (m, t, t)
    assert fields.anchors.shape == (m, t, t)
    assert fields.textures["elevation"].shape == (m, t, t, 1)
    assert fields.textures["temperature"].shape == (m, t, t, 4)
    assert fields.textures["precipitation"].shape == (m, t, t, 4)
    assert fields.textures["biome"].shape == (m, t, t, 1)


def test_downsampling_recovers_the_global_map(fields, synthetic_globe):
    """Averaging each global texel's regional samples must return its value."""
    report = conservation_report(fields, synthetic_globe)
    assert report["elevation"]["n_anchors"] > 100
    assert report["elevation"]["max"] < 1e-3          # km
    assert report["temperature"]["max"] < 1e-3        # °C
    assert report["precipitation"]["max"] < 1e-1      # mm/season


def test_detail_is_actually_added(fields, synthetic_globe):
    """Refined elevation must vary *within* a global texel, not just tile it."""
    assert fields.detail_octaves >= 1
    flat = fields.textures["elevation"].reshape(-1)
    anchors = fields.anchors.reshape(-1)
    inside = fields.in_hex.reshape(-1)
    spreads = []
    for anchor in np.unique(anchors[inside])[:200]:
        group = flat[inside & (anchors == anchor)]
        if len(group) > 4:
            spreads.append(group.std())
    assert np.mean(spreads) > 1e-3


def test_no_detail_when_relief_is_disabled(synthetic_globe):
    flat = build_region_fields(
        synthetic_globe, REGION_ID, _params(relief_factor=0.0, precip_relative_amplitude=0.0)
    )
    bumpy = build_region_fields(synthetic_globe, REGION_ID, _params())
    assert flat.textures["elevation"].std() < bumpy.textures["elevation"].std()
    # Both still satisfy conservation.
    assert conservation_report(flat, synthetic_globe)["elevation"]["max"] < 1e-3


def test_detail_follows_slope(synthetic_globe):
    """Steeper ground gets more added relief than flat ground."""
    params = _params()
    fields = build_region_fields(synthetic_globe, REGION_ID, params)
    smooth = build_region_fields(synthetic_globe, REGION_ID, _params(relief_factor=0.0))
    detail = np.abs(fields.textures["elevation"] - smooth.textures["elevation"]).reshape(-1)

    from region.sampling import GlobalSampler
    sampler = GlobalSampler(synthetic_globe.mesh)
    faces, u, v = sampler.locate(fields.positions.reshape(-1, 3))
    slope = sampler.sample(
        global_slope(synthetic_globe.mesh, synthetic_globe.mesh.face_textures["elevation"]),
        faces, u, v,
    )[:, 0]
    steep = slope > np.percentile(slope, 75)
    gentle = slope < np.percentile(slope, 25)
    assert detail[steep].mean() > detail[gentle].mean()


def test_masks_come_verbatim_from_the_anchor(fields, synthetic_globe):
    """The coastline is a global fact — regional texels copy it exactly."""
    ocean = synthetic_globe.mesh.face_textures["ocean_mask"].reshape(-1, 1)
    expected = ocean[fields.anchors.reshape(-1), 0]
    assert np.array_equal(fields.textures["ocean_mask"].reshape(-1), expected)


def test_biome_indices_are_valid(fields):
    from worldgen.climate.classification import N_BIOMES
    biome = fields.textures["biome"]
    assert biome.dtype == np.uint8
    assert biome.min() >= 0 and biome.max() < N_BIOMES


def test_generation_is_deterministic(synthetic_globe):
    a = build_region_fields(synthetic_globe, REGION_ID, _params())
    b = build_region_fields(synthetic_globe, REGION_ID, _params())
    assert np.array_equal(a.textures["elevation"], b.textures["elevation"])
    assert np.array_equal(a.textures["precipitation"], b.textures["precipitation"])


def test_neighbouring_regions_do_not_share_detail(synthetic_globe):
    """Different regions are different worlds' worth of detail, not a repeat."""
    a = build_region_fields(synthetic_globe, REGION_ID, _params())
    b = build_region_fields(synthetic_globe, REGION_ID + 1, _params())
    assert not np.allclose(
        a.textures["elevation"][: b.n_hexes], b.textures["elevation"][: a.n_hexes]
    )


def test_temperature_follows_the_refined_relief(synthetic_globe):
    """Lapse rate must make the high texels the cold ones inside a global texel."""
    fields = build_region_fields(synthetic_globe, REGION_ID, _params())
    elev = fields.textures["elevation"].reshape(-1)
    temp = fields.textures["temperature"].reshape(-1, 4).mean(axis=1)
    anchors = fields.anchors.reshape(-1)
    inside = fields.in_hex.reshape(-1)
    correlations = []
    for anchor in np.unique(anchors[inside])[:100]:
        sel = inside & (anchors == anchor)
        if sel.sum() > 8 and elev[sel].std() > 1e-4 and temp[sel].std() > 1e-6:
            correlations.append(np.corrcoef(elev[sel], temp[sel])[0, 1])
    assert len(correlations) > 5
    assert np.mean(correlations) < -0.5


def test_no_detail_band_when_region_is_not_finer(synthetic_globe):
    """A regional grid coarser than the globe gets no invented structure."""
    coarse = build_region_fields(synthetic_globe, REGION_ID, _params(n_cols=1, tex_size=4))
    assert coarse.detail_octaves == 0


def test_missing_elevation_is_an_error(synthetic_globe):
    from core.world import World
    from core.mesh import Mesh
    bare = World.from_mesh(Mesh.icosphere(1).mesh_dual())
    with pytest.raises(ValueError, match="elevation"):
        build_region_fields(bare, REGION_ID, _params())
