"""The critical invariant: a region refines the globe, it never replaces it."""
import numpy as np
import pytest

from region.upsample import (
    RegionUpsampleParams,
    _box_filter,
    build_region_fields,
    conservation_report,
    flood_to_level,
)
from region.source import global_slope

REGION_ID = 40   # the region conftest carves its lake into: coast *and* a lake
BIOME_REGION = 20  # region 40's synthetic climate is all one biome; this one is not
TEX_SIZE = 16


def _params(**overrides) -> RegionUpsampleParams:
    base: dict = dict(n_cols=6, tex_size=TEX_SIZE)
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


def test_texel_area_matches_the_cell_not_the_hex_overspill(fields):
    # The hexes spill past the cell, so in-hex texels cover more than one cell:
    # cell area / in-hex count would undercount each texel's area.
    import math
    from region.geometry import N_REGIONS
    from region.upsample import EARTH_RADIUS_KM
    cell_km2 = 4.0 * math.pi * EARTH_RADIUS_KM ** 2 / N_REGIONS
    covered_km2 = fields.texel_area_km2 * fields.in_hex.sum()
    assert covered_km2 > cell_km2
    assert fields.texel_area_km2 == pytest.approx(cell_km2 * (2.0 * fields.grid.radius / TEX_SIZE) ** 2)


def test_a_world_without_resources_refines_none(fields):
    """The synthetic globe has no resource layers or danger: none are invented."""
    for key in ("fertility", "forest_density", "dominant_crop", "rural_density", "danger"):
        assert key not in fields.textures, key


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
        synthetic_globe, REGION_ID, _params(relief_factor=0.0, orographic_gain=0.0)
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


def _anchor_mask(fields, globe, key) -> np.ndarray:
    """The global mask, read at each regional texel's anchor — the unrefined answer."""
    tex = np.asarray(globe.mesh.face_textures[key]).reshape(-1, 1)
    return tex[fields.anchors.reshape(-1), 0] > 0.5


def test_masks_are_verbatim_when_refinement_is_off(synthetic_globe):
    """``refine_water_masks=False`` restores the blocky global coastline."""
    fields = build_region_fields(
        synthetic_globe, REGION_ID, _params(refine_water_masks=False)
    )
    for key in ("ocean_mask", "lake_mask"):
        assert np.array_equal(
            fields.textures[key].reshape(-1) > 0.5, _anchor_mask(fields, synthetic_globe, key)
        )


def test_the_ocean_ends_at_sea_level_on_the_refined_relief(fields):
    """Water may only stand where the *refined* ground is below its surface."""
    from region.upsample import SEA_LEVEL_KM
    elev = fields.textures["elevation"].reshape(-1)
    ocean = fields.textures["ocean_mask"].reshape(-1) > 0.5
    assert np.all(elev[ocean] <= SEA_LEVEL_KM)


def test_the_coastline_moves_both_ways(fields, synthetic_globe):
    """Sub-grid rises dry out global ocean texels; sub-grid hollows flood land ones."""
    anchor = _anchor_mask(fields, synthetic_globe, "ocean_mask")
    ocean = fields.textures["ocean_mask"].reshape(-1) > 0.5
    assert (anchor & ~ocean).sum() > 0        # islands and spits
    assert (~anchor & ocean).sum() > 0        # bays and inlets
    # …but the shore only *moves*, it does not migrate: the two maps still agree
    # almost everywhere.
    assert (anchor != ocean).mean() < 0.05


def test_water_does_not_advance_when_the_spread_is_zero(synthetic_globe):
    """Zero spread still dries islands, but nothing the globe called land floods."""
    fields = build_region_fields(synthetic_globe, REGION_ID, _params(water_spread_texels=0))
    for key in ("ocean_mask", "lake_mask"):
        water = fields.textures[key].reshape(-1) > 0.5
        assert np.all(_anchor_mask(fields, synthetic_globe, key)[water])


def test_the_lake_keeps_its_global_surface(fields, synthetic_globe):
    """A lake's fill level is a global fact; only its shoreline is refined."""
    lake = fields.textures["lake_mask"].reshape(-1) > 0.5
    anchor = _anchor_mask(fields, synthetic_globe, "lake_mask")
    assert anchor.sum() > 50 and lake.sum() > 50
    surface = np.asarray(synthetic_globe.mesh.face_textures["filled_height"]).reshape(-1)[
        np.asarray(synthetic_globe.mesh.face_textures["lake_mask"]).reshape(-1) > 0.5
    ].max()
    assert np.all(fields.textures["elevation"].reshape(-1)[lake] <= surface)
    assert (lake != anchor).sum() > 0                     # the shore did move
    assert 0.5 < lake.sum() / anchor.sum() < 2.0          # but the lake is still the lake


def test_lakes_fall_back_to_the_anchor_without_a_fill_height(synthetic_globe):
    """A world with no hydrology stage has no lake surface to flood to."""
    import copy
    bare = copy.copy(synthetic_globe)
    bare.mesh = copy.copy(synthetic_globe.mesh)
    bare.mesh.face_textures = {
        k: v for k, v in synthetic_globe.mesh.face_textures.items() if k != "filled_height"
    }
    fields = build_region_fields(bare, REGION_ID, _params())
    assert np.array_equal(
        fields.textures["lake_mask"].reshape(-1) > 0.5, _anchor_mask(fields, bare, "lake_mask")
    )


def test_shallow_water_damping_only_touches_submerged_ground():
    from region.upsample import _shallow_water_damping
    params = _params(water_relief_depth_km=0.1)
    depth = np.array([np.inf, 1.0, 0.1, 0.05, 0.0, -0.2])   # inf = dry land
    damping = _shallow_water_damping(depth, params)
    assert damping.tolist() == [1.0, 1.0, 1.0, 0.5, 0.0, 0.0]
    # Disabled, every texel gets the full amplitude again.
    assert np.all(_shallow_water_damping(depth, _params(water_relief_depth_km=0.0)) == 1.0)


def test_shallow_lake_beds_do_not_dither(synthetic_globe):
    """The whole point of the fade: a lake keeps its body, not just its outline.

    Undamped noise re-emerges through the shallow margins of a lake and the mask,
    being a threshold on the refined elevation, breaks the water into speckle.
    """
    def dried(**overrides) -> float:
        f = build_region_fields(synthetic_globe, REGION_ID, _params(**overrides))
        anchor = _anchor_mask(f, synthetic_globe, "lake_mask")
        return float((anchor & ~(f.textures["lake_mask"].reshape(-1) > 0.5)).mean())

    assert dried() < dried(water_relief_depth_km=0.0)


def test_ocean_and_lake_never_overlap(fields):
    ocean = fields.textures["ocean_mask"] > 0.5
    lake = fields.textures["lake_mask"] > 0.5
    assert not np.any(ocean & lake)


def test_flood_to_level_stops_at_a_ridge():
    """Water crosses low ground but not a bar above its surface, and only so far."""
    elevation = np.array([[[-1.0, -0.5, 0.4, -0.2, -0.3, -0.4]]])   # (1, 1, 6)
    seed = np.array([[[True, False, False, False, False, False]]])
    level = np.zeros_like(elevation)
    filled = flood_to_level(elevation, seed, level, iterations=10)
    assert filled.tolist() == [[[True, True, False, False, False, False]]]
    # Lower the bar and the water runs the length of the valley…
    elevation[0, 0, 2] = -0.1
    assert flood_to_level(elevation, seed, level, 10).all()
    # …but no further than `iterations` texels per pass.
    assert flood_to_level(elevation, seed, level, 2).tolist() == [
        [[True, True, True, False, False, False]]
    ]


def test_biome_indices_are_valid(fields):
    from worldgen.climate.classification import N_BIOMES
    biome = fields.textures["biome"]
    assert biome.dtype == np.uint8
    assert biome.min() >= 0 and biome.max() < N_BIOMES


def _biome_edge_fraction(fields) -> float:
    """Fraction of interior texels adjacent to a biome change — boundary raggedness."""
    b = fields.textures["biome"][..., 0].astype(np.int16)
    c = b[:, 1:-1, 1:-1]
    differs = (
        (c != b[:, :-2, 1:-1]) | (c != b[:, 2:, 1:-1])
        | (c != b[:, 1:-1, :-2]) | (c != b[:, 1:-1, 2:])
    )
    core = fields.in_hex[:, 1:-1, 1:-1]
    return float(differs[core].mean())


def test_biome_smoothing_cleans_the_boundaries(synthetic_globe):
    """Without it the refined climate frays every biome edge into a dithered band.

    The classifier is a hard threshold and the refined temperature swings ~1 °C
    texel-to-texel, so near a cut neighbouring texels land on opposite sides.
    """
    rough = build_region_fields(synthetic_globe, BIOME_REGION, _params(biome_smoothing=0))
    smooth = build_region_fields(synthetic_globe, BIOME_REGION, _params())
    assert _biome_edge_fraction(smooth) < _biome_edge_fraction(rough)


def test_biome_smoothing_leaves_the_climate_layers_sharp(synthetic_globe):
    """Only the classifier reads the average — the displayed fields are untouched."""
    rough = build_region_fields(synthetic_globe, REGION_ID, _params(biome_smoothing=0))
    smooth = build_region_fields(synthetic_globe, REGION_ID, _params())
    for key in ("temperature", "precipitation"):
        assert np.array_equal(rough.textures[key], smooth.textures[key])


def test_box_filter_preserves_the_mean_and_is_identity_when_disabled():
    rng = np.random.default_rng(0)
    tex = rng.normal(size=(3, 9, 9, 2))
    assert _box_filter(tex, 1) is tex
    assert _box_filter(tex, 0) is tex
    blurred = _box_filter(tex, 3)
    assert blurred.shape == tex.shape
    assert blurred.std() < tex.std()
    # Edge replication biases the mean only slightly; the interior is exact.
    assert np.allclose(blurred[:, 3:-3, 3:-3].mean(), tex[:, 3:-3, 3:-3].mean(), atol=0.2)


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


def test_precipitation_follows_the_refined_relief(synthetic_globe):
    """Orographic, not random: high texels inside a global texel are the wet ones.

    This is what keeps the biome map from speckling — an independent noise stream
    in precipitation would scatter texels across the classifier's thresholds.
    """
    fields = build_region_fields(synthetic_globe, REGION_ID, _params())
    elev = fields.textures["elevation"].reshape(-1)
    precip = fields.textures["precipitation"].reshape(-1, 4).sum(axis=1)
    anchors = fields.anchors.reshape(-1)
    inside = fields.in_hex.reshape(-1)
    correlations = []
    for anchor in np.unique(anchors[inside])[:100]:
        sel = inside & (anchors == anchor)
        if sel.sum() > 8 and elev[sel].std() > 1e-4 and precip[sel].std() > 1e-6:
            correlations.append(np.corrcoef(elev[sel], precip[sel])[0, 1])
    assert len(correlations) > 5
    assert np.mean(correlations) > 0.9


def test_precipitation_detail_can_be_disabled(synthetic_globe):
    """orographic_gain=0 leaves precipitation as the smooth coarse field."""
    fields = build_region_fields(synthetic_globe, REGION_ID, _params(orographic_gain=0.0))
    bumpy = build_region_fields(synthetic_globe, REGION_ID, _params())
    assert fields.textures["precipitation"].std() < bumpy.textures["precipitation"].std()
    assert conservation_report(fields, synthetic_globe)["precipitation"]["max"] < 1e-1


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
