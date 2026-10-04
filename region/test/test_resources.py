"""Refining the resource, rural-density and danger fields."""
import numpy as np
import pytest

from region.resources import FaceFanField, refined_slope
from region.sampling import GlobalSampler, anchor_means
from region.upsample import RegionUpsampleParams, build_region_fields, conservation_report

REGION_ID = 40
TEX_SIZE = 16
_KEYS = ("fertility", "forest_density", "dominant_crop", "rural_density", "fishing_density", "mining_density")


@pytest.fixture(scope="module")
def globe(synthetic_globe):
    """The synthetic globe plus smooth resource layers and a sparse danger field,
    removed again after the module."""
    mesh = synthetic_globe.mesh
    pos = mesh.face_texture_positions(*mesh.face_textures["elevation"].shape[1:3])
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    water = (mesh.face_textures["ocean_mask"][..., 0] > 0.5) | (mesh.face_textures["lake_mask"][..., 0] > 0.5)
    fert = np.where(water, 0.3, 0.5 + 0.3 * np.sin(5.0 * x) * np.cos(3.0 * y)).astype(np.float32)
    forest = np.where(water, 0.0, np.clip(0.6 + 0.4 * np.cos(4.0 * z + x), 0.0, 1.0)).astype(np.float32)
    rho = np.where(water, 0.0, 20.0 * fert ** 2).astype(np.float32)
    mesh.set_face_textures("fertility", fert[..., None], n_channels=1)
    mesh.set_face_textures("forest_density", forest[..., None], n_channels=1)
    mesh.set_face_textures("dominant_crop", np.ones_like(fert, dtype=np.uint8)[..., None], n_channels=1)
    mesh.set_face_textures("rural_density", rho[..., None], n_channels=1)
    mesh.set_face_textures("fishing_density", (0.3 * rho)[..., None], n_channels=1)
    # Miners on a few land texels of the region, each beside its deposit.
    from region.geometry import ecef_to_cell
    flat = pos.reshape(-1, 3)
    unit = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    cand = np.flatnonzero((ecef_to_cell(unit, synthetic_globe.antimeridian) == REGION_ID) & (rho.reshape(-1) > 1.0))
    picks = cand[:: max(len(cand) // 6, 1)][:6]
    mine = np.zeros(rho.size, dtype=np.float32)
    mine[picks] = 0.5 * rho.reshape(-1)[picks]
    mesh.set_face_textures("mining_density", mine.reshape(rho.shape)[..., None], n_channels=1)
    synthetic_globe.add_feature("minerals", (unit[picks], np.zeros(len(picks), dtype=np.int8),
                                             np.ones(len(picks), dtype=np.float32)))
    synthetic_globe.add_feature("mine_workers", np.full(len(picks), 1_000.0))
    rng = np.random.default_rng(5)
    danger = np.where(rng.random(len(mesh.faces)) < 0.3, rng.random(len(mesh.faces)), 0.0).astype(np.float32)
    synthetic_globe.add_feature("danger", danger)
    yield synthetic_globe
    for key in _KEYS:
        mesh.face_textures.pop(key, None)
    synthetic_globe.remove_feature("danger")
    synthetic_globe.remove_feature("minerals")
    synthetic_globe.remove_feature("mine_workers")


@pytest.fixture(scope="module")
def fields(globe):
    return build_region_fields(globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=TEX_SIZE))


def _land(fields) -> np.ndarray:
    t = fields.textures
    return (t["ocean_mask"][..., 0] < 0.5) & (t["lake_mask"][..., 0] < 0.5)


def test_the_resource_layers_are_refined(fields):
    m, t = fields.n_hexes, fields.tex_size
    for key in (*_KEYS, "danger"):
        assert fields.textures[key].shape == (m, t, t, 1), key


def test_rural_people_are_conserved_exactly(fields, globe):
    report = conservation_report(fields, globe)
    assert report["rural_density"]["n_anchors"] > 100
    assert report["rural_density"]["max"] < 1e-4


def test_fishers_are_conserved_and_stay_on_the_shore(fields, globe):
    report = conservation_report(fields, globe)
    assert report["fishing_density"]["n_anchors"] > 100
    assert report["fishing_density"]["max"] < 1e-4
    fish = fields.textures["fishing_density"][..., 0]
    rho = fields.textures["rural_density"][..., 0]
    assert np.all(fish <= rho + 1e-6)
    # Land beyond the shore band holds no fishers wherever its global texel has
    # shore of its own (then the band takes them all).
    from region.resources import _box_mean
    land = _land(fields)
    params = RegionUpsampleParams(n_cols=6, tex_size=TEX_SIZE)
    r = max(int(round(params.fishing_shore_km / np.sqrt(fields.texel_area_km2))), 1)
    near = _box_mean((~land).astype(np.float64), r) > 0
    anchors = fields.anchors
    has_shore = np.zeros(anchors.max() + 1, dtype=bool)
    has_shore[anchors[land & near & fields.in_hex]] = True
    inland = land & ~near & has_shore[anchors]
    assert inland.any()
    assert fish[inland].max() == 0.0


def test_miners_are_conserved_and_gather_by_their_deposit(fields, globe):
    report = conservation_report(fields, globe)
    assert report["mining_density"]["n_anchors"] > 100
    assert report["mining_density"]["max"] < 1e-4
    mine = fields.textures["mining_density"][..., 0].astype(np.float64)
    rho = fields.textures["rural_density"][..., 0].astype(np.float64)
    assert mine.sum() > 0 and np.all(mine <= rho + 1e-6)
    # Within the global texels holding miners, they sit far closer to their
    # deposits than the countryside as a whole does.
    deposits = globe.get_feature("minerals")[0]
    p = fields.positions / np.linalg.norm(fields.positions, axis=-1, keepdims=True)
    d = np.arccos(np.clip(p @ deposits.T, -1.0, 1.0)).min(axis=-1)
    has_mine = np.zeros(fields.anchors.max() + 1, dtype=bool)
    has_mine[fields.anchors[(mine > 0) & fields.in_hex]] = True
    sel = has_mine[fields.anchors] & fields.in_hex
    farm = rho - mine
    assert (mine[sel] * d[sel]).sum() / mine[sel].sum() < 0.5 * (farm[sel] * d[sel]).sum() / farm[sel].sum()


def test_no_one_lives_on_refined_water_that_has_land_beside_it(fields):
    rho = fields.textures["rural_density"][..., 0]
    land = _land(fields)
    anchors = fields.anchors
    # Global texels with some refined land keep all their people on it.
    has_land = np.zeros(anchors.max() + 1, dtype=bool)
    has_land[anchors[land & fields.in_hex]] = True
    wet = ~land & fields.in_hex & has_land[anchors]
    assert rho[wet].max(initial=0.0) == 0.0


def test_fertility_and_forest_are_conserved_over_land(fields, globe):
    report = conservation_report(fields, globe)
    assert report["fertility"]["mean"] < 1e-4
    assert report["forest_density"]["mean"] < 1e-3


def test_fertility_is_higher_on_valley_floors(fields, globe):
    """Within a global texel, ground below the coarse surface is more fertile."""
    mesh = globe.mesh
    sampler = GlobalSampler(mesh)
    coarse = sampler.sample(mesh.face_textures["elevation"], fields.faces.reshape(-1),
                            fields.face_uv[..., 0].reshape(-1), fields.face_uv[..., 1].reshape(-1))[:, 0]
    anomaly = fields.textures["elevation"][..., 0].reshape(-1) - coarse
    fert = fields.textures["fertility"][..., 0].reshape(-1)
    keep = (_land(fields) & fields.in_hex).reshape(-1)
    anchors = fields.anchors.reshape(-1)
    n = int(anchors.max()) + 1
    mean_a, _ = anchor_means(anomaly, anchors, keep.astype(float), n)
    mean_f, _ = anchor_means(fert, anchors, keep.astype(float), n)
    da = anomaly[keep] - mean_a[anchors[keep], 0]
    df = fert[keep] - mean_f[anchors[keep], 0]
    assert np.corrcoef(da, df)[0, 1] < -0.3


def test_bounded_fields_stay_in_range(fields):
    for key in ("fertility", "forest_density"):
        values = fields.textures[key][..., 0]
        assert values.min() >= 0.0 and values.max() <= 1.0, key
    assert fields.textures["forest_density"][..., 0][~_land(fields)].max() == 0.0


def test_danger_keeps_the_smooth_fields_texel_means(fields, globe):
    """The terrain nudge moves danger within a global texel, never its mean."""
    base = FaceFanField(globe.mesh, np.asarray(globe.get_feature("danger"), dtype=np.float64))
    b = base(fields.positions.reshape(-1, 3))
    d = fields.textures["danger"][..., 0].reshape(-1).astype(np.float64)
    anchors = fields.anchors.reshape(-1)
    w = fields.in_hex.reshape(-1).astype(float)
    n = int(anchors.max()) + 1
    mb, counts = anchor_means(b, anchors, w, n)
    md, _ = anchor_means(d, anchors, w, n)
    hit = counts > 0
    assert np.abs(mb[hit] - md[hit]).max() < 1e-5
    assert d.min() >= 0.0


def test_danger_is_higher_on_rugged_ground(globe):
    """With the gain on, steeper texels in the same global texel carry more danger
    (read per texel: the score's area averaging off)."""
    params = RegionUpsampleParams(n_cols=6, tex_size=TEX_SIZE, biome_smoothing=0)
    fields = build_region_fields(globe, REGION_ID, params)
    slope = refined_slope(fields.textures["elevation"][..., 0], np.sqrt(fields.texel_area_km2)).reshape(-1)
    d = fields.textures["danger"][..., 0].reshape(-1)
    keep = (_land(fields) & fields.in_hex).reshape(-1) & (d > 1e-3)
    anchors = fields.anchors.reshape(-1)
    n = int(anchors.max()) + 1
    ms, _ = anchor_means(slope, anchors, keep.astype(float), n)
    md, _ = anchor_means(d, anchors, keep.astype(float), n)
    rel = d[keep] / np.maximum(md[anchors[keep], 0], 1e-9)
    assert np.corrcoef(slope[keep] - ms[anchors[keep], 0], rel)[0, 1] > 0.1


def test_no_terrain_gain_is_the_smooth_field(globe):
    params = RegionUpsampleParams(n_cols=6, tex_size=TEX_SIZE, danger_terrain_gain=0.0)
    f = build_region_fields(globe, REGION_ID, params)
    base = FaceFanField(globe.mesh, np.asarray(globe.get_feature("danger"), dtype=np.float64))
    assert np.allclose(f.textures["danger"][..., 0].reshape(-1), base(f.positions.reshape(-1, 3)), atol=1e-6)


def test_the_refined_layers_draw_from_the_refined_fields(fields):
    from core.display import SchemeSpec
    from realmgen.towns.builder import rural_density_recipe
    from region.display import build_layer_texture, has_refined_layer
    from worldgen.danger.display import DANGER_RECIPE
    from worldgen.resources.display import DOMINANT_CROP_RECIPE, FERTILITY_RECIPE, FOREST_DENSITY_RECIPE
    shape = (*fields.in_hex.shape, 4)
    for recipe in (FERTILITY_RECIPE, FOREST_DENSITY_RECIPE, DOMINANT_CROP_RECIPE, DANGER_RECIPE,
                   rural_density_recipe(20.0)):
        spec = SchemeSpec(lambda meta: (0.0, 0.0, 0.0, 1.0), recipe=recipe)
        assert has_refined_layer(fields, spec)
        rgba = build_layer_texture(fields, spec)
        assert rgba.shape == shape and np.isfinite(rgba).all()
