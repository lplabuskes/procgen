"""Region layers: rebuilt by the stage's own recipe where refined, sampled from the globe otherwise."""
import dataclasses

import numpy as np
import pytest

from core.display import SchemeSpec, TextureRecipe
from region.display import build_layer_texture, describe_texel, has_refined_layer
from region.upsample import RegionUpsampleParams, build_region_fields
from worldgen.climate.display import (
    ANNUAL_TEMPERATURE_RECIPE,
    BIOME_RECIPE,
    PRECIPITATION_RECIPE,
    SUMMER_TEMPERATURE_RECIPE,
    WINTER_TEMPERATURE_RECIPE,
)
from worldgen.geology.display import ELEVATION_RECIPE

REGION_ID = 40   # has land (the conftest lake sits on its highland)

RECIPES = [
    ELEVATION_RECIPE, BIOME_RECIPE, ANNUAL_TEMPERATURE_RECIPE,
    SUMMER_TEMPERATURE_RECIPE, WINTER_TEMPERATURE_RECIPE, PRECIPITATION_RECIPE,
]


def _flat(_meta):
    return (0.25, 0.5, 0.75, 1.0)


def _spec(recipe=None, texture=None) -> SchemeSpec:
    return SchemeSpec(_flat, texture=texture, recipe=recipe)


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    return build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16))


@pytest.mark.parametrize("recipe", RECIPES)
def test_world_recipes_render_refined_fields(fields, recipe):
    spec = _spec(recipe)
    assert has_refined_layer(fields, spec)
    rgba = build_layer_texture(fields, spec)
    assert rgba.shape == (fields.n_hexes, fields.tex_size, fields.tex_size, 4)
    assert np.isfinite(rgba).all()


@pytest.mark.parametrize("recipe", [r for r in RECIPES if not r.relief])
def test_flat_recipe_output_is_the_recipe_itself(fields, recipe):
    """No region-side colouring: the layer is exactly the stage's function on refined data."""
    assert np.array_equal(build_layer_texture(fields, _spec(recipe)), recipe(fields.textures))


@pytest.mark.parametrize("recipe", [r for r in RECIPES if r.relief])
def test_relief_recipe_is_hillshaded_on_land_only(fields, recipe):
    raw = recipe(fields.textures)
    out = build_layer_texture(fields, _spec(recipe))
    water = (fields.textures["ocean_mask"][..., 0] > 0) | (fields.textures["lake_mask"][..., 0] > 0)
    assert water.any() and (~water).any()
    assert np.array_equal(out[water], raw[water])
    assert not np.allclose(out[~water], raw[~water])


def test_seasonal_temperature_layers_read_their_season(fields):
    # The synthetic climate has identical seasons 1 and 3; make summer distinct.
    temperature = fields.textures["temperature"].copy()
    temperature[..., 1] += 30.0
    hot = dataclasses.replace(fields, textures={**fields.textures, "temperature": temperature})
    summer = build_layer_texture(hot, _spec(SUMMER_TEMPERATURE_RECIPE))
    winter = build_layer_texture(hot, _spec(WINTER_TEMPERATURE_RECIPE))
    assert not np.allclose(summer, winter)
    assert np.allclose(winter, build_layer_texture(fields, _spec(WINTER_TEMPERATURE_RECIPE)))


def test_recipe_on_unrefined_sources_falls_back_to_the_globe(fields, synthetic_globe):
    recipe = TextureRecipe(lambda tex: np.zeros((1, 1, 1, 4)), sources=("fertility",))
    spec = _spec(recipe)
    assert not has_refined_layer(fields, spec)
    out = build_layer_texture(fields, spec, synthetic_globe.mesh)
    assert np.allclose(out, np.array(_flat({}), dtype=np.float32))


def test_unrefined_layer_reads_the_global_texture_at_the_anchor(fields, synthetic_globe):
    mesh = synthetic_globe.mesh
    n, h, w = mesh.face_textures["elevation"].shape[:3]
    # Encode each face's index in its red channel so the sample is checkable.
    code = (np.arange(n, dtype=np.float32) / n)[:, None, None] * np.ones((h, w), dtype=np.float32)
    rgba = np.stack([code, np.zeros_like(code), np.zeros_like(code), np.ones_like(code)], axis=-1)
    mesh.set_face_textures("face_code_rgba", rgba, n_channels=4)
    try:
        out = build_layer_texture(fields, _spec(texture="face_code_rgba"), mesh)
    finally:
        del mesh.face_textures["face_code_rgba"]
    assert out.shape == (fields.n_hexes, fields.tex_size, fields.tex_size, 4)
    assert np.allclose(out[..., 0], fields.anchors // (h * w) / n)


def test_unrefined_layer_without_texture_colors_the_owning_face(fields, synthetic_globe):
    mesh = synthetic_globe.mesh
    out = build_layer_texture(fields, _spec(), mesh)
    assert np.allclose(out, np.array(_flat({}), dtype=np.float32))
    assert fields.faces.max() < len(mesh.face_metadata)


def test_unrefined_layer_needs_the_globe(fields):
    with pytest.raises(ValueError):
        build_layer_texture(fields, _spec())


ALL_REFINED = ("elevation", "biome", "temperature", "precipitation")


def test_describe_texel_reports_the_refined_values(fields):
    hex_idx, row, col = (int(x) for x in np.argwhere(fields.in_hex)[0])
    parts = describe_texel(fields, hex_idx, row, col, ALL_REFINED)
    assert parts is not None
    elev = float(fields.textures["elevation"][hex_idx, row, col, 0])
    assert f"[px: {elev:.2f} km]" in parts
    assert any(p.startswith("T: ") for p in parts)
    assert any(p.startswith("P: ") for p in parts)


def test_describe_texel_reports_only_the_requested_keys(fields):
    assert describe_texel(fields, 0, 0, 0, ("elevation",)) == [
        f"[px: {float(fields.textures['elevation'][0, 0, 0, 0]):.2f} km]"
    ]


def test_describe_texel_defers_unrefined_keys_to_the_globe(fields):
    assert describe_texel(fields, 0, 0, 0, ("fertility",)) is None
    assert describe_texel(fields, 0, 0, 0, ("elevation", "fertility")) is None
    assert describe_texel(fields, 0, 0, 0, ()) is None


def test_describe_texel_names_water(fields):
    ocean = np.argwhere(fields.textures["ocean_mask"][..., 0] > 0)
    assert len(ocean)
    hex_idx, row, col = (int(x) for x in ocean[0])
    parts = describe_texel(fields, hex_idx, row, col, ("biome", "temperature", "precipitation"))
    assert parts is not None and parts[0] == "Ocean"


def test_recipe_can_read_texel_positions(fields):
    """A layer that is a function of position (the polity fill) is evaluated at the
    region's own texels, handed to its recipe under ``POSITIONS_KEY``."""
    from core.display import POSITIONS_KEY

    def fn(textures):
        p = textures[POSITIONS_KEY]
        return np.concatenate([0.5 + 0.5 * p, np.ones(p.shape[:-1] + (1,))], axis=-1)

    spec = _spec(TextureRecipe(fn, sources=(POSITIONS_KEY,)))
    assert has_refined_layer(fields, spec)
    rgba = build_layer_texture(fields, spec)
    assert np.allclose(rgba[..., :3], 0.5 + 0.5 * fields.positions)
