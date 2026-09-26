"""Region layers: refined where the upsampler covers them, sampled from the globe otherwise."""
import dataclasses

import numpy as np
import pytest

from region.display import REFINED_LAYERS, build_layer_texture, has_refined_layer
from region.upsample import RegionUpsampleParams, build_region_fields

REGION_ID = 40   # has land (the conftest lake sits on its highland)


@pytest.fixture(scope="module")
def fields(synthetic_globe):
    return build_region_fields(synthetic_globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16))


@pytest.mark.parametrize("layer", REFINED_LAYERS)
def test_refined_layers_render(fields, layer):
    assert has_refined_layer(fields, layer)
    rgba = build_layer_texture(fields, layer)
    assert rgba.shape == (fields.n_hexes, fields.tex_size, fields.tex_size, 4)
    assert np.isfinite(rgba).all()


def test_seasonal_temperature_layers_read_their_season(fields):
    # The synthetic climate has identical seasons 1 and 3; make summer distinct.
    temperature = fields.textures["temperature"].copy()
    temperature[..., 1] += 30.0
    hot = dataclasses.replace(fields, textures={**fields.textures, "temperature": temperature})
    summer = build_layer_texture(hot, "Summer Temperature")
    winter = build_layer_texture(hot, "Winter Temperature")
    assert not np.allclose(summer, winter)
    assert np.allclose(winter, build_layer_texture(fields, "Winter Temperature"))


def test_unrefined_layer_reads_the_global_texture_at_the_anchor(fields, synthetic_globe):
    mesh = synthetic_globe.mesh
    n, h, w = mesh.face_textures["elevation"].shape[:3]
    # Encode each face's index in its red channel so the sample is checkable.
    code = (np.arange(n, dtype=np.float32) / n)[:, None, None] * np.ones((h, w), dtype=np.float32)
    rgba = np.stack([code, np.zeros_like(code), np.zeros_like(code), np.ones_like(code)], axis=-1)
    mesh.set_face_textures("face_code_rgba", rgba, n_channels=4)
    try:
        out = build_layer_texture(fields, "Face Code", mesh, texture_key="face_code_rgba")
    finally:
        del mesh.face_textures["face_code_rgba"]
    assert out.shape == (fields.n_hexes, fields.tex_size, fields.tex_size, 4)
    assert np.allclose(out[..., 0], fields.anchors // (h * w) / n)


def test_unrefined_layer_without_texture_colors_the_owning_face(fields, synthetic_globe):
    mesh = synthetic_globe.mesh
    n = len(mesh.face_metadata)
    out = build_layer_texture(
        fields, "Plates", mesh, scheme_fn=lambda meta: (0.25, 0.5, 0.75, 1.0)
    )
    assert np.allclose(out, np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32))
    assert fields.faces.max() < n


def test_unrefined_layer_needs_the_globe(fields):
    with pytest.raises(ValueError):
        build_layer_texture(fields, "Polities")
