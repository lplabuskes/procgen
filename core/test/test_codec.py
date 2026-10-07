"""Save codecs: quantized texture round trips and columnar face metadata."""
import numpy as np
import pytest

from core.codec import (
    TEXTURE_CODECS, TextureCodec, decode_metadata, decode_texture, encode_metadata, encode_texture,
    register_texture_codec,
)
from core.mesh import PolygonMesh
from core.mesh_cache import MeshParams
from core.persistence import load_world, save_world
from core.world import World


def _smooth(rng, shape, scale):
    return (np.cumsum(rng.normal(size=shape), axis=2) * scale).astype(np.float32)


def test_linear_codec_error_is_half_a_step():
    rng = np.random.default_rng(0)
    tex = _smooth(rng, (20, 8, 8, 2), 0.3)
    codec = TextureCodec(step=0.01)
    codes = encode_texture(tex, codec)
    assert codes is not None and codes.dtype.itemsize <= 2
    back = decode_texture(codes, codec, tex.dtype)
    assert back.dtype == np.float32
    assert np.abs(back - tex).max() <= 0.005 + 1e-6


def test_log_codec_error_is_relative():
    rng = np.random.default_rng(1)
    tex = np.exp(rng.uniform(0, 8, size=(10, 4, 4, 4)))
    codec = TextureCodec(step=0.002, log=True, log_offset=1.0)
    codes = encode_texture(tex, codec)
    assert codes is not None
    back = decode_texture(codes, codec, np.dtype(np.float64))
    assert np.all(np.abs(back - tex) <= 0.0011 * (tex + 1.0))
    zero = np.zeros((1, 2, 2, 1))
    zero_codes = encode_texture(zero, codec)
    assert zero_codes is not None
    assert np.all(decode_texture(zero_codes, codec, zero.dtype) == 0.0)


def test_relative_codec_is_exact_off_the_difference():
    rng = np.random.default_rng(2)
    base = _smooth(rng, (5, 6, 6, 1), 0.1)
    filled = base.copy()
    filled[0, 2:4, 2:4] += 0.5
    codec = TextureCodec(step=1e-4, relative_to="base")
    codes = encode_texture(filled, codec, base)
    assert codes is not None and not codes[1:].any()
    back = decode_texture(codes, codec, filled.dtype, base)
    assert np.abs(back - filled).max() <= 5e-5 + 1e-6


def test_non_finite_textures_are_not_coded():
    tex = np.array([[[[1.0], [np.nan]]]], dtype=np.float32)
    assert encode_texture(tex, TextureCodec(step=0.1)) is None


def test_conflicting_registration_raises():
    register_texture_codec("_test_conflict", TextureCodec(step=0.1))
    register_texture_codec("_test_conflict", TextureCodec(step=0.1))
    with pytest.raises(ValueError):
        register_texture_codec("_test_conflict", TextureCodec(step=0.2))
    del TEXTURE_CODECS["_test_conflict"]


def test_metadata_columns_round_trip():
    meta = [
        {"kind": "land", "id": 3, "flag": True, "x": 0.25, "mixed": 1, "extra": ["a"]},
        {"kind": "sea", "id": -40000, "flag": False, "x": 1.5, "mixed": 2.5},
    ]
    arrays: dict = {}
    encode_metadata(meta, arrays)
    assert arrays["meta_kind"].dtype == np.int8
    assert arrays["meta_id"].dtype == np.int32
    assert arrays["meta_x"].dtype == np.float32
    assert "meta_extra" not in arrays
    assert decode_metadata(arrays) == meta


def test_saved_world_round_trips_coded_textures(tmp_path):
    register_texture_codec("_test_height", TextureCodec(step=1e-3))
    register_texture_codec("_test_level", TextureCodec(step=1e-3, relative_to="_test_height"))
    try:
        rng = np.random.default_rng(3)
        mesh = PolygonMesh(np.eye(3, dtype=np.float32), [(0, 1, 2)])
        height = _smooth(rng, (1, 4, 4, 1), 0.2)
        level = np.maximum(height, 0.1).astype(np.float16)
        mesh.set_face_textures("_test_height", height, n_channels=1)
        mesh.set_face_textures("_test_level", level, n_channels=1)
        mesh.set_face_textures("mask", (height > 0).astype(np.uint8), n_channels=1)
        mesh.face_metadata[0].update({"a": 1.5, "b": "x"})
        path = tmp_path / "w.npz"
        save_world(World.from_mesh(mesh), MeshParams(), [], path)
        _, loaded, data = load_world(path, MeshParams)
        assert "texq__test_level" in data.files and "tex_mask" in data.files
        tex = loaded.mesh.face_textures
        assert tex["_test_level"].dtype == np.float16
        assert np.abs(tex["_test_height"] - height).max() <= 5e-4 + 1e-6
        assert np.abs(tex["_test_level"].astype(np.float32) - level).max() <= 2e-3
        assert np.array_equal(tex["mask"], (height > 0).astype(np.uint8))
        assert loaded.mesh.face_metadata[0] == {"a": 1.5, "b": "x"}
    finally:
        del TEXTURE_CODECS["_test_height"], TEXTURE_CODECS["_test_level"]
