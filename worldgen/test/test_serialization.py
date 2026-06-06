"""Round-trip + back-compat tests for the decomposed save/load framework.

These run against a committed legacy fixture (``saved_worlds/test.npz``, saved
before the cost stage existed) rather than a full pipeline build, so they stay
fast while exercising every stage's ``load`` and ``serialize`` path:

* ``test_load_legacy_world`` — the per-stage ``load`` loop rehydrates a globe
  saved by an older code path, and the cost stage's absent-key back-compat
  branch is taken silently.
* ``test_resave_roundtrip`` — re-serializing the loaded globe and loading it
  again reproduces the mesh textures, face metadata, and feature set exactly,
  covering both ``serialize`` and ``load`` for every stage with data present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worldgen import load_globe, save_globe

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_WORLD = REPO_ROOT / "saved_worlds" / "test.npz"


pytestmark = pytest.mark.skipif(
    not LEGACY_WORLD.exists(), reason="legacy fixture saved_worlds/test.npz not present"
)


def _non_display_textures(mesh) -> dict:
    return {k: v for k, v in mesh.face_textures.items() if not k.endswith("_rgba")}


def test_load_legacy_world():
    builder, globe = load_globe(LEGACY_WORLD)

    # Mesh + per-stage builder state rehydrated.
    assert globe.mesh is not None
    n = len(globe.mesh.faces)
    assert len(builder.elevation_builder.plate_ids) == n
    assert builder.elevation_builder.max_speed > 0.0
    assert builder.hydrology_builder.rivers  # non-empty

    # The fixture predates the cost stage: its keys are absent, so cost.load
    # takes the back-compat early return and registers no surfaces.
    assert builder.cost_builder.surfaces is None
    assert not globe.has_feature("cost_surfaces")

    # Features other stages own are present.
    for feat in ("rivers", "lakes", "minerals", "winds"):
        assert globe.has_feature(feat), feat


def test_resave_roundtrip(tmp_path):
    builder, globe = load_globe(LEGACY_WORLD)

    out = tmp_path / "roundtrip.npz"
    save_globe(builder, out)
    builder2, globe2 = load_globe(out)

    m1, m2 = globe.mesh, globe2.mesh
    assert len(m1.faces) == len(m2.faces)

    # Non-display textures identical.
    t1, t2 = _non_display_textures(m1), _non_display_textures(m2)
    assert set(t1) == set(t2)
    for k in t1:
        assert np.array_equal(t1[k], t2[k]), f"texture {k} differs"

    # Face metadata identical (floats compared with tolerance).
    assert len(m1.face_metadata) == len(m2.face_metadata)
    for a, b in zip(m1.face_metadata, m2.face_metadata):
        assert set(a) == set(b)
        for key, av in a.items():
            bv = b[key]
            if isinstance(av, float):
                assert av == pytest.approx(bv, abs=1e-6), key
            else:
                assert av == bv, key

    # Same feature set.
    assert set(globe.feature_names()) == set(globe2.feature_names())
