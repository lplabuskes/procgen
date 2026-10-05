"""Unit tests for `worldgen.resources.forest`."""
from __future__ import annotations

from typing import Any

import numpy as np

from worldgen.resources.forest import (
    BIOME_FOREST_BASELINE,
    BIOME_PRECIP_REFERENCE_MM,
    compute_forest_density,
)
from worldgen.resources.builder import ForestDensityParams


TUNDRA = 1
TEMPERATE = 3
TROPICAL = 6
DESERT = 8


def _default_kwargs(**overrides) -> dict[str, Any]:
    """``compute_forest_density`` kwargs (``params`` + ``seed``), with ``overrides`` setting fields."""
    seed = overrides.pop("seed", 42)
    defaults = dict(
        precipitation_modulation_strength=0.3,
        treeline_elevation_km=3.0,
        treeline_falloff_km=0.5,
        riparian_bonus=0.10,
        riparian_decay_per_hop=0.5,
        riparian_max_hops=2,
        noise_amplitude=0.0,  # disable noise by default in tests
        noise_octaves=3,
    )
    defaults.update(overrides)
    return dict(params=ForestDensityParams(**defaults), seed=seed)  # type: ignore[arg-type]


def _build_inputs(
    n=1,
    h=4,
    w=4,
    k=4,
    p_total_mm=900.0,
    elev_km=0.5,
    biome=TEMPERATE,
    ocean=False,
    riparian_hops_val=100,
):
    precip = np.full((n, h, w, k), p_total_mm / k, dtype=np.float32)
    elev = np.full((n, h, w, 1), elev_km, dtype=np.float32)
    ocean_mask = np.full((n, h, w, 1), 1.0 if ocean else 0.0, dtype=np.float32)
    face_biome = np.full(n, biome, dtype=np.int32)
    riparian = np.full(n, riparian_hops_val, dtype=np.int32)
    return precip, elev, ocean_mask, face_biome, riparian


class TestComputeForestDensity:
    def test_ocean_pixels_zero(self):
        precip, elev, ocean, biome, rip = _build_inputs(ocean=True)
        f = compute_forest_density(
            precipitation_tex=precip,
            elevation_tex=elev,
            ocean_mask_tex=ocean,
            face_biome=biome,
            riparian_hops=rip,
            pixel_positions=None,
            **_default_kwargs(),
        )
        assert (f == 0.0).all()
        assert f.shape == (1, 4, 4, 1)

    def test_desert_biome_zero(self):
        precip, elev, ocean, biome, rip = _build_inputs(biome=DESERT, p_total_mm=100.0)
        f = compute_forest_density(
            precipitation_tex=precip, elevation_tex=elev, ocean_mask_tex=ocean,
            face_biome=biome, riparian_hops=rip, pixel_positions=None,
            **_default_kwargs(),
        )
        assert (f == 0.0).all()

    def test_temperate_baseline_in_range(self):
        precip, elev, ocean, biome, rip = _build_inputs(
            biome=TEMPERATE, p_total_mm=BIOME_PRECIP_REFERENCE_MM[TEMPERATE],
            elev_km=0.5,
        )
        f = compute_forest_density(
            precipitation_tex=precip, elevation_tex=elev, ocean_mask_tex=ocean,
            face_biome=biome, riparian_hops=rip, pixel_positions=None,
            **_default_kwargs(),
        )
        # At reference P, P_modulation=1; treeline factor ≈ 1 (elev=0.5 well below
        # 3.0km cutoff). Result should ≈ baseline.
        baseline = BIOME_FOREST_BASELINE[TEMPERATE]
        assert abs(f.max() - baseline) < 1e-4

    def test_treeline_zeros_high_elevation(self):
        precip, _, ocean, biome, rip = _build_inputs(
            biome=TEMPERATE, p_total_mm=BIOME_PRECIP_REFERENCE_MM[TEMPERATE]
        )
        # Elevation well above treeline (3.0 + 0.5 falloff = 3.5 km cutoff).
        elev_high = np.full((1, 4, 4, 1), 4.0, dtype=np.float32)
        f = compute_forest_density(
            precipitation_tex=precip, elevation_tex=elev_high, ocean_mask_tex=ocean,
            face_biome=biome, riparian_hops=rip, pixel_positions=None,
            **_default_kwargs(),
        )
        assert (f == 0.0).all()

    def test_p_modulation_monotone_in_p(self):
        biome = TEMPERATE
        elev_km = 0.2
        ocean = np.zeros((1, 4, 4, 1), dtype=np.float32)
        face_biome = np.full(1, biome, dtype=np.int32)
        rip = np.full(1, 100, dtype=np.int32)
        elev = np.full((1, 4, 4, 1), elev_km, dtype=np.float32)

        outs = []
        for p_mm in (200.0, 600.0, 900.0, 1800.0, 3600.0):
            precip = np.full((1, 4, 4, 4), p_mm / 4, dtype=np.float32)
            f = compute_forest_density(
                precipitation_tex=precip, elevation_tex=elev, ocean_mask_tex=ocean,
                face_biome=face_biome, riparian_hops=rip, pixel_positions=None,
                **_default_kwargs(),
            )
            outs.append(float(f.max()))
        # Strictly non-decreasing in P (drier→wetter raises density up to clip).
        for a, b in zip(outs, outs[1:]):
            assert b >= a - 1e-6

    def test_riparian_bonus_raises_density(self):
        precip, elev, ocean, biome, _ = _build_inputs(
            biome=TUNDRA,  # baseline 0.05, leaves room for bonus to register
            p_total_mm=BIOME_PRECIP_REFERENCE_MM[TUNDRA],
            elev_km=0.0,
        )
        far = np.full(1, 100, dtype=np.int32)
        near = np.full(1, 0, dtype=np.int32)
        kw = _default_kwargs()
        f_far = compute_forest_density(
            precipitation_tex=precip, elevation_tex=elev, ocean_mask_tex=ocean,
            face_biome=biome, riparian_hops=far, pixel_positions=None, **kw,
        )
        f_near = compute_forest_density(
            precipitation_tex=precip, elevation_tex=elev, ocean_mask_tex=ocean,
            face_biome=biome, riparian_hops=near, pixel_positions=None, **kw,
        )
        assert f_near.max() > f_far.max()

    def test_noise_bounded(self):
        precip, elev, ocean, biome, rip = _build_inputs(
            biome=TEMPERATE, p_total_mm=BIOME_PRECIP_REFERENCE_MM[TEMPERATE],
            elev_km=0.5,
        )
        # Fake "spread" pixel positions on the unit sphere.
        positions = np.zeros((1, 4, 4, 3), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                positions[0, i, j] = [
                    float(i), float(j), 5.0
                ]
        positions /= np.linalg.norm(positions, axis=-1, keepdims=True)
        kw = _default_kwargs(noise_amplitude=0.05)
        f = compute_forest_density(
            precipitation_tex=precip, elevation_tex=elev, ocean_mask_tex=ocean,
            face_biome=biome, riparian_hops=rip, pixel_positions=positions, **kw,
        )
        baseline = BIOME_FOREST_BASELINE[TEMPERATE]
        # All pixels within [baseline - amp, baseline + amp] after clipping.
        assert f.max() <= baseline + 0.05 + 1e-5
        assert f.min() >= max(0.0, baseline - 0.05 - 1e-5)

    def test_baseline_constants_ordered(self):
        b = BIOME_FOREST_BASELINE
        assert b[TROPICAL] >= b[TEMPERATE]
        assert b[TEMPERATE] >= b[2]            # Temperate ≥ Taiga
        assert b[2] >= b[5]                    # Taiga ≥ Savannah
        assert b[DESERT] == 0.0
        assert b[0] == 0.0                     # Icecap
