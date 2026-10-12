"""Unit tests for `worldgen.resources.fertility`."""
from __future__ import annotations

from typing import Any

import numpy as np

from worldgen.resources.fertility import (
    BIOME_FERTILITY_MULT,
    apply_delta_fertility,
    apply_wetland_fertility,
    compute_fertility,
    neighbour_water_fertility,
)
from worldgen.climate.classification import WETLAND_BIOME
from worldgen.resources.builder import FertilityParams


# Biome indices (from worldgen.climate.classification.BIOME_NAMES)
TEMPERATE = 3
DESERT = 8
TROPICAL = 6


def _default_kwargs(**overrides) -> dict[str, Any]:
    """``compute_fertility`` kwargs (``params=<FertilityParams>``), with ``overrides`` setting fields."""
    defaults = dict(
        optimal_temperature_c=18.0,
        temperature_sigma_c=12.0,
        optimal_precipitation_mm=900.0,
        precipitation_sigma_log=0.7,
        elevation_range_threshold_km=2.5,
        floodplain_bonus=0.25,
        floodplain_decay_per_hop=0.5,
        floodplain_max_hops=4,
        volcanic_bonus=0.20,
        volcanic_decay_per_hop=0.3,
        volcanic_max_hops=6,
    )
    defaults.update(overrides)
    return dict(params=FertilityParams(**defaults))  # type: ignore[arg-type]


def _build_inputs(
    n=1,
    h=4,
    w=4,
    k=4,
    t_c=18.0,
    p_total_mm=900.0,
    biome=TEMPERATE,
    ocean=False,
    elev_range_km=0.5,
    water_hops_val=100,
    volc_hops_val=100,
):
    temp = np.full((n, h, w, k), t_c, dtype=np.float32)
    precip = np.full((n, h, w, k), p_total_mm / k, dtype=np.float32)
    ocean_mask = np.full((n, h, w, 1), 1.0 if ocean else 0.0, dtype=np.float32)
    face_biome = np.full(n, biome, dtype=np.int32)
    face_elev_range = np.full(n, elev_range_km, dtype=np.float32)
    water_hops = np.full(n, water_hops_val, dtype=np.int32)
    volc_hops = np.full(n, volc_hops_val, dtype=np.int32)
    return temp, precip, ocean_mask, face_biome, face_elev_range, water_hops, volc_hops


class TestComputeFertility:
    def test_ocean_pixels_zero(self):
        temp, precip, ocean, biome, rng, wh, vh = _build_inputs(ocean=True)
        f = compute_fertility(
            temperature_tex=temp,
            precipitation_tex=precip,
            ocean_mask_tex=ocean,
            face_biome=biome,
            face_relief_km=rng,
            water_hops=wh,
            volcanism_hops=vh,
            **_default_kwargs(),
        )
        assert (f == 0.0).all()
        assert f.shape == (1, 4, 4, 1)

    def test_desert_biome_low(self):
        temp, precip, ocean, biome, rng, wh, vh = _build_inputs(
            biome=DESERT, p_total_mm=150.0
        )
        f = compute_fertility(
            temperature_tex=temp,
            precipitation_tex=precip,
            ocean_mask_tex=ocean,
            face_biome=biome,
            face_relief_km=rng,
            water_hops=wh,
            volcanism_hops=vh,
            **_default_kwargs(),
        )
        # Desert biome multiplier is 0.05; product can't exceed that.
        assert f.max() < 0.10

    def test_optimal_climate_temperate_flat_face(self):
        temp, precip, ocean, biome, rng, wh, vh = _build_inputs(
            biome=TEMPERATE, t_c=18.0, p_total_mm=900.0, elev_range_km=0.0
        )
        f = compute_fertility(
            temperature_tex=temp,
            precipitation_tex=precip,
            ocean_mask_tex=ocean,
            face_biome=biome,
            face_relief_km=rng,
            water_hops=wh,
            volcanism_hops=vh,
            **_default_kwargs(),
        )
        # Climate=1, terrain=1, water=1, geology=1, biome=1.0 → fertility ≈ 1.0
        assert f.max() > 0.95

    def test_high_elevation_range_penalty(self):
        temp, precip, ocean, biome, _, wh, vh = _build_inputs(
            biome=TEMPERATE, t_c=18.0, p_total_mm=900.0
        )
        flat = np.full(1, 0.0, dtype=np.float32)
        rugged = np.full(1, 5.0, dtype=np.float32)  # well above 2.5 threshold
        f_flat = compute_fertility(
            temperature_tex=temp, precipitation_tex=precip, ocean_mask_tex=ocean,
            face_biome=biome, face_relief_km=flat,
            water_hops=wh, volcanism_hops=vh, **_default_kwargs(),
        )
        f_rugged = compute_fertility(
            temperature_tex=temp, precipitation_tex=precip, ocean_mask_tex=ocean,
            face_biome=biome, face_relief_km=rugged,
            water_hops=wh, volcanism_hops=vh, **_default_kwargs(),
        )
        assert f_rugged.max() < 0.05
        assert f_flat.max() > f_rugged.max()

    def test_floodplain_bonus_raises_fertility(self):
        temp, precip, ocean, biome, rng, _, vh = _build_inputs(
            biome=TEMPERATE, t_c=18.0, p_total_mm=900.0, elev_range_km=0.0
        )
        far = np.full(1, 100, dtype=np.int32)
        near = np.full(1, 0, dtype=np.int32)
        # Both are clipped to 1; lower the climate term so the bonus shows.
        kw2 = _default_kwargs(optimal_temperature_c=5.0)  # detune climate so f < 1
        f_far2 = compute_fertility(
            temperature_tex=temp, precipitation_tex=precip, ocean_mask_tex=ocean,
            face_biome=biome, face_relief_km=rng,
            water_hops=far, volcanism_hops=vh, **kw2,
        )
        f_near2 = compute_fertility(
            temperature_tex=temp, precipitation_tex=precip, ocean_mask_tex=ocean,
            face_biome=biome, face_relief_km=rng,
            water_hops=near, volcanism_hops=vh, **kw2,
        )
        assert f_near2.max() > f_far2.max()

    def test_volcanic_bonus_raises_fertility(self):
        # Detune climate so neither result saturates at 1.
        temp, precip, ocean, biome, rng, wh, _ = _build_inputs(
            biome=TEMPERATE, t_c=5.0, p_total_mm=900.0, elev_range_km=0.0
        )
        far = np.full(1, 100, dtype=np.int32)
        near = np.full(1, 0, dtype=np.int32)
        kw = _default_kwargs(optimal_temperature_c=18.0)  # 5° vs 18° → climate < 1
        f_far = compute_fertility(
            temperature_tex=temp, precipitation_tex=precip, ocean_mask_tex=ocean,
            face_biome=biome, face_relief_km=rng,
            water_hops=wh, volcanism_hops=far, **kw,
        )
        f_near = compute_fertility(
            temperature_tex=temp, precipitation_tex=precip, ocean_mask_tex=ocean,
            face_biome=biome, face_relief_km=rng,
            water_hops=wh, volcanism_hops=near, **kw,
        )
        assert f_near.max() > f_far.max()

    def test_biome_multiplier_constants_ordered(self):
        # Pin a few relative orderings rather than exact values.
        m = BIOME_FERTILITY_MULT
        assert m[DESERT] < m[7]               # Desert < Semi-Arid
        assert m[7] < m[4]                    # Semi-Arid < Steppe
        assert m[TEMPERATE] >= m[2]           # Temperate ≥ Taiga
        assert m[TROPICAL] >= m[5]            # Tropical ≥ Savannah
        assert m[0] == 0.0                    # Icecap is zero


# ── River deltas ─────────────────────────────────────────────────────────────


def _delta_case(t_c: float, base: float, plain: float = 1.0):
    fert = np.full((1, 2, 2, 1), base, dtype=np.float32)
    delta = np.zeros((1, 2, 2, 1), dtype=np.float32)
    delta[0, 0, 0] = plain                                # one texel on the delta
    temp = np.full((1, 2, 2, 4), t_c, dtype=np.float32)
    return apply_delta_fertility(fert, delta, temp, FertilityParams())


def test_delta_plain_is_raised_toward_delta_fertility():
    p = FertilityParams()
    out = _delta_case(p.optimal_temperature_c, base=0.1)
    assert np.isclose(out[0, 0, 0, 0], p.delta_fertility, atol=1e-5)
    assert np.allclose(out[0, 1, :, 0], 0.1)              # off the delta: untouched


def test_delta_fertility_never_lowers_and_cold_deltas_stay_poor():
    p = FertilityParams()
    assert np.isclose(_delta_case(p.optimal_temperature_c, base=0.95)[0, 0, 0, 0], 0.95)
    cold = _delta_case(-15.0, base=0.05)[0, 0, 0, 0]
    warm = _delta_case(p.optimal_temperature_c, base=0.05)[0, 0, 0, 0]
    assert 0.05 <= cold < 0.3 < warm


def _wetland_case(t_c: float, base: float):
    fert = np.full((1, 2, 2, 1), base, dtype=np.float32)
    biome = np.full((1, 2, 2, 1), 3, dtype=np.uint8)
    biome[0, 0, 0] = WETLAND_BIOME                        # one wetland texel
    temp = np.full((1, 2, 2, 4), t_c, dtype=np.float32)
    return apply_wetland_fertility(fert, biome, temp, FertilityParams())


def test_humid_wetland_is_docked():
    p = FertilityParams()
    out = _wetland_case(p.optimal_temperature_c, base=0.95)
    assert np.isclose(out[0, 0, 0, 0], 0.95 * (1.0 - p.wetland_penalty))
    assert np.allclose(out[0, 1, :, 0], 0.95)             # off the wetland: untouched


def test_arid_wetland_is_watered_but_cold_stays_poor():
    p = FertilityParams()
    arid = _wetland_case(p.optimal_temperature_c, base=0.02)[0, 0, 0, 0]
    assert np.isclose(arid, p.wetland_fertility * (1.0 - p.wetland_penalty), atol=1e-5)
    assert arid > 0.02 * 10                               # an oasis, not desert
    cold = _wetland_case(-15.0, base=0.02)[0, 0, 0, 0]
    assert cold < 0.5 * arid


def test_land_faces_take_their_water_neighbours_aquatic_fertility() -> None:
    aquatic = np.array([0.0, 0.8, 0.4, 0.0, 0.0], dtype=np.float32)
    is_water = np.array([False, True, True, False, False])
    adjacency = [[1, 2, 3], [0, 2], [0, 1], [0, 4], [3]]
    out = neighbour_water_fertility(aquatic, is_water, adjacency)
    assert out[0] == np.float32(0.6)       # mean of its two water neighbours
    assert out[1] == np.float32(0.8) and out[2] == np.float32(0.4)   # water keeps its own
    assert out[3] == 0.0 and out[4] == 0.0  # no water neighbour
