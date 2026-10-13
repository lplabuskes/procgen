"""A preview of the refined ruggedness, before climate and rivers exist — for the relief band.

The relief band (:mod:`worldgen.geology.relief_band`) needs the ruggedness the
region will draw from, but the ruggedness stage refines the seed only after
climate, hydrology and wetlands, which need the banded elevation.  This estimates
the refinement from the elevation alone, with a cheap stand-in for each input the
ruggedness stage reads, through its own :func:`refine_ruggedness`:

- **temperature** — the climate stage's annual-mean parabola in latitude minus
  its lapse rate over the face's height (no seasons, inertia or currents; ~1 °C
  mean error on land) → the **glacial** term, by far the largest face-scale term;
- river mouths and wetlands are left out: at face scale they move ruggedness by
  ~0.001 (seeds 4/31/35), and the floodplain term is texel-scale along rivers
  the preview can't place.  The final ruggedness has them; conservation absorbs
  the gap (``docs/plans/global_relief_band.md``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from core.mesh import PolygonMesh
    from worldgen.climate.builder import TemperatureParams
    from worldgen.ruggedness.builder import RuggednessRefineParams


def preview_temperature(
    mesh: "PolygonMesh", face_elevation_km: np.ndarray, params: Optional["TemperatureParams"] = None,
) -> np.ndarray:
    """``(N,)`` annual mean °C: latitude parabola minus the lapse rate over the ground (sea level on water)."""
    from worldgen.climate.builder import ClimateParams
    from worldgen.climate.insolation import face_latitudes
    from worldgen.climate.temperature import annual_mean_temperature

    p = params or ClimateParams().temperature
    lat = face_latitudes(mesh.face_centroids)
    t = annual_mean_temperature(lat, float(p.equator_annual_c), float(p.pole_annual_c))
    return t - float(p.lapse_rate_c_per_km) * np.maximum(np.asarray(face_elevation_km, dtype=np.float64), 0.0)


def preview_ruggedness(
    mesh: "PolygonMesh",
    seed_values: np.ndarray,
    face_elevation_km: np.ndarray,
    *,
    temperature: Optional["TemperatureParams"] = None,
    refine: Optional["RuggednessRefineParams"] = None,
) -> np.ndarray:
    """``(N,)`` the seed ruggedness refined by the preview's stand-in inputs, in [0, 1]."""
    from worldgen.ruggedness.builder import RuggednessRefineParams, refine_ruggedness

    rho, _ = refine_ruggedness(
        mesh, seed_values, refine or RuggednessRefineParams(),
        temperature_annual=preview_temperature(mesh, face_elevation_km, temperature),
    )
    return rho
