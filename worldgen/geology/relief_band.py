"""The relief band — the part of the region tier's relief the globe can resolve.

The region draws a ruggedness-set relief spectrum from the globe's Nyquist down to
its own (``region.upsample.spectrum_amplitudes``) and conserves each global texel's
mean.  Whatever of that detail a global texel can see — its mean over the texel's
footprint — conservation would take straight back out, pushing what is left toward
texel scale.  So the globe draws it instead: each global texel gets the region's
detail averaged over its footprint (``S × S`` sub-points), with the region's noise
stream, octaves and amplitudes; the region starts from the globe minus this band
and adds its full detail, and the two already agree before conservation runs.

See ``docs/plans/global_relief_band.md``.  The spectrum is the region tier's at its
defaults (``RegionUpsampleParams()``): the band is part of the world, the region's
params are view-side, and conservation absorbs any difference if they are retuned.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.params import param_field as _f, paramclass

if TYPE_CHECKING:
    from core.mesh import PolygonMesh
    from region.upsample import RegionUpsampleParams

# Faces per chunk of sub-point evaluation (~1M points at tex_size 16, S = 3).
_CHUNK_FACES = 512


@paramclass
class ReliefBandParams:
    # Off: the globe carries no band and the region conserves its whole detail (the old behaviour).
    enabled: bool = _f(True, label="Relief band on globe", min=0, max=1)
    # Sub-points per global texel side for the footprint mean (odd; the region is ~3× finer).
    subsamples: int = _f(3, label="Footprint sub-points per side (odd)", min=1, max=9, step=2)
    # Size the band from a preview of the refined ruggedness (worldgen/geology/preview.py);
    # off: from the tectonic seed alone.
    preview: bool = _f(True, label="Preview refined ruggedness", min=0, max=1)


def trend_elevation(mesh: "PolygonMesh") -> np.ndarray:
    """``(N, H, W, 1)`` float64 the globe's elevation without its relief band (km).

    The band is sub-grid roughness the world's ruggedness already stands for, so
    terms that read the globe's slope as a regional gradient (wetland and floodplain
    flatness) read it here; water still runs over the full elevation.
    """
    from worldgen.geology.channels import ELEVATION, RELIEF_BAND
    elev = np.asarray(ELEVATION.read(mesh), dtype=np.float64)
    if RELIEF_BAND.present(mesh):
        elev = elev - np.asarray(RELIEF_BAND.read(mesh), dtype=np.float64)
    return elev


def globe_spacing_rad(mesh: "PolygonMesh", tex_size: int) -> float:
    """Mean global texel spacing (radians), as ``region.source.GlobalSource`` measures it."""
    return float(np.sqrt(np.mean(mesh.face_areas))) / max(int(tex_size) - 1, 1)


def region_spacing_rad(params: "RegionUpsampleParams") -> float:
    """Mean texel spacing (radians) of a default region grid."""
    from region.geometry import RegionGeometry
    from region.grid import RegionHexGrid
    grid = RegionHexGrid(RegionGeometry(0, 0.0), int(params.n_cols))
    return grid.texel_spacing_rad(int(params.tex_size))


def _upsample_axis(tex: np.ndarray, s: int, axis: int) -> np.ndarray:
    """Linear interpolation of ``tex`` along ``axis`` onto ``(n − 1) · s + 1`` aligned points."""
    n = tex.shape[axis]
    x = np.arange((n - 1) * s + 1, dtype=np.float64) / s
    i0 = np.clip(np.floor(x).astype(np.int64), 0, max(n - 2, 0))
    i1 = np.minimum(i0 + 1, n - 1)
    frac = x - i0
    shape = [1] * tex.ndim
    shape[axis] = -1
    frac = frac.reshape(shape)
    return np.take(tex, i0, axis=axis) * (1.0 - frac) + np.take(tex, i1, axis=axis) * frac


def _footprint_mean(fine: np.ndarray, s: int, tex_size: int) -> np.ndarray:
    """``(C, T, T)`` mean of ``(C, F, F)`` aligned sub-points over each texel's ``s × s`` footprint.

    Texel ``i`` sits on sub-point ``s · i``; edge texels average the sub-points that exist.
    """
    half = (s - 1) // 2
    centre = np.arange(tex_size) * s
    total = np.zeros((fine.shape[0], tex_size, tex_size))
    count = np.zeros((tex_size, tex_size))
    f = fine.shape[1]
    for dy in range(-half, half + 1):
        rows = centre + dy
        ok_r = (rows >= 0) & (rows < f)
        for dx in range(-half, half + 1):
            cols = centre + dx
            ok_c = (cols >= 0) & (cols < f)
            ok = ok_r[:, None] & ok_c[None, :]
            r, c = np.clip(rows, 0, f - 1), np.clip(cols, 0, f - 1)
            total += np.where(ok, fine[:, r[:, None], c[None, :]], 0.0)
            count += ok
    return total / count


def compute_relief_band(
    mesh: "PolygonMesh",
    ruggedness: np.ndarray,
    jaggedness: Optional[np.ndarray],
    world_seed: int,
    params: ReliefBandParams,
    spectrum: Optional["RegionUpsampleParams"] = None,
) -> np.ndarray:
    """``(N, T, T, 1)`` float32: the region's detail averaged over each global texel (km).

    ``ruggedness`` / ``jaggedness`` are ``(N, T, T, 1)`` textures (jaggedness None →
    neutral), read bilinearly at the sub-points as the region reads them.
    """
    from region.upsample import (
        _SEED_SALT_ELEVATION, _SEED_SALT_REGION, EARTH_RADIUS_KM, RegionUpsampleParams,
        _octave_frequencies, _octave_noise, hurst_from_jaggedness, spectrum_amplitudes,
    )
    from worldgen.geology.jaggedness import NEUTRAL_JAGGEDNESS

    spectrum = spectrum or RegionUpsampleParams()
    n, t = ruggedness.shape[0], ruggedness.shape[1]
    out = np.zeros((n, t, t, 1), dtype=np.float32)
    freqs = _octave_frequencies(
        0.5 / max(globe_spacing_rad(mesh, t), 1e-12),
        float(spectrum.nyquist_safety) / max(region_spacing_rad(spectrum), 1e-12), spectrum)
    if freqs.size == 0 or not params.enabled:
        return out
    s = max(int(params.subsamples), 1) | 1                      # odd: texel centres are sub-points
    f = (t - 1) * s + 1
    seed = (int(world_seed) ^ _SEED_SALT_REGION) ^ _SEED_SALT_ELEVATION
    wavelengths = EARTH_RADIUS_KM / freqs
    rug = np.clip(np.asarray(ruggedness[..., 0], dtype=np.float64), 0.0, 1.0)
    jag = (np.full_like(rug, NEUTRAL_JAGGEDNESS) if jaggedness is None
           else np.clip(np.asarray(jaggedness[..., 0], dtype=np.float64), 0.0, 1.0))
    # Sub-points interpolated between the cached texel positions (a texel is ~14 km: the
    # chord's bow is metres) — far cheaper than placing each on its face's fan triangles.
    texel_pos = mesh.face_texture_positions(t, t)
    for start in range(0, n, _CHUNK_FACES):
        faces = np.arange(start, min(start + _CHUNK_FACES, n))
        pos = _upsample_axis(_upsample_axis(np.asarray(texel_pos[faces], dtype=np.float64), s, 1), s, 2)
        pos = pos.reshape(-1, 3)
        pos /= np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-12)
        r = _upsample_axis(_upsample_axis(rug[faces], s, 1), s, 2).reshape(-1)
        j = _upsample_axis(_upsample_axis(jag[faces], s, 1), s, 2).reshape(-1)
        amps = spectrum_amplitudes(wavelengths, r, hurst_from_jaggedness(j, spectrum), spectrum)
        detail = (amps * _octave_noise(pos, seed, freqs)).sum(axis=0)
        out[faces, :, :, 0] = _footprint_mean(detail.reshape(len(faces), f, f), s, t)
    return out
