# Global relief band

Status: done (2026-10-12).
Branch `feature/ruggedness`.

As built, departures from the design below: the preview is temperature → glacial
only (river mouths and wetlands move face ruggedness ~0.001 with their real inputs,
so no river or wetland proxies); the spectrum functions stay in `region/upsample.py`,
imported lazily; wetland and floodplain flatness read the slope of the trend surface
(`trend_elevation`, the elevation without the band) — on the full elevation seed 4's
wetlands fell 4.7% → 0.5%; sub-sea dry land left as is by user choice.
Phase 3 found the band's dips filling as ~4× as many ragged globe lakes; by user
choice lakes are found on the trend surface and the band is zeroed under their beds
(docs/hydrology.md). Retuned: region `lake_breach_relief` 0.3 → 1.25 (local keeps
0.3), region carve `absorb_relief` 1 → 2, wetland `slope_scale_m_per_km` 0.4 → 0.45,
`sea_range_factor` 1.2 → 1.1.

## Problem

The region tier's relief looks noisy, and rugged lowlands dig dry land far below
sea level. Measured on seeds 4/31/35 (the region with the most land, the roughest and the smoothest per seed):

| | before conservation | after |
|---|---|---|
| detail variance in per-global-texel means (what conservation removes) | 0.30–0.40 | — |
| correction vs detail correlation | — | −0.6 to −0.7 |
| neighbour-texel correlation | 0.65–0.72 | 0.51–0.59 |
| texel-scale variance share (after a 3×3 box) | 0.20–0.27 | 0.33–0.40 |

- The region's detail band is the globe's Nyquist (28 km) down to its own (~9 km):
  2–6 region texels. Conservation removes the part each global texel can see (most
  of the 28 km octave), so what remains is pushed toward texel scale. Conservation with
  no detail is smooth (texel-scale share 0.06–0.08): the problem is the interaction.
- Roughest regions: dry land below 0 is 1.3–1.6% (globe 0.06–0.10%), lowest dry texel
  −0.95 to −1.23 km. Those texels' parents sit at a median of 150–275 m with ρ 0.65–0.77, and detail of
  350–500 m rms drives them under, 10–35 km inland, already below 0 right after conservation.

## Idea

Relief the globe can resolve belongs on the globe. Each global texel gets the
**box average of the region's detail** over its footprint (the *relief band*). The
region then starts from the globe *minus* the band and adds its full detail: the
per-global-texel means already match, so the conservation correction drops from
tens to hundreds of metres to a few, and the region keeps its full spectrum.

The band's amplitude needs ruggedness, and the best ruggedness needs climate and rivers,
which need the final elevation. So a **fast, low-fidelity pre-pass** estimates them
from the elevation before the band is added, refines a *preview* ruggedness, and sizes
the band from that. The normal pipeline then runs on the banded elevation, and the
ruggedness stage re-refines from the real climate and rivers as it does now.

```
elevation stage:  tectonics → textures → globe noise → seams → ocean mask (provisional)
                  → pre-pass proxies → preview ρ → relief band → + band → ocean mask
then as today:    climate → hydrology → delta → lake feedback → wetland → ruggedness (final ρ) → …
```

## Design

### Pre-pass proxies (new, `worldgen/geology/preview.py`)

Per face unless noted, all on the provisional ocean mask. Feed `refine_ruggedness`
(its existing function, so the terms stay one definition):

| input | proxy | replaces |
|---|---|---|
| `temperature_annual` | `annual_mean_temperature(lat)` − lapse × max(elev, 0) (climate's params) | climate stage |
| moisture (P/PET) | zonal curve (wet ITCZ, dry ~30°, wet ~55°) × decay with distance to coast | precipitation |
| river mouths, face flow | priority flood + steepest-descent accumulation on the **face graph** (10k nodes), rain-weighted by the moisture proxy | hydrology |
| `wetland_fraction` | flatness × wetness × low height above the face-graph drainage, scaled to the stage's land share (~5%) | wetland stage |

- Floodplain term: **skipped** in the pre-pass (texel-scale; face-graph rivers are
  ~200 km off in position). The final ρ still has it; conservation absorbs the gap along big rivers.
- Jaggedness: the **seed** is used for the band (the 16 km Hurst pivot sits inside the
  region band, so character barely moves a region's relief).
- The preview ρ is used only for the band, not written: the seed stays the seed, and the
  ruggedness stage refines it as now.

### Relief band (`worldgen/geology/relief_band.py`)

- Octaves, amplitudes and noise stream exactly the region tier's: `region.upsample`'s
  `_octave_frequencies` / `spectrum_amplitudes` / `_octave_noise` with
  `RegionUpsampleParams` defaults, seed `world.seed ^ _SEED_SALT_REGION ^ _SEED_SALT_ELEVATION`.
  These move to a shared module (worldgen must not import `region/`).
- Per global texel: mean over S×S sub-points (S = 3, the region's ~3× refinement), ρ
  bilinear from the preview texture at each sub-point. ~23M points × 2 octaves on the
  numba `fbm_sum`: ~1–2 s.
- Everywhere, land and sea (as the region). The region damps detail under lakes, which
  don't exist yet; lake anchors already take a constant correction, so the gap stays hidden under water.
- Stored as texture `relief_band` (persisted with the mesh).

### Region tier

- `elev_coarse` = bilinear of `elevation − relief_band` (absent band → 0, so old worlds still open).
- Targets unchanged (the globe's `elevation`): conservation stays exact.
- Detail unchanged, drawn from the final ρ. Correction ≈ band − box-mean(detail):
  small where preview and final ρ agree.
- Local and settlement tiers unchanged (the region → local step has the same issue; later, same recipe).

## Costs

- Globe gets rougher: band rms ≈ 0.55–0.63 × region detail rms (rugged ~200–300 m per
  texel, plains ~20–30 m). Visible in globe hillshade.
- Climate, hydrology (more pits → lakes), land fraction, coastlines, costs,
  population and polities all move. Rebuild test/tuning worlds; re-run river, lake,
  population, polity, border measurements; retune as needed.
- Build time: +~2–3 s (proxies ≲ 0.5 s, band ~1–2 s, second ocean mask ~0.3 s).

## Phases

0. **Harness.** `scripts/measure_relief_noise.py` (from the scratch measurement): per
   region, detail rms, conservation-removed share, correction rms and correlation,
   lag-1 and texel-scale share before/after, dry max/min, share of dry land below 0;
   globe equivalents. Record the baseline.
1. **Band from the seed ρ.** Relief band + region `elev_coarse` change, ρ = seed (no
   proxies). Proves the noise win end to end and shows the globe-side fallout before
   the proxies exist. Measure: noise harness, `measure_relief`, `measure_lakes_rivers`.
2. **Proxies → preview ρ.** Add the pre-pass; compare preview ρ to the final ρ per face
   (correlation, mean abs error, by term) and the conservation correction rms vs phase 1.
3. **Rebuild + retune.** Test/tuning worlds; rivers, population (absolute), polities,
   borders, wetlands, settlements; retune what drifts.
4. **Docs.** `elevation.md`, `ruggedness.md`, `region_fields.md`, CLAUDE.md pipeline line.

## Open questions

- Pre-pass floodplain: skip (above), or a face-graph approximation?
- Should the band mirror the region's coast style near shores? Proposed: no
  (conservation absorbs it; the coast style is a shoreline effect, not relief).
- Smooth near-sea-level plains (seed 35's smoothest region: 3.45% of dry land below 0 from
  tens of metres of detail on a ~10 m parent) aren't addressed by this; separate fix.
