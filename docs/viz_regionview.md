# Region and Local Map Viewer

`viz/mapview/regionview.py` — `RegionView`, the Qt widget that shows one region
([region.md](region.md)) or one local map ([local.md](local.md)). The world map
and the shared overlay renderers are in [viz.md](viz.md).

## Wiring (`MapWindow`)

`MapWindow` holds a `QStackedWidget` of the world map, a `RegionView` for the
region and a second `RegionView` for local maps, so going back from a local map
shows the region untouched.

- **Regions** toggles the region-cell overlay on the world map; clicking a cell
  (`_enter_region`) refines it and shows it. **Back** (`← World Map` /
  `← Region`) steps local → region → world.
- **Local maps** (shown in a region) turns on hex picking: the hex under the
  cursor is highlighted, and a click (a press and release under 4 px apart, so
  drags still pan) opens its local map (`_enter_local`). The **Villages**
  overlay toggle is added with the first local map.
- The shared **Coloration** combo picks the layer in every view, so it carries
  across switching (`_sync_region_layer` hands the world map's `SchemeSpec`,
  recipe and legend to the zoomed view). One Overlays-menu toggle drives all
  three views. The toolbar's −/+/Reset act on whichever view is showing.
- `main.py` passes `set_world_params` / `set_realm_params` (the world's own
  resource, land-cost, town, population-γ and danger tuning, used when refining)
  and `set_region_params` / `set_local_params` (`AppParams.region` /
  `AppParams.local`, view-side only).
- A new world (`set_globe`) drops back to the world map and clears the local
  cache.

## `RegionView`

**`set_region(geometry, globe)`** refines synchronously (~1.0–1.4 s for a
region at defaults, a third of it the resource/danger refinement):
`build_region_fields` with the world's `GlobeParams.resources` and
`PopulationParams.fertility_exponent`, then `_refine_features` — 
`build_region_rivers`, `refine_region_cities`, `build_region_roads` (with the
world's `cost.land` params, the realm's danger `movement_scale`, and `moved` =
the cities' site → refined positions), `snap_deltas`, then `refine_region_towns`
(with the realm's town and population params and `refine.river_threshold_factor`),
~0.8 s on top of the fields. Without a globe it falls back to the bare
scaffolding.

**`set_local(region_view, hex)`** builds the local map with `build_local_map`
from the region view's fields, rivers, roads and deltas and the world's tuning,
or reuses it from an LRU cache (`LOCAL_CACHE_SIZE` = 6, ~0.02 s), valid while
the region's fields are the same object and the local params are unchanged. The
`rural_density` it draws is `Villages.rural_density` (the countryside net of
villages).

**Params.** `set_params(RegionParams)` resets every refined town and city to its
stage site; a new `upsample` re-refines the region (keeping zoom and pan), any
other change only re-runs `_refine_features`. `set_local_params(LocalParams)`
drops the cached local maps and every settlement's local refinement
(`reset_local_refinement`). `MapWindow` clears the local cache on a region-params
change too, and rebuilds a local map on screen (`_reopen_local`). See
[region.md](region.md#regionparams-regionparamspy).

**Drawing.** Each hex's layer texture (`region.display.build_layer_texture`) is
cached as a `QImage`. Local `(s, t)` → screen is a flip-and-scale, so each hex
texture is an axis-aligned blit clipped to its hexagon (drawn 0.75 px proud so
antialiased neighbours leave no seam), wrapped in a whole-cell rotation from
`geometry.up_direction()` (`_up_angle`) so the rendered map is actually
up-is-north (up-is-antimeridian on the 2 pole regions); this is the one place
that direction gets turned into a screen rotation, since `region/` itself stays
Qt-free. A local map is clipped to its parent hexagon (`_outline`).

The active layer's legend is drawn outside the cell, to the left of its
(rotated) footprint; the cell is sized so the rotated footprint fits beside it,
and the map is clipped short of the legend. The status bar shows
`status_text()`: hex/texel counts and the detail-octave count (a local map: its
village count and people).

**Zoom and pan.** Scroll zooms (1×–`RegionView.MAX_ZOOM` = 20×, anchored to the
cursor) and drag pans, like the world map. The zoom is a screen-space
magnification of the laid-out cell, composed after the north-up rotation, so
hover goes through the same inverse transform. Pan is stored in footprint widths
(resize-safe) and clamped so the view never leaves the cell. Entering a region or
a local map resets the view. Outlines are cosmetic pens, so they stay 1 px at any
zoom.

**Hover.** The cursor maps back through the rotation to a texel
(`RegionHexGrid.texel_at`), and the status bar shows the world map's full hover
text for the texel's owning face and UV, except that contributors registered
with `add_hover(..., refined=keys)` are replaced in place by the refined values
when the map has all of those keys (`describe_texel`: elevation → `[px: …]`,
biome/temperature/precipitation → water-or-biome, T, P — the latter only under
`--debug`, where the climate hover is registered). Contributors registered
`world_only` (the per-face city hover) are dropped. Everything else (fertility,
danger, polity, …) is the world value, even where the layer drawn is refined.

The status bar leads with the view's own settlement fragments
(`settlements_under_cursor`):

- **City** (region and local, while the Settlements overlay is on): the city
  whose marker is under the cursor (its drawn radius plus `_MARKER_HOVER_SLOP`),
  or on a local map the city whose built-up ground holds the texel —
  "Major City · 45,210 people".
- **Towns** — in a region, the hovered hexagon's towns, counted by their drawn
  position whatever the overlay ("Towns in hex: 3 (9k)", "2 market, 1 river");
  on a local map, the town dot under the cursor while the Towns overlay is on
  ("Market town · 3,400 people").
- **Village** (local only): the dot within 6 px — "Farming village · 340 people".

Marker sizes come from the same helpers the overlays draw with
(`settlement_markers`, `town_markers`, `city_floor_radius`), so hit areas track
the drawn markers at every zoom. Texels outside a local map's parent hexagon
report nothing. Leaving the cell restores the summary.

## Overlays in a zoomed view

The world map's feature overlays (rivers, borders, settlements, trade routes,
resources, winds, …) draw here too, from the same Overlays menu toggles and the
same renderers (`viz/mapview/overlays.py`). `_RegionProjector` maps sphere points
through `ecef_to_local` and the cell's screen transform, culling anything beyond
0.15 of the cell or on the far side; overlays are drawn in screen space (upright
markers, pixel widths) and clipped to the rotated cell. They size themselves by
the *equivalent world zoom* — the world-map zoom that would show this ground at
the same pixel scale — so a region looks like the world map zoomed to the same
place. `regional` is True on the projector, which is what lets the **Towns**
overlay draw.

**Refined features.** Overlays are world-resolution data — borders follow the
global paths rather than the refined terrain — except these, handed to
`draw_overlays(..., refined={...})` in place of the world features:

| key | region | local map |
|---|---|---|
| `rivers` | `RegionRivers.paths` | the local rivers |
| `trade_routes` | `RegionRoads.segments` | the local roads |
| `deltas` | snapped deltas | snapped onto the local mouths |
| `towns` | — (refined positions are in the world feature) | a `Towns` at the positions the map draws them |
| `villages` | — | `LocalMap.villages` |
| `cities` | — | `(city_index, positions, CityFootprints)` |

Widths and thresholds still come from the *world* feature — river width against
the world's largest flow, route width against the world's largest route flow
and the world's median as the drawing cut — so a re-routed road is drawn exactly
as heavily as on the world map. **River widths are the exception**: they scale
by the view's *own* zoom (`view_zoom`), so a region at 1× draws 1.5× the line
weights the world map does at 1×, and a local map draws each river at its
physical channel width (`px_per_km`; see [viz.md](viz.md)).

**Settlement markers.** In a region, cities are drawn to stand apart from the
towns: 1.5× their world-map size (and at least twice the largest town dot),
opaque, and ringed dark-then-white. On a local map, towns take that dark-then-white
ring; cities are drawn as their footprints, or as a floor marker twice the
largest town's radius while the footprint is smaller than that on screen, walled
(dark, white, dark rings, a dark centre dot) so it stands apart from the towns.
Villages are 1.3–2.4 px dots coloured by kind (farming, river, fishing,
roadside), under the towns, behind the **Villages** toggle. Farmland is baked per
hexagon into translucent images (`_rebuild_farm_images`, from
`localmap.farmland.farmland_rgba`) and drawn over the terrain and under all of
these, each kind with its settlements' overlay: village fields with
**Villages**, town fields with **Towns**, market gardens with **Settlements**.

**Settlements onto land.** A town or city the view refines lands on its dry
texels, but one it does not own (sited in a neighbouring region or hex, not yet
visited) still stands where the tier above put it, which the refined shoreline
can flood. The projector's `on_land` runs `region.display.snap_to_land`: a marker
on refined water moves to the nearest in-hexagon land texel within
`water_spread_texels + 2` texels of this tier (the farthest the refinement lets
water advance), typically 3–6 km in a region. `RegionView` caches the result per
feature. It is display-only; the re-routed roads end on the same snapped positions (`build_region_roads`' `land_reach` is the view's).
