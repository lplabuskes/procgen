# Region and Local Map Viewer

`viz/mapview/regionview.py` — `RegionView`, the Qt widget showing one region
([region.md](region.md)) or one local map ([local.md](local.md)). World map and
shared overlay renderers: [viz.md](viz.md).

## Wiring (`MapWindow`)

`QStackedWidget` of world map, region `RegionView`, local `RegionView` — going
back from a local map shows the region untouched.

- **Regions** toggles the cell overlay; clicking a cell (`_enter_region`) refines
  and shows it. **Back** (`← World Map` / `← Region`) steps local → region →
  world.
- **Local maps** (in a region) enables hex picking: hovered hex highlighted; a
  click (press/release < 4 px apart, so drags still pan) opens its local map
  (`_enter_local`). The **Villages** toggle is added with the first local map.
- The shared **Coloration** combo picks the layer in every view
  (`_sync_region_layer` hands the world's `SchemeSpec`, recipe, legend to the
  zoomed view). One Overlays toggle drives all three views; −/+/Reset act on the
  visible view.
- `main.py` passes `set_world_params` / `set_realm_params` (world resource,
  land-cost, town, population-γ, danger tuning used when refining) and
  `set_region_params` / `set_local_params` (`AppParams.region` / `.local`,
  view-side only).
- New world (`set_globe`) → back to world map, local cache cleared.

## `RegionView`

**`set_region(geometry, globe)`** refines synchronously (~1.0–1.4 s at defaults,
⅓ resource/danger): `build_region_fields` (world `GlobeParams.resources`,
`PopulationParams.fertility_exponent`), then `_refine_features` (~0.8 s), which
calls the Qt-free `region.builder.refine_region_features` and keeps its
`RegionMap`: `build_region_rivers` → `refine_region_cities` →
`build_region_roads` (world `cost.land` params, realm danger `movement_scale`,
`moved` = cities' site → refined positions) → `snap_deltas` →
`refine_region_towns` (realm town and population params,
`refine.river_threshold_factor`). No globe → bare scaffolding.

**`set_local(region_view, hex)`** → `build_local_map` from the region view's
fields, rivers, roads, deltas, the region's streams (`RegionMap.streams` →
`build_region_streams`, ~0.35 s, built on the region's first local map and kept
until the region, its rivers or `LocalParams.streams` change) and world tuning — or from an LRU cache
(`LOCAL_CACHE_SIZE` = 6, ~0.02 s) valid while the region's fields are the same
object and local params unchanged. Draws `Villages.rural_density` (countryside
net of villages) as `rural_density`.

**Params.** `set_params(RegionParams)` resets all refined towns/cities to stage
sites; new `upsample` re-refines the region (keeps zoom/pan), other changes only
re-run `_refine_features`. `set_local_params(LocalParams)` drops cached local
maps and every local refinement (`reset_local_refinement`). `MapWindow` also
clears the local cache on region-params change and rebuilds an open local map
(`_reopen_local`). See [region.md](region.md#regionparams-regionparamspy).

**Drawing.**
- Each hex's layer texture (`build_layer_texture`) cached as a `QImage`. Local
  `(s, t)` → screen is flip-and-scale, so each hex is an axis-aligned blit
  clipped to its hexagon (0.75 px proud, so antialiased neighbours leave no
  seam).
- Whole-cell rotation from `geometry.up_direction()` (`_up_angle`) → up-is-north
  (up-is-antimeridian on the 2 pole regions). The only place that direction
  becomes a screen rotation (`region/` stays Qt-free).
- Local maps clipped to the parent hexagon (`_outline`).
- Legend drawn left of the rotated footprint (cell sized to fit; map clipped
  short of it). Status bar: `status_text()` — hex/texel counts, detail octaves
  (local: village count and people).

**Zoom and pan.** Wheel 1×–`MAX_ZOOM` (20×) anchored to cursor; drag pans. Zoom
is screen-space magnification after the rotation (hover uses the same inverse).
Pan stored in footprint widths (resize-safe), clamped to the cell; reset on
entering a region/local map. Outlines are cosmetic pens (1 px at any zoom).

**Hover.** Cursor → inverse rotation → texel (`RegionHexGrid.texel_at`); status
bar shows the world hover text for the texel's owning face and UV, except:
- contributors registered `add_hover(..., refined=keys)` are replaced by refined
  values when the map has all keys (`describe_texel`: elevation → `[px: …]`;
  biome/temperature/precipitation → water-or-biome, T, P — `--debug` only, where
  the climate hover is registered);
- `world_only` contributors (per-face city hover) are dropped;
- everything else (fertility, danger, polity, …) is the world value even where
  the drawn layer is refined.

Led by the view's own settlement fragments (`settlements_under_cursor`):
- **City** (Settlements overlay on): marker under cursor (drawn radius +
  `_MARKER_HOVER_SLOP`), or on a local map the city whose footprint holds the
  texel — "Major City · 45,210 people".
- **Towns** — region: the hovered hex's towns by drawn position regardless of
  overlay ("Towns in hex: 3 (9k)", "2 market, 1 river"); local: dot under cursor
  while Towns is on ("Market town · 3,400 people").
- **Village** (local): dot within 6 px — "Farming village · 340 people".

Hit areas use the overlays' own helpers (`settlement_markers`, `town_markers`,
`city_floor_radius`), so they track markers at any zoom. Texels outside a local
map's parent hexagon report nothing; leaving the cell restores the summary.

## Overlays in a zoomed view

Same Overlays toggles and renderers as the world map (`overlays.py`).
`_RegionProjector` maps through `ecef_to_local` and the cell's screen transform,
culling beyond 0.15 of the cell or on the far side; drawn in screen space
(upright markers, pixel widths), clipped to the rotated cell. Sized by the
*equivalent world zoom* (world zoom showing this ground at the same pixel scale).
`regional=True` enables **Towns**.

**Refined features.** Overlays are world-resolution data (e.g. borders follow
global paths), except these, passed as `draw_overlays(..., refined={...})`:

| key | region | local map |
|---|---|---|
| `rivers` | `RegionRivers.paths` | local rivers + streams (`paths` + `streams`) |
| `trade_routes` | `RegionRoads.segments` | local roads |
| `deltas` | snapped deltas | snapped onto local mouths |
| `minerals` | each deposit as its workings (`mineral_sites`; deposit kind, size abundance × share), off refined water | same, off local water |
| `towns` | — (refined positions live in the world feature) | `Towns` at drawn positions |
| `villages` | — | `LocalMap.villages` |
| `cities` | — | `(city_index, positions, CityFootprints)` |
| `polity_borders` | lines contoured from the region's own fill | same, local |

**Polities and borders.** The view's drawn rivers (`paths`, packed by
`pack_river_segments`) go to every layer build (`build_layer_texture(...,
river_segments=)`), so the Polities fill's river borders follow them, not the world
centreline up to a corridor away. With **Polity Borders** on, `view_polity_borders`
(`region/borders.py`) contours that same field — `polity_label_field.with_rivers` —
on a lattice over the view's `(s, t)` frame (1.5 texels; water and off-cell from the
refined masks; the world's tracer), so lines bound the fill drawn. Segments and
lines are cached in `_polity_cache` against the rivers and fields objects (lines
traced on first draw, ~0.4 s a region); new river params re-route and rebuild the
images.

Widths and thresholds come from the *world* feature (route width vs world max
flow, world median cut), so a re-routed road is drawn as heavily as on the world
map. **Exception — rivers** scale by the view's own `view_zoom` (a region at 1×
draws 1.5× world line weight) and on local maps at physical width (`px_per_km`;
[viz.md](viz.md)).

**Settlement markers.**
- Region: cities 1.5× world size (≥ 2× largest town dot), opaque, ringed
  dark-then-white.
- Local: towns get that ring; cities drawn as footprints, or a walled floor
  marker (2× largest town radius; dark, white, dark rings, dark centre dot) while
  the footprint is smaller on screen.
- Villages: 1.3–2.4 px dots by kind, under towns, **Villages** toggle.
- Farmland: baked per hex into translucent images (`_rebuild_farm_images`, from
  `localmap.farmland.farmland_rgba`), over terrain and under everything; each
  kind follows its settlements' toggle (village fields → **Villages**, town fields
  → **Towns**, market gardens → **Settlements**).

**Settlements onto land.** A town/city this view doesn't own (sited in an
unvisited neighbouring region/hex) stands where the tier above put it, possibly
on refined water. The projector's `on_land` runs `snap_to_land`: markers on water
move to the nearest in-hex land texel within `water_spread_texels + 2` texels
(the farthest water can advance), typically 3–6 km in a region. Cached per
feature; display-only, but re-routed roads end on the same snapped positions
(`build_region_roads`' `land_reach` is the view's).
