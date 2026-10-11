from __future__ import annotations

import dataclasses
import math
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QWidget

from core.display import SchemeSpec
from localmap.builder import LocalMap, build_local_map
from localmap.geometry import PARENT_CENTER, PARENT_RADIUS
from localmap.params import LocalParams
from localmap.farmland import farmland_rgba
from localmap.villages import VILLAGE_KIND_NAMES
from region.borders import view_polity_borders
from region.display import build_layer_texture, snap_to_land
from realmgen.refine.barriers import pack_river_segments
from viz.mapview.legend import draw_legend, legend_size
from viz.mapview.overlays import city_floor_radius, draw_overlays, fort_markers, settlement_markers, town_markers
from region.builder import RegionMap, land_reach, refine_region_features
from region.geometry import RegionGeometry
from region.grid import RegionHexGrid
from region.params import RegionParams
from region.rivers import RegionRivers, RiverPath
from region.roads import RegionRoads
from region.streams import RegionStreamParams
from realmgen.placement.types import TIER_NAMES, format_population, population_display_tier
from realmgen.towns.types import TOWN_KIND_NAMES, Towns
from region.upsample import RegionFields, build_region_fields
from worldgen.hydrology.discharge import EARTH_RADIUS_KM
from worldgen.resources.minerals import mineral_workings

if TYPE_CHECKING:
    from core.world import World
    from realmgen.builder import RealmParams
    from worldgen.builder import GlobeParams

# The overlays a local map's farmland follows: village fields, town fields, market gardens.
_FARM_OVERLAYS = ("villages", "towns", "settlements")


class RegionView(QWidget):
    """Zoomed-in view of a single region, or of one regional hex's local map.

    The region cell is a unit square in the equal-area ``(s, t)`` frame
    (:mod:`region.geometry`), tiled with flat-top hexagons, each carrying its
    refined texture (:mod:`region.upsample`).  ``(s, t)`` maps to screen by a
    flip-and-scale, so each texture is an axis-aligned image clipped to its
    hexagon.  Without a globe the bare hex scaffolding is drawn.

    A local map (:meth:`set_local`) uses the parent hex's square as its frame,
    clipped to the parent hexagon.  In a region, :meth:`set_local_pick` turns
    clicks into "open this hex's local map".

    Zoom (1×–``MAX_ZOOM``) is a screen-space magnification applied after the
    north-up rotation, so the legend stays put.
    """

    MAX_ZOOM = 20.0
    # Local maps kept for revisiting, most recent last.
    LOCAL_CACHE_SIZE = 6
    # A press-release that moved less than this (px) is a click, not a drag.
    _CLICK_SLOP = 4.0
    # How close (px) the cursor must be to a village dot to describe it.
    _VILLAGE_HOVER_PX = 6.0
    # How far (px) past a city or town marker's fill still counts as hovering it.
    _MARKER_HOVER_SLOP = 3.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._geometry: Optional[RegionGeometry] = None
        self._grid: Optional[RegionHexGrid] = None
        self._fields: Optional[RegionFields] = None
        self._region_params = RegionParams()
        self._realm_params: Optional["RealmParams"] = None
        self._world_params: Optional["GlobeParams"] = None
        # The region's rivers, roads, deltas and streams (region mode only).
        self._region_map: Optional[RegionMap] = None
        # What is drawn: the region map's, or in local mode the local map's.
        self._rivers: Optional[RegionRivers] = None
        self._roads: Optional[RegionRoads] = None
        self._deltas: Optional[np.ndarray] = None
        # (rivers, fields, packed segments, borders) — the drawn rivers' segments, which the
        # Polities fill and border lines snap to, and those lines (None until first drawn).
        self._polity_cache: Optional[tuple] = None
        self._globe: Optional["World"] = None
        self._layer: Optional[str] = None
        self._spec: Optional[SchemeSpec] = None
        self._legend: Optional[tuple[str, list]] = None
        self._overlay_visible: dict[str, bool] = {}
        self._hex_images: list[QImage] = []
        # A local map's farmland, one image per hexagon (empty: none shown).
        self._farm_images: list[QImage] = []
        # Per feature: (source positions, positions moved onto the refined land).
        self._land_positions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # (x0, y0, size, rotation) of the cell as last painted — hover hit-tests against it.
        self._cell_layout: Optional[tuple[float, float, float, QTransform]] = None
        self._hover_callback: Optional[Callable[[Optional[tuple[int, int, int]]], None]] = None
        # Pan is in units of the footprint width, so it survives a resize.
        self._zoom: float = 1.0
        self._pan = np.zeros(2, dtype=np.float64)
        # (centre x, centre y, footprint px) of the last paint — zoom/pan anchor.
        self._view_frame: Optional[tuple[float, float, float]] = None
        self._drag_start: Optional[tuple[float, float]] = None
        self._drag_start_pan = np.zeros(2, dtype=np.float64)
        self._press_pos: Optional[tuple[float, float]] = None
        # ── local maps ──
        self._local_params = LocalParams()
        self._local: Optional[LocalMap] = None
        # (region_id, hex) → LocalMap, valid for ``_cache_owner`` (the region
        # fields they were built from) and the current local params.
        self._local_cache: "OrderedDict[tuple[int, int], LocalMap]" = OrderedDict()
        self._cache_owner: Optional[RegionFields] = None
        # Region mode: clicking a hex opens its local map.
        self._local_pick = False
        self._hover_hex: Optional[int] = None
        self._hex_click_callback: Optional[Callable[[int], None]] = None
        self._mouse: Optional[tuple[float, float]] = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ── data ─────────────────────────────────────────────────────────────────

    def set_region(self, geometry: RegionGeometry, globe: Optional["World"] = None) -> None:
        """Show a region, refining the global fields inside it when a globe is given."""
        self._geometry = geometry
        self._globe = globe
        self._grid = RegionHexGrid(geometry, int(self._region_params.upsample.n_cols))
        self._local = None
        self._hover_hex = None
        self._fields = None
        self._region_map = None
        self._rivers = None
        self._roads = None
        self._deltas = None
        self._hex_images = []
        self._farm_images = []
        self._land_positions = {}
        self._zoom = 1.0
        self._pan[:] = 0.0
        if globe is not None and "elevation" in globe.mesh.face_textures:
            realm, world = self._realm_params, self._world_params
            self._fields = build_region_fields(
                globe, geometry.region_id, self._region_params.upsample,
                resources=world.resources if world is not None else None,
                fertility_exponent=float(realm.population.fertility_exponent) if realm is not None else 2.0,
            )
            self._grid = self._fields.grid
            self._refine_features()   # first: riparian wetland edits the biome the images draw
            self._rebuild_images()
        self.update()

    def _refine_features(self) -> None:
        """Re-route rivers and roads over the refined terrain, snap deltas, and refine
        this region's cities and towns.  Settlements are refined in the world's
        features, so they persist (and are saved) after the region is left."""
        assert self._fields is not None and self._globe is not None
        realm, world = self._realm_params, self._world_params
        m = self._region_map = refine_region_features(
            self._globe, self._fields, self._region_params,
            land=world.cost.land if world is not None else None,
            danger_scale=float(realm.danger.movement_scale) if realm is not None else 2.0,
            town_params=realm.towns if realm is not None else None,
            population_params=realm.population if realm is not None else None,
            river_threshold_factor=realm.refine.river_threshold_factor if realm is not None else 1.5,
        )
        self._rivers, self._roads, self._deltas = m.rivers, m.roads, m.deltas

    # ── local maps ───────────────────────────────────────────────────────────

    def _region_streams(self, params: RegionStreamParams) -> list[RiverPath]:
        """This region's streams for its local maps, cached until the region or ``params`` change."""
        if self._region_map is None or self._globe is None:
            return []
        return self._region_map.streams(self._globe, params)

    def set_local(self, parent: "RegionView", hex_index: int, keep_view: bool = False) -> None:
        """Show the local map of ``parent``'s regional hex ``hex_index`` (cached per region fields).

        ``keep_view`` keeps the zoom and pan (a rebuild of the map on screen).
        """
        fields, globe = parent._fields, parent._globe
        if fields is None or globe is None:
            return
        if self._cache_owner is not fields:
            self._local_cache.clear()
            self._cache_owner = fields
        key = (fields.region_id, int(hex_index))
        local = self._local_cache.pop(key, None)
        if local is None:
            realm, world = self._realm_params, self._world_params
            local = build_local_map(
                globe, fields, int(hex_index), self._local_params,
                region_rivers=parent._rivers, region_roads=parent._roads, region_deltas=parent._deltas,
                region_streams=parent._region_streams(self._local_params.streams),
                resources=world.resources if world is not None else None,
                fertility_exponent=float(realm.population.fertility_exponent) if realm is not None else 2.0,
                land=world.cost.land if world is not None else None,
                danger_scale=float(realm.danger.movement_scale) if realm is not None else 2.0,
                town_params=realm.towns if realm is not None else None,
                population_params=realm.population if realm is not None else None,
                river_threshold_factor=realm.refine.river_threshold_factor if realm is not None else 1.5,
            )
        self._local_cache[key] = local
        while len(self._local_cache) > self.LOCAL_CACHE_SIZE:
            self._local_cache.popitem(last=False)

        self._local = local
        self._globe = globe
        self._geometry = local.fields.geometry
        self._grid = local.fields.grid
        # The countryside the map draws is net of its villages.
        self._fields = dataclasses.replace(
            local.fields, textures={**local.fields.textures, "rural_density": local.villages.rural_density},
        )
        self._rivers, self._roads, self._deltas = local.rivers, local.roads, local.deltas
        self._hex_images = []
        self._farm_images = []
        self._land_positions = {}
        if not keep_view:
            self._zoom = 1.0
            self._pan[:] = 0.0
        self._rebuild_images()
        self._rebuild_farm_images()
        self.update()

    def set_local_params(self, params: LocalParams, globe: Optional["World"] = None) -> None:
        """Replace the local tuning, dropping cached local maps (the caller re-opens the one
        on screen) and resetting every settlement's local refinement in ``globe``."""
        if params == self._local_params:
            return
        self._local_params = params
        self._local_cache.clear()
        globe = globe if globe is not None else self._globe
        if globe is not None:
            towns = globe.get_feature("towns")
            if towns is not None:
                towns.reset_local_refinement()
            for city in globe.get_feature("settlements") or []:
                city.reset_local_refinement()
            self._land_positions = {}

    def clear_local_cache(self) -> None:
        self._local_cache.clear()
        self._cache_owner = None

    @property
    def local_map(self) -> Optional[LocalMap]:
        return self._local

    def set_local_pick(self, enabled: bool) -> None:
        """Region mode: highlight the hex under the cursor and report clicks on it."""
        self._local_pick = bool(enabled)
        if not enabled:
            self._hover_hex = None
        self.update()

    def set_hex_click_callback(self, cb: Optional[Callable[[int], None]]) -> None:
        """Called with the regional hex index clicked while local picking is on."""
        self._hex_click_callback = cb

    def set_params(self, params: RegionParams, globe: Optional["World"] = None) -> None:
        """Replace the region tuning and refine the region on screen again.

        Every settlement in ``globe`` is reset to its stage site, since any refinement
        is now stale.  New river or settlement params keep the refined fields.
        """
        if params == self._region_params:
            return
        old, self._region_params = self._region_params, params
        globe = globe if globe is not None else self._globe
        if globe is not None:
            towns = globe.get_feature("towns")
            if towns is not None:
                towns.reset_refinement()
            for city in globe.get_feature("settlements") or []:
                city.reset_refinement()
            self._land_positions = {}
        if self._geometry is None or self._fields is None:
            return
        if params.upsample != old.upsample:
            zoom, pan = self._zoom, self._pan.copy()
            self.set_region(self._geometry, self._globe)
            self._zoom, self._pan = zoom, pan
        else:
            self._refine_features()
            self._rebuild_images()   # the rivers moved: layers that snap to them follow
        self.update()

    def set_realm_params(self, params: Optional["RealmParams"]) -> None:
        """The realm tuning the world was built with, used when refining (changes come with a new world)."""
        self._realm_params = params

    def set_world_params(self, params: Optional["GlobeParams"]) -> None:
        """The physical tuning the world was built with, used when refining (changes come with a new world)."""
        self._world_params = params

    def clear_region(self) -> None:
        """Drop the current region (and its refined fields) until the next ``set_region``."""
        self._geometry = None
        self._globe = None
        self._grid = None
        self._fields = None
        self._local = None
        self._hover_hex = None
        self._region_map = None
        self._rivers = None
        self._roads = None
        self._deltas = None
        self._hex_images = []
        self._farm_images = []
        self._land_positions = {}
        self._cell_layout = None
        self._view_frame = None
        self._drag_start = None

    def set_layer(self, layer: Optional[str], spec: Optional[SchemeSpec] = None) -> None:
        """Show one of the world map's layers, by scheme name and spec."""
        if (layer, spec) == (self._layer, self._spec):
            return
        self._layer = layer
        self._spec = spec
        self._rebuild_images()
        self.update()

    def set_overlay_visible(self, name: str, visible: bool) -> None:
        """Toggle one of the world map's feature overlays (same names, same styling)."""
        self._overlay_visible[name] = visible
        if name in _FARM_OVERLAYS and self._local is not None:
            self._rebuild_farm_images()
        self.update()

    def set_legend(self, legend: Optional[tuple[str, list]]) -> None:
        """The active layer's ``(kind, entries)`` legend, drawn beside the cell."""
        self._legend = legend
        self.update()

    def set_hover_callback(
        self, cb: Optional[Callable[[Optional[tuple[int, int, int]]], None]]
    ) -> None:
        """Called with the ``(hex, row, col)`` texel under the cursor, or None off the map."""
        self._hover_callback = cb

    @property
    def layer(self) -> Optional[str]:
        return self._layer

    @property
    def region_id(self) -> Optional[int]:
        return self._geometry.region_id if self._geometry is not None else None

    @property
    def fields(self) -> Optional[RegionFields]:
        return self._fields

    @property
    def hex_local(self) -> np.ndarray:
        """(M, 2) hex-centre coordinates in the cell's local [0, 1]^2 frame."""
        if self._grid is None:
            return np.empty((0, 2), dtype=np.float64)
        return self._grid.centers

    @property
    def hex_ecef(self) -> np.ndarray:
        """(M, 3) ECEF unit-sphere position of each hex centre."""
        if self._grid is None:
            return np.empty((0, 3), dtype=np.float64)
        return self._grid.centers_ecef()

    def status_text(self) -> str:
        """One-line summary of the refinement, for the window's status bar."""
        if self._fields is None or self._grid is None:
            return f"Region {self.region_id}"
        f = self._fields
        if self._local is not None:
            v = self._local.villages
            return (
                f"Local map · region {f.region_id} hex {self._local.hex_index} · {f.n_hexes} hexes × "
                f"{f.tex_size}² texels (~{f.texel_area_km2:.2f} km²/texel) · {len(v)} villages "
                f"({float(v.population.sum()):,.0f} people)"
            )
        return (
            f"Region {f.region_id} · {f.n_hexes} hexes × {f.tex_size}² texels "
            f"(~{f.texel_area_km2:.0f} km²/texel) · {f.detail_octaves} detail octaves"
        )

    def _rebuild_images(self) -> None:
        """Convert each hexagon's refined texture to a QImage for painting."""
        self._hex_images = []
        if self._fields is None or self._spec is None:
            return
        rgba = build_layer_texture(                                 # (M, T, T, 4)
            self._fields, self._spec, self._globe.mesh if self._globe is not None else None,
            river_segments=self._river_segments(),
        )
        data = np.ascontiguousarray((rgba * 255.0).clip(0, 255).astype(np.uint8))
        m, h, w, _ = data.shape
        for i in range(m):
            plane = np.ascontiguousarray(data[i])
            self._hex_images.append(
                QImage(plane.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
            )

    def _rebuild_farm_images(self) -> None:
        """A local map's farmland as one image per hexagon, for the settlement overlays shown."""
        self._farm_images = []
        farmland = self._local.farmland if self._local is not None else None
        if farmland is None:
            return
        show = (
            bool(self._overlay_visible.get("villages", True)),
            bool(self._overlay_visible.get("towns")),
            bool(self._overlay_visible.get("settlements")),
        )
        rgba = farmland_rgba(farmland, show)
        if not rgba[..., 3].any():
            return
        data = np.ascontiguousarray((rgba * 255.0).clip(0, 255).astype(np.uint8))
        m, h, w, _ = data.shape
        for i in range(m):
            plane = np.ascontiguousarray(data[i])
            self._farm_images.append(
                QImage(plane.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
            )

    # ── painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(24, 28, 34))
        self._cell_layout = None

        if self._grid is None:
            return
        W, H = float(self.width()), float(self.height())
        margin = 16.0
        legend_gap = 12.0
        legend = self._legend if self._legend is not None and len(self._legend[1]) >= 2 else None
        legend_w, legend_h = (
            legend_size(painter, legend[0], legend[1], H - 2.0 * margin) if legend else (0, 0)
        )
        reserved = legend_w + legend_gap if legend else 0.0

        # Size the cell so its *rotated* footprint fits beside the legend; the
        # rotation about the centre swings the corners out by |cos| + |sin|.
        angle = self._up_angle()
        spread = abs(math.cos(angle)) + abs(math.sin(angle))
        footprint = min(H - 2.0 * margin, W - 2.0 * margin - reserved)
        size = footprint / spread
        if size <= 0.0:
            return
        # Centred when there is room, otherwise pushed right to clear the legend.
        fx0 = max((W - footprint) * 0.5, margin + reserved)
        fy0 = (H - footprint) * 0.5
        x0 = fx0 + 0.5 * (footprint - size)
        y0 = fy0 + 0.5 * (footprint - size)
        cx, cy = fx0 + 0.5 * footprint, fy0 + 0.5 * footprint
        self._view_frame = (cx, cy, footprint)
        transform = self._cell_transform(angle, x0, y0, size) * self._view_transform()
        self._cell_layout = (x0, y0, size, transform)

        painter.save()
        # Zoomed in, the map fills everything right of the legend but never covers it.
        if legend:
            left = fx0 - 0.5 * legend_gap
            painter.setClipRect(QRectF(left, 0.0, W - left, H))
        painter.setTransform(transform, True)
        if self._hex_images:
            self._draw_textured_hexes(painter, x0, y0, size)
        else:
            self._draw_hex_grid(painter, x0, y0, size)

        painter.setPen(_cosmetic_pen(QColor(70, 85, 100)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(self._outline(x0, y0, size))
        if self._local_pick and self._hover_hex is not None and self._local is None:
            r_px = self._grid.radius * size
            hs, ht = self._grid.centers[self._hover_hex]
            pen = _cosmetic_pen(QColor(255, 235, 140, 230))
            pen.setWidthF(2.0)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 235, 140, 40))
            painter.drawPolygon(self._hex_polygon(x0 + hs * size, y0 + (1.0 - ht) * size, r_px))
        painter.restore()

        # Overlays in screen space (markers stay upright, pixel-sized), clipped to the rotated cell.
        if self._globe is not None and self._geometry is not None:
            painter.save()
            if legend:
                left = fx0 - 0.5 * legend_gap
                painter.setClipRect(QRectF(left, 0.0, W - left, H))
            cell = QPainterPath()
            cell.addPolygon(transform.map(self._outline(x0, y0, size)))
            painter.setClipPath(cell, Qt.ClipOperation.IntersectClip)
            draw_overlays(
                painter, self._globe, self._overlay_visible, self._overlay_projector(),
                refined=self._refined_features(),
            )
            painter.restore()

        # Outside the map, bottom-aligned with the cell's footprint to its left.
        if legend:
            draw_legend(
                painter, legend[0], legend[1],
                fx0 - legend_gap - legend_w, fy0 + footprint, H - 2.0 * margin,
            )

    def _refined_features(self) -> dict:
        """Overlay features drawn finer than the world map: re-routed rivers and roads, snapped
        deltas, mineral workings; local maps add villages, their towns and city footprints.
        (Region towns and cities need no override: their refined positions are in the world features.)"""
        out: dict = {}
        if self._local is not None:
            out["villages"] = self._local.villages
            towns = self._globe.get_feature("towns") if self._globe is not None else None
            if towns is not None and len(towns):
                idx = self._local.town_index
                out["towns"] = Towns(
                    positions=self._local.town_positions, population=towns.population[idx],
                    face_id=towns.face_id[idx], parent=towns.parent[idx], kind=towns.kind[idx],
                    site_positions=towns.site_positions[idx], refined=towns.refined[idx],
                    has_walls=towns.has_walls[idx], wall_strength=towns.wall_strength[idx],
                )
            if self._local.city_footprints is not None:
                out["cities"] = (self._local.city_index, self._local.city_positions, self._local.city_footprints)
        if self._rivers is not None:
            out["rivers"] = [*self._rivers.paths, *self._rivers.streams]
        if self._overlay_visible.get("polity_borders"):
            borders = self._polity_borders()
            if borders is not None:
                out["polity_borders"] = borders
        if self._roads is not None:
            out["trade_routes"] = self._roads.segments
        if self._deltas is not None:
            out["deltas"] = self._deltas
        minerals = self._globe.get_feature("minerals") if self._globe is not None else None
        workings = mineral_workings(self._globe) if minerals is not None else None
        if minerals is not None and workings is not None:
            # Each deposit as the workings it is split into, sized by their share.
            pos, deposit, share = workings
            out["minerals"] = (
                self._on_land("mineral_sites", pos), np.asarray(minerals[1])[deposit],
                np.asarray(minerals[2], dtype=np.float64)[deposit] * share,
            )
        return out

    def _polity_entry(self) -> Optional[tuple]:
        """The polity cache for the rivers and fields on screen (rebuilt when either changes)."""
        if self._rivers is None or self._fields is None:
            return None
        c = self._polity_cache
        if c is None or c[0] is not self._rivers or c[1] is not self._fields:
            c = (self._rivers, self._fields,
                 pack_river_segments(self._rivers.paths, river_ids=self._rivers.river_ids()), None)
            self._polity_cache = c
        return c

    def _river_segments(self) -> Optional[np.ndarray]:
        """Packed segments of the rivers drawn here, for layers that snap to them."""
        c = self._polity_entry()
        return c[2] if c is not None else None

    def _polity_borders(self) -> Optional[dict]:
        """Border lines contoured from this view's Polities fill (its rivers), cached."""
        c = self._polity_entry()
        globe, fields = self._globe, self._fields
        field = globe.get_feature("polity_label_field") if globe is not None else None
        if c is None or field is None or globe is None or fields is None:
            return None
        if c[3] is None:
            borders = view_polity_borders(fields, field, globe.get_feature("polities") or [], c[2])
            c = self._polity_cache = (c[0], c[1], c[2], borders)
        return c[3]

    def _overlay_projector(self) -> "_RegionProjector":
        """The projector overlays are drawn with, for the cell as last laid out."""
        assert self._geometry is not None and self._cell_layout is not None
        x0, y0, size, transform = self._cell_layout
        return _RegionProjector(
            self._geometry, x0, y0, size, transform, self._equivalent_zoom(size, float(self.width())),
            self._on_land, view_zoom=self._zoom,
            px_per_km=self._px_per_rad(size) / EARTH_RADIUS_KM if self._local is not None else None,
        )

    def _land_reach(self) -> float:
        """How far (chord) a settlement on refined water is moved onto land: the
        refinement's water spread plus two texels of slack."""
        f = self._fields
        if f is None:
            return 0.0
        upsample = self._local_params.upsample if self._local is not None else self._region_params.upsample
        return land_reach(f, upsample.water_spread_texels)

    def _on_land(self, key: str, ecef: np.ndarray) -> np.ndarray:
        """``ecef`` moved off the refined water (``snap_to_land``), cached per feature."""
        if self._fields is None:
            return ecef
        ecef = np.asarray(ecef, dtype=np.float64).reshape(-1, 3)
        cached = self._land_positions.get(key)
        if cached is not None and np.array_equal(cached[0], ecef):
            return cached[1]
        moved = snap_to_land(self._fields, ecef, self._land_reach())
        self._land_positions[key] = (ecef.copy(), moved)
        return moved

    def _up_angle(self) -> float:
        """Screen rotation (radians) that turns ``RegionGeometry.up_direction()`` to screen-up.

        Up is true north, or the antimeridian tangent on the two polar regions.
        +t is already screen-up unrotated.
        """
        if self._geometry is None:
            return 0.0
        ds, dt = self._geometry.up_direction()
        angle = math.atan2(-ds, dt)
        return 0.0 if abs(angle) < 1e-9 else angle

    def _equivalent_zoom(self, size: float, W: float) -> float:
        """The world-map zoom showing this cell's ground at its on-screen scale.

        Overlays size markers and lines by zoom; world zoom 1 spans 2π of longitude across the width.
        """
        return self._px_per_rad(size) / (W / (2.0 * np.pi))

    def _px_per_rad(self, size: float) -> float:
        """Screen pixels per radian of ground across the cell, as drawn now."""
        assert self._geometry is not None
        ends = self._geometry.local_to_ecef(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        side_rad = float(np.arccos(np.clip(np.dot(ends[0], ends[1]), -1.0, 1.0)))
        return size * self._zoom / max(side_rad, 1e-9)

    def _view_transform(self) -> QTransform:
        """Zoom about the footprint centre, then pan — applied after the rotation."""
        t = QTransform()
        if self._view_frame is None:
            return t
        cx, cy, footprint = self._view_frame
        t.translate(cx + self._pan[0] * footprint, cy + self._pan[1] * footprint)
        t.scale(self._zoom, self._zoom)
        t.translate(-cx, -cy)
        return t

    # ── zoom / pan ───────────────────────────────────────────────────────────

    @property
    def zoom(self) -> float:
        return self._zoom

    def zoom_toward(self, mx: float, my: float, factor: float) -> None:
        """Scale by ``factor`` keeping the screen point ``(mx, my)`` fixed."""
        if self._view_frame is None:
            return
        cx, cy, footprint = self._view_frame
        new_zoom = max(1.0, min(self.MAX_ZOOM, self._zoom * factor))
        if new_zoom == 1.0:
            self._pan[:] = 0.0
        else:
            # The unzoomed point under the cursor must land under it again.
            px = cx + self._pan[0] * footprint
            py = cy + self._pan[1] * footprint
            qx = (mx - px) / self._zoom
            qy = (my - py) / self._zoom
            self._pan[0] = (mx - cx - new_zoom * qx) / footprint
            self._pan[1] = (my - cy - new_zoom * qy) / footprint
        self._zoom = new_zoom
        self._clamp_pan()
        self.update()

    def zoom_in(self) -> None:
        self.zoom_toward(self.width() * 0.5, self.height() * 0.5, 1.2)

    def zoom_out(self) -> None:
        self.zoom_toward(self.width() * 0.5, self.height() * 0.5, 1.0 / 1.2)

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan[:] = 0.0
        self.update()

    def _clamp_pan(self) -> None:
        """Keep the footprint centre's view covered — no panning off into the void."""
        limit = 0.5 * (self._zoom - 1.0)
        self._pan[:] = np.clip(self._pan, -limit, limit)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta:
            pos = event.position()
            self.zoom_toward(float(pos.x()), float(pos.y()), 1.2 ** (delta / 120.0))
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._press_pos = (float(pos.x()), float(pos.y()))
            self._drag_start = (float(pos.x()), float(pos.y()))
            self._drag_start_pan = self._pan.copy()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            pos = event.position()
            press, self._press_pos = self._press_pos, None
            clicked = press is not None and math.hypot(pos.x() - press[0], pos.y() - press[1]) < self._CLICK_SLOP
            if clicked and self._local_pick and self._local is None and self._hex_click_callback is not None:
                texel = self.texel_at_screen(float(pos.x()), float(pos.y()))
                if texel is not None:
                    self._hex_click_callback(texel[0])

    @staticmethod
    def _cell_transform(angle: float, x0: float, y0: float, size: float) -> QTransform:
        """Rotation of the whole cell about its centre by ``angle``."""
        cx, cy = x0 + 0.5 * size, y0 + 0.5 * size
        t = QTransform()
        t.translate(cx, cy)
        t.rotate(math.degrees(angle))
        t.translate(-cx, -cy)
        return t

    # ── hover ────────────────────────────────────────────────────────────────

    def texel_at_screen(self, mx: float, my: float) -> Optional[tuple[int, int, int]]:
        """``(hex, row, col)`` of the refined texel under a screen point, or None."""
        if self._cell_layout is None or self._fields is None:
            return None
        x0, y0, size, transform = self._cell_layout
        inverse, ok = transform.inverted()
        if not ok:
            return None
        p = inverse.map(QPointF(mx, my))
        s = (p.x() - x0) / size
        t = 1.0 - (p.y() - y0) / size
        texel = self._fields.grid.texel_at(s, t, self._fields.tex_size)
        if texel is not None and self._fields.in_map is not None and not self._fields.in_map[texel]:
            return None                     # a local map's square corner, outside the parent hexagon
        return texel

    def settlements_under_cursor(self, texel: Optional[tuple[int, int, int]]) -> list[str]:
        """Status fragments for the settlements at the cursor: the fort, the city, then the
        town/village dot (local map) or the towns in ``texel``'s hexagon (region)."""
        parts: list[str] = []
        for item in (self.fort_under_cursor(), self.city_under_cursor(texel)):
            if item is not None:
                parts.append(item)
        if self._local is None:
            parts.extend(self.towns_in_hex(texel[0]) if texel is not None else [])
        else:
            for item in (self.town_under_cursor(), self.village_under_cursor()):
                if item is not None:
                    parts.append(item)
        return parts

    def _cursor_ready(self) -> bool:
        return (
            self._globe is not None and self._geometry is not None
            and self._cell_layout is not None and self._mouse is not None
        )

    def _nearest_marker(self, sx: np.ndarray, sy: np.ndarray, keep: np.ndarray, radii) -> Optional[int]:
        """Index of the nearest marker within ``radius + _MARKER_HOVER_SLOP`` of the cursor, or None."""
        assert self._mouse is not None
        d = np.where(keep, np.hypot(sx - self._mouse[0], sy - self._mouse[1]), np.inf)
        d = np.where(d <= np.asarray(radii) + self._MARKER_HOVER_SLOP, d, np.inf)
        if d.size == 0 or not np.isfinite(d.min()):
            return None
        return int(np.argmin(d))

    def city_under_cursor(self, texel: Optional[tuple[int, int, int]]) -> Optional[str]:
        """``"<tier> · <people>"`` for the city under the cursor (marker, or local-map footprint)."""
        if not self._cursor_ready() or not self._overlay_visible.get("settlements"):
            return None
        assert self._globe is not None
        cities = self._globe.get_feature("settlements") or []
        if not cities:
            return None
        proj = self._overlay_projector()
        footprints = self._local.city_footprints if self._local is not None else None
        if self._local is not None and footprints is not None:
            index = np.asarray(self._local.city_index, dtype=np.int64)
            c = int(footprints.owner[texel]) if texel is not None else -1
            if c < 0:
                sx, sy, keep = proj.points(np.asarray(self._local.city_positions, dtype=np.float64))
                found = self._nearest_marker(sx, sy, keep, np.full(len(index), city_floor_radius(proj, walled=True)))
                c = -1 if found is None else found
            if c < 0:
                return None
            w = int(index[c])
            max_pop = max(float(city.population) for city in cities)
            tier = population_display_tier(float(cities[w].population), max_pop)
        else:
            markers = settlement_markers(self._globe, proj)
            if markers is None:
                return None
            sx, sy, keep, tiers, radii = markers
            found = self._nearest_marker(sx, sy, keep, radii)
            if found is None:
                return None
            w, tier = found, int(tiers[found])
        return f"{TIER_NAMES[min(tier, len(TIER_NAMES) - 1)]} · {float(cities[w].population):,.0f} people"

    def fort_under_cursor(self) -> Optional[str]:
        """``"Fort (<garrison> men)"`` for the fort keep under the cursor, if any."""
        if not self._cursor_ready() or not self._overlay_visible.get("forts"):
            return None
        assert self._globe is not None
        forts = self._globe.get_feature("forts")
        if forts is None or len(forts) == 0:
            return None
        sx, sy, keep, r = fort_markers(forts, self._overlay_projector())
        found = self._nearest_marker(sx, sy, keep, np.full(len(forts), r))
        if found is None:
            return None
        return f"Fort ({float(forts.garrison[found]):.0f} men)"

    def town_under_cursor(self) -> Optional[str]:
        """``"<kind> · <people>"`` for a local map's town dot under the cursor, if any."""
        if self._local is None or not self._cursor_ready() or not self._overlay_visible.get("towns"):
            return None
        towns = self._refined_features().get("towns")
        if towns is None or len(towns) == 0:
            return None
        sx, sy, keep, radii = town_markers(towns, self._overlay_projector())
        found = self._nearest_marker(sx, sy, keep, radii)
        if found is None:
            return None
        return f"{TOWN_KIND_NAMES[int(towns.kind[found])]} · {float(towns.population[found]):,.0f} people"

    def towns_in_hex(self, hex_idx: int) -> list[str]:
        """``["Towns in hex: N (people)", "<count> <kind>, …"]`` for a region hexagon; empty if none."""
        if self._local is not None or self._globe is None or self._fields is None or self._geometry is None:
            return []
        towns = self._globe.get_feature("towns")
        if towns is None or len(towns) == 0:
            return []
        s, t, valid = self._geometry.ecef_to_local(self._on_land("towns", towns.positions))
        hexes, _, _, inside = self._fields.grid.texels_at(s, t, self._fields.tex_size)
        mine = valid & inside & (hexes == hex_idx)
        if not mine.any():
            return []
        kinds = np.bincount(towns.kind[mine], minlength=len(TOWN_KIND_NAMES))
        return [
            f"Towns in hex: {int(mine.sum())} ({format_population(float(towns.population[mine].sum()))})",
            ", ".join(f"{k} {TOWN_KIND_NAMES[i].split()[0].lower()}" for i, k in enumerate(kinds) if k),
        ]

    def village_under_cursor(self) -> Optional[str]:
        """``"<kind> · <people>"`` for the village dot under the cursor, if any."""
        if self._local is None or self._cell_layout is None or self._mouse is None or self._geometry is None:
            return None
        v = self._local.villages
        if len(v) == 0:
            return None
        x0, y0, size, transform = self._cell_layout
        proj = _RegionProjector(self._geometry, x0, y0, size, transform, 1.0)
        sx, sy, keep = proj.points(v.positions)
        d = np.where(keep, np.hypot(sx - self._mouse[0], sy - self._mouse[1]), np.inf)
        i = int(np.argmin(d))
        if d[i] > self._VILLAGE_HOVER_PX:
            return None
        return f"{VILLAGE_KIND_NAMES[int(v.kind[i])]} · {float(v.population[i]):,.0f} people"

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        mx, my = float(pos.x()), float(pos.y())
        self._mouse = (mx, my)
        if self._local_pick and self._local is None and self._drag_start is None:
            texel = self.texel_at_screen(mx, my)
            hover = texel[0] if texel is not None else None
            if hover != self._hover_hex:
                self._hover_hex = hover
                self.update()
        if self._drag_start is not None and self._view_frame is not None:
            footprint = self._view_frame[2]
            self._pan[0] = self._drag_start_pan[0] + (mx - self._drag_start[0]) / footprint
            self._pan[1] = self._drag_start_pan[1] + (my - self._drag_start[1]) / footprint
            self._clamp_pan()
            self.update()
        if self._hover_callback is not None:
            self._hover_callback(self.texel_at_screen(mx, my))

    def leaveEvent(self, event) -> None:
        self._mouse = None
        if self._hover_hex is not None:
            self._hover_hex = None
            self.update()
        if self._hover_callback is not None:
            self._hover_callback(None)

    def _outline(self, x0: float, y0: float, size: float) -> QPolygonF:
        """The map's edge in cell-layout pixels: the region's square, or a local
        map's parent hexagon (flat-top, filling the square's width)."""
        if self._local is None:
            return QPolygonF(QRectF(x0, y0, size, size))
        return self._hex_polygon(x0 + PARENT_CENTER * size, y0 + (1.0 - PARENT_CENTER) * size, PARENT_RADIUS * size)

    def _clip_to_outline(self, painter: QPainter, x0: float, y0: float, size: float) -> None:
        path = QPainterPath()
        path.addPolygon(self._outline(x0, y0, size))
        painter.setClipPath(path, Qt.ClipOperation.IntersectClip)

    def _hex_polygon(self, cx: float, cy: float, r_px: float) -> QPolygonF:
        return QPolygonF([
            QPointF(cx + r_px * math.cos(math.radians(60 * i)),
                    cy + r_px * math.sin(math.radians(60 * i)))
            for i in range(6)
        ])

    def _draw_textured_hexes(
        self, painter: QPainter, x0: float, y0: float, size: float
    ) -> None:
        """Blit each hexagon's texture, clipped to the hexagon.

        The texture square spans ``2R`` centred on the hex, row 0 at the top, so the
        image-to-screen map is a scale plus translate; the over-hanging corners are clipped.
        """
        assert self._grid is not None and self._fields is not None
        r_px = self._grid.radius * size
        tex_size = self._fields.tex_size
        scale = (2.0 * r_px) / tex_size
        # Overdraw a sub-pixel hair so antialiased neighbours don't leave a seam
        # (the over-hang is real data for the neighbour anyway).
        r_draw = r_px + 0.75 / self._zoom

        painter.save()
        self._clip_to_outline(painter, x0, y0, size)
        painter.setPen(Qt.PenStyle.NoPen)
        for i, (s, t) in enumerate(self._grid.centers):
            cx = x0 + s * size
            cy = y0 + (1.0 - t) * size       # local +t is up; _up_angle makes that true north
            brush = QBrush(self._hex_images[i])
            brush.setTransform(
                QTransform(scale, 0.0, 0.0, scale, cx - r_px, cy - r_px)
            )
            painter.setBrush(brush)
            painter.drawPolygon(self._hex_polygon(cx, cy, r_draw))
        # Farmland: over the terrain, under the overlays, no overdraw.
        for i, (s, t) in enumerate(self._grid.centers if self._farm_images else []):
            cx = x0 + s * size
            cy = y0 + (1.0 - t) * size
            brush = QBrush(self._farm_images[i])
            brush.setTransform(QTransform(scale, 0.0, 0.0, scale, cx - r_px, cy - r_px))
            painter.setBrush(brush)
            painter.drawPolygon(self._hex_polygon(cx, cy, r_px))

        # Faint hex outlines: in a region only while picking a local map.
        if self._local is not None or self._local_pick:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(_cosmetic_pen(QColor(20, 25, 30, 60)))
            for s, t in self._grid.centers:
                painter.drawPolygon(
                    self._hex_polygon(x0 + s * size, y0 + (1.0 - t) * size, r_px)
                )
        painter.restore()

    def _draw_hex_grid(self, painter: QPainter, x0: float, y0: float, size: float) -> None:
        """Fallback: bare scaffolding when no globe data is available."""
        assert self._grid is not None
        r_px = self._grid.radius * size
        painter.save()
        self._clip_to_outline(painter, x0, y0, size)
        painter.setPen(_cosmetic_pen(QColor(90, 110, 130)))
        painter.setBrush(QColor(34, 40, 48))
        for s, t in self._grid.centers:
            cx = x0 + s * size
            cy = y0 + (1.0 - t) * size
            painter.drawPolygon(self._hex_polygon(cx, cy, r_px))
        painter.restore()


def _cosmetic_pen(color: QColor) -> QPen:
    """A 1-px line that stays 1 px however far the view is zoomed."""
    pen = QPen(color, 1.0)
    pen.setCosmetic(True)
    return pen


class _RegionProjector:
    """Overlay projector for a region: the cell's local frame, then its screen transform.

    Features beyond ``_MARGIN`` of the cell (local units) are culled; the far side has no image.
    """

    _MARGIN = 0.15

    def __init__(
        self, geometry: RegionGeometry, x0: float, y0: float, size: float,
        transform: QTransform, zoom: float,
        on_land: Optional[Callable[[str, np.ndarray], np.ndarray]] = None,
        view_zoom: float = 1.0, px_per_km: Optional[float] = None,
    ) -> None:
        self._geometry = geometry
        self._x0, self._y0, self._size = x0, y0, size
        self._t = transform
        self._zoom = zoom
        self._on_land = on_land
        self._view_zoom = view_zoom
        self._px_per_km = px_per_km

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def view_zoom(self) -> float:
        return self._view_zoom

    @property
    def px_per_km(self) -> Optional[float]:
        return self._px_per_km

    @property
    def regional(self) -> bool:
        return True

    def on_land(self, key: str, ecef: np.ndarray) -> np.ndarray:
        return self._on_land(key, ecef) if self._on_land is not None else ecef

    def _project(self, ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(sx, sy, valid, near)``: screen position, has-an-image, near-the-cell."""
        s, t, valid = self._geometry.ecef_to_local(np.asarray(ecef).reshape(-1, 3))
        m = self._MARGIN
        near = valid & (s > -m) & (s < 1.0 + m) & (t > -m) & (t < 1.0 + m)
        x = self._x0 + s * self._size
        y = self._y0 + (1.0 - t) * self._size
        # QTransform maps x' = m11 x + m21 y + dx, y' = m12 x + m22 y + dy.
        tr = self._t
        sx = tr.m11() * x + tr.m21() * y + tr.dx()
        sy = tr.m12() * x + tr.m22() * y + tr.dy()
        return sx, sy, valid, near

    def points(self, ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sx, sy, _, near = self._project(ecef)
        return sx, sy, near

    def segments(
        self, p0: np.ndarray, p1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sx0, sy0, valid0, near0 = self._project(p0)
        sx1, sy1, valid1, near1 = self._project(p1)
        return sx0, sy0, sx1, sy1, valid0 & valid1 & (near0 | near1)

    def split_segment(self, p0: np.ndarray, p1: np.ndarray) -> list[tuple[float, float, float, float]]:
        sx0, sy0, sx1, sy1, keep = self.segments(p0, p1)
        if not keep[0]:
            return []
        return [(float(sx0[0]), float(sy0[0]), float(sx1[0]), float(sy1[0]))]

    def polygons(self, verts: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        sx, sy, valid, near = self._project(verts)
        if not valid.all() or not near.any():
            return []
        return [(sx, sy)]
