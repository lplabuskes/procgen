"""Danger stage — DangerBuilder.

Runs after the resources stage. Seeds per-face danger values from terrain
proxies (geology, climate, hydrology, resources) and writes a clamped [0, 1]
float32 danger field per face. Consumed by realmgen and display.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from core.display import DisplayRegistry
from core.pipeline import BuildContext, Stage
from core.world import World
from worldgen.climate.channels import BIOME
from worldgen.danger.channels import DANGER, DANGER_ARRAY
from worldgen.danger.sources import (
    DangerParams,
    DangerTerrain,
    MAX_COAST_HOPS,
    MAX_OCEAN_DEPTH_HOPS,
    MAX_WATER_HOPS,
    SOURCE_CAT_LEGEND,
    SOURCE_NAMES,
    build_sources,
    coastal_land_faces,
    source_contribution,
)
from worldgen.geology.channels import (
    BOUNDARY_TYPE,
    ELEVATION,
    IS_OCEAN,
    PLATE_TYPE,
    VOLCANIC_ACTIVITY,
)
from worldgen.hydrology.channels import LAKE_MASK, RIVER_OBJECTS
from worldgen.resources.channels import FERTILITY_MEAN
from worldgen.resources.terrain import (
    coastal_ocean_faces,
    face_elevation_range,
    face_hops_from_seeds,
    lake_containing_faces,
    river_traversed_faces,
)


def _pos_to_face(mesh, pos: np.ndarray) -> Optional[int]:
    """Return the face index whose centroid is closest to ``pos``."""
    diffs = mesh.face_centroids - pos.astype(np.float64)
    idx = int(np.argmin((diffs * diffs).sum(axis=1)))
    return idx


class DangerBuilder(Stage):
    """Computes per-face danger values from terrain proxies.

    Slots into the globe pipeline between resources and cost. Writes the
    ``danger`` MetaChannel (per-face metadata) and ``danger`` FeatureChannel
    (numpy array) for realmgen consumption.
    """

    name = "danger"

    def __init__(self, params: DangerParams | None = None) -> None:
        self.params: DangerParams = params or DangerParams()
        self.danger: Optional[np.ndarray] = None   # (N,) float32
        self.sources: Optional[tuple] = None        # (positions (M,3), kinds (M,) int8)

    def build(self, ctx: BuildContext) -> None:
        mesh = ctx.mesh
        rng = ctx.rng("danger")
        n = len(mesh.faces)

        ctx.report("Danger: reading inputs…")
        volcanic_activity = VOLCANIC_ACTIVITY.read(mesh)
        is_ocean = IS_OCEAN.read(mesh)
        biome = BIOME.read(mesh)
        elevation_tex = ELEVATION.read(mesh)
        lake_mask_tex = LAKE_MASK.read(mesh) if LAKE_MASK.present(mesh) else None
        rivers = RIVER_OBJECTS.read(ctx.world)
        fertility_mean = FERTILITY_MEAN.read(mesh).astype(np.float32)
        forest_density_mean = np.array(
            [m.get("forest_density_mean", 0.0) for m in mesh.face_metadata],
            dtype=np.float32,
        )
        boundary_type = BOUNDARY_TYPE.read(mesh)
        plate_type = PLATE_TYPE.read(mesh)

        is_lake_face = (
            lake_containing_faces(lake_mask_tex) if lake_mask_tex is not None
            else np.zeros(n, dtype=bool)
        )
        is_water = is_ocean | is_lake_face
        elev_mean = elevation_tex[..., 0].mean(axis=(1, 2)).astype(np.float32)
        elev_range = face_elevation_range(elevation_tex)

        ctx.report("Danger: computing terrain inputs…")
        ocean_depth_hops = face_hops_from_seeds(
            mesh, coastal_ocean_faces(mesh, is_ocean), max_hops=MAX_OCEAN_DEPTH_HOPS,
        )
        coast_hops = face_hops_from_seeds(
            mesh, coastal_land_faces(mesh, is_ocean), max_hops=MAX_COAST_HOPS,
        )
        water_seed_mask = river_traversed_faces(rivers, n_faces=n)
        if lake_mask_tex is not None:
            water_seed_mask |= lake_containing_faces(lake_mask_tex)
        water_hops = face_hops_from_seeds(mesh, water_seed_mask, max_hops=MAX_WATER_HOPS)

        terrain = DangerTerrain(
            n=n,
            is_water=is_water,
            is_ocean=is_ocean,
            elev_mean=elev_mean,
            elev_range=elev_range,
            volcanic_activity=volcanic_activity,
            biome=biome,
            fertility_mean=fertility_mean,
            forest_density_mean=forest_density_mean,
            ocean_depth_hops=ocean_depth_hops,
            coast_hops=coast_hops,
            water_hops=water_hops,
            river_mask=river_traversed_faces(rivers, n_faces=n),
            boundary_type=boundary_type,
            plate_type=plate_type,
        )

        ctx.report("Danger: seeding sources…")
        sources = build_sources(self.params)

        danger = np.zeros(n, dtype=np.float32)
        _seed_records: list[tuple[int, int]] = []

        for kind, source in enumerate(sources):
            if source.params.enabled:
                contrib, seeds = source_contribution(mesh, rng, source.mask(terrain), source.params)
                danger += contrib
                for s in seeds:
                    _seed_records.append((int(s), kind))

        danger = np.clip(danger, 0.0, 1.0).astype(np.float32)
        self.danger = danger

        DANGER.write(mesh, danger)
        DANGER_ARRAY.write(ctx.world, danger)

        centroids = mesh.face_centroids.astype(np.float32)
        if _seed_records:
            rec_faces = np.array([f for f, _ in _seed_records], dtype=np.intp)
            rec_kinds = np.array([k for _, k in _seed_records], dtype=np.int8)
            positions = centroids[rec_faces]
        else:
            positions = np.empty((0, 3), dtype=np.float32)
            rec_kinds = np.empty(0, dtype=np.int8)

        self.sources = (positions, rec_kinds)
        ctx.world.add_feature("danger_sources", self.sources)

        for face_idx, kind in _seed_records:
            meta = mesh.face_metadata[face_idx]
            srcs = meta.get("danger_sources")
            if srcs is None:
                meta["danger_sources"] = [SOURCE_NAMES[kind]]
            elif SOURCE_NAMES[kind] not in srcs:
                srcs.append(SOURCE_NAMES[kind])

        ctx.report("Danger done.")

    # ── Display ───────────────────────────────────────────────────────────────

    def register_display(
        self, world: World, reg: DisplayRegistry, *, debug: bool = False
    ) -> None:
        if not debug:
            return
        from worldgen.danger.display import danger_color, danger_legend_entries

        reg.add_scheme(
            "Danger", danger_color,
            legend=("gradient", danger_legend_entries()),
        )

        if world.has_feature("danger_sources"):
            reg.add_overlay(
                "danger_sources", "Danger Sources",
                legend=("categorical", SOURCE_CAT_LEGEND),
                checked=False
            )

        mesh = world.mesh

        def _danger_hover(face: int, _uv, _w) -> list:
            meta = mesh.face_metadata[face]
            parts: list = []
            if "danger" in meta:
                parts.append(f"Danger: {meta['danger']:.2f}")
            for src in meta.get("danger_sources", []):
                parts.append(src)
            return parts

        reg.add_hover(_danger_hover)

    # ── Persistence ───────────────────────────────────────────────────────────

    def serialize(self) -> dict:
        arrays: dict = {}
        if self.danger is not None:
            arrays["danger"] = self.danger.astype(np.float32)
        if self.sources is not None:
            positions, kinds = self.sources
            if len(positions):
                arrays["danger_source_positions"] = positions.astype(np.float32)
                arrays["danger_source_kinds"] = kinds.astype(np.int8)
        return arrays

    def load(self, arrays: Mapping[str, np.ndarray], ctx: BuildContext) -> None:
        if "danger" in arrays:
            danger = arrays["danger"].astype(np.float32)
            self.danger = danger
            DANGER.write(ctx.mesh, danger)
            DANGER_ARRAY.write(ctx.world, danger)

        if "danger_source_positions" in arrays:
            positions = arrays["danger_source_positions"].astype(np.float32)
            kinds = arrays["danger_source_kinds"].astype(np.int8)
            self.sources = (positions, kinds)
            ctx.world.add_feature("danger_sources", self.sources)

            mesh = ctx.mesh
            for pos, kind in zip(positions, kinds):
                face_idx = _pos_to_face(mesh, pos)
                if face_idx is None:
                    continue
                meta = mesh.face_metadata[face_idx]
                srcs = meta.get("danger_sources")
                name = SOURCE_NAMES[int(kind)]
                if srcs is None:
                    meta["danger_sources"] = [name]
                elif name not in srcs:
                    srcs.append(name)
