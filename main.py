import dataclasses
import random
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication, QFileDialog, QWidget

from core.persistence import load_world, save_world
from core.pipeline import BuildContext, Pipeline
from core.world import World
from viz.mapview.mapview import MapWindow
from viz.mapview.projections import projection_by_name
from viz.params_dialog import ParamsDialog
from realmgen.builder import RealmParams, RealmPipeline
from localmap.params import LocalParams
from localmap.upsample import with_region_detail
from settlegen.params import SettlementMapParams
from settlegen.upsample import with_local_detail
from region.params import RegionParams
from worldgen import GlobePipeline, GlobeParams


@dataclasses.dataclass
class AppParams:
    # World seed — drives mesh selection and every stage's draws. None → random each run.
    seed: Optional[int] = dataclasses.field(
        default=None, metadata={"label": "Seed", "kind": "optional_int", "min": 0, "max": 2**31 - 1}
    )
    globe: GlobeParams = dataclasses.field(
        default_factory=GlobeParams, metadata={"label": "World"}
    )
    realm: RealmParams = dataclasses.field(
        default_factory=RealmParams, metadata={"label": "Realm / Settlements"}
    )
    # View-side only: never changes the world build (see AppPipeline).
    region: RegionParams = dataclasses.field(
        default_factory=RegionParams, metadata={"label": "Region"}
    )
    local: LocalParams = dataclasses.field(
        default_factory=LocalParams, metadata={"label": "Local map"}
    )
    # Settlement maps; view-side only, like ``region`` / ``local``.
    settlement: SettlementMapParams = dataclasses.field(
        default_factory=SettlementMapParams, metadata={"label": "Settlement map"}
    )

    def __post_init__(self) -> None:
        # Coupled params live once in the dialog; copy them to where the stages read them
        # (every replace() re-runs this, so they can't drift apart).
        self.globe = dataclasses.replace(self.globe, seed=self.seed)
        lapse = self.globe.climate.temperature.lapse_rate_c_per_km
        self.region = dataclasses.replace(
            self.region, upsample=dataclasses.replace(self.region.upsample, lapse_rate_c_per_km=lapse)
        )
        self.local = dataclasses.replace(
            self.local, upsample=with_region_detail(self.local.upsample, self.region.upsample)
        )
        self.settlement = dataclasses.replace(
            self.settlement, upsample=with_local_detail(self.settlement.upsample, self.local.upsample)
        )


# The params dialog's default view: `fnmatch` patterns over dotted field paths.
# `--advanced` shows everything.
BASIC_PARAMS: tuple[str, ...] = (
    "seed",
    "globe.elevation.plates.n_plates",
    "globe.elevation.plates.oceanic_fraction",
    "globe.elevation.ruggedness.base",
    "globe.climate.temperature.equator_annual_c",
    "globe.climate.temperature.pole_annual_c",
    "globe.climate.precipitation.precipitation_scale",
    "globe.climate.currents.enabled",
    "globe.hydrology.river_flow_threshold",
    "globe.hydrology.stream_flow_threshold",
    "globe.delta.min_flow",
    "globe.wetland.threshold",
    "globe.resources.minerals.deposit_threshold",
    "globe.resources.rare_bio.base_threshold",
    "globe.danger.enabled",
    "realm.population.rural_density_max",
    "realm.population.trade_food_weight",
    "realm.population.city_min",
    "realm.polity.projection.sea_range_factor",
    "realm.polity.hierarchy.direct_threshold",
    "realm.polity.hierarchy.vassal_floor",
    "realm.polity.fragmentation.n_epochs",
    "realm.polity.marches.weight",
    "realm.polity.forts.enabled",
    "realm.towns.town_fraction",
    "region.upsample.n_cols",
    "region.upsample.tex_size",
    "region.upsample.relief_factor",
    "region.upsample.refine_water_masks",
    "region.upsample.reshape_coast",
    "region.upsample.connect_ocean",
    "region.upsample.small_lakes",
    "region.rivers.distributaries.enabled",
    "local.upsample.n_cols",
    "local.upsample.tex_size",
    "local.villages.village_fraction",
)


class AppPipeline(Pipeline):
    """Top-level nested pipeline: physical world then human geography."""

    def __init__(self, params: AppParams) -> None:
        self.globe_pipeline = GlobePipeline(params.globe)
        self.realm_pipeline = RealmPipeline(params.realm)
        super().__init__([self.globe_pipeline, self.realm_pipeline])


class AppParamsDialog(ParamsDialog):
    def __init__(self, params: AppParams, *, advanced: bool = False, parent: "QWidget | None" = None) -> None:
        super().__init__(params, title="Parameters", basic=None if advanced else BASIC_PARAMS, parent=parent)

    def result_params(self, base: AppParams) -> AppParams:  # type: ignore[override]
        return super().result_params(base)


def main() -> None:
    debug = "--debug" in sys.argv
    # --advanced shows every param in the dialog, not just BASIC_PARAMS.
    advanced = "--advanced" in sys.argv
    file_arg = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    # --projection=NAME picks the world map's projection; Robinson when absent.
    proj_arg = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--projection=")), None)
    projection = projection_by_name(proj_arg) if proj_arg is not None else None

    app = QApplication(sys.argv)

    window = MapWindow(projection=projection)
    window.set_debug(debug)
    window.setWindowTitle("Procedural World")

    current_params: list[AppParams] = [AppParams()]
    current_pipeline: list[Pipeline | None] = [None]

    def _build_and_apply(params: AppParams) -> None:
        window.set_rebuild_enabled(False)

        def _progress(msg: str) -> None:
            window.set_status(msg)
            QApplication.processEvents()

        if params.seed is None:
            params = dataclasses.replace(params, seed=random.randint(0, 2**31 - 1))
        globe_params = params.globe
        current_params[0] = params
        window.set_region_params(params.region)
        window.set_local_params(params.local)
        window.set_settlement_params(params.settlement)
        window.set_realm_params(params.realm)
        window.set_world_params(params.globe)

        world = World(globe_params.mesh, seed=globe_params.seed, progress=_progress)
        ctx = BuildContext(world=world, progress=_progress)
        pipeline = AppPipeline(params)
        pipeline.build(ctx)
        current_pipeline[0] = pipeline

        prev_scheme = window.active_scheme
        window.set_globe(ctx.world)
        pipeline.apply_display(window, debug=debug)
        if prev_scheme:
            window.set_active_scheme(prev_scheme)

        window.set_status("")
        window.set_rebuild_enabled(True)

    def _load_from_file(path: str) -> None:
        window.set_rebuild_enabled(False)
        window.set_status(f"Loading {Path(path).name}…")
        QApplication.processEvents()

        params, world, data = load_world(path, AppParams)
        pipeline = AppPipeline(params)
        pipeline.load(data, BuildContext(world=world))
        current_params[0] = params
        current_pipeline[0] = pipeline
        window.set_region_params(params.region)
        window.set_local_params(params.local)
        window.set_settlement_params(params.settlement)
        window.set_realm_params(params.realm)
        window.set_world_params(params.globe)

        window.set_globe(world)
        pipeline.apply_display(window, debug=debug)

        window.set_status("")
        window.set_rebuild_enabled(True)

    def _rerun() -> None:
        p = current_params[0]
        _build_and_apply(dataclasses.replace(p, seed=None))

    def _open_params() -> None:
        dialog = AppParamsDialog(current_params[0], advanced=advanced, parent=window)
        if dialog.exec() == AppParamsDialog.DialogCode.Accepted:
            new_params = dialog.result_params(current_params[0])
            if new_params == current_params[0]:
                return
            # Region, local and settlement params only re-refine the views; anything else rebuilds the world.
            old = current_params[0]
            if dataclasses.replace(new_params, region=old.region, local=old.local, settlement=old.settlement) == old:
                current_params[0] = new_params
                window.set_region_params(new_params.region)
                window.set_local_params(new_params.local)
                window.set_settlement_params(new_params.settlement)
            else:
                _build_and_apply(new_params)

    def _save() -> None:
        pipeline = current_pipeline[0]
        if pipeline is None or pipeline.world is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            window, "Save World", "", "World files (*.npz)"
        )
        if not path:
            return
        if not path.endswith(".npz"):
            path += ".npz"
        window.set_status(f"Saving {Path(path).name}…")
        QApplication.processEvents()
        save_world(pipeline.world, current_params[0], pipeline, path)
        window.set_status("")

    window.set_save_callback(_save)
    window.set_rebuild_callback(_rerun)
    window.set_params_callback(_open_params)

    if file_arg:
        _load_from_file(file_arg)
    else:
        _build_and_apply(AppParams())

    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
