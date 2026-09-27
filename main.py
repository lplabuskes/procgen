import dataclasses
import random
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFileDialog, QWidget

from core.persistence import load_world, save_world
from core.pipeline import BuildContext, Pipeline
from core.world import World
from viz.mapview.mapview import MapWindow
from viz.params_dialog import ParamsDialog
from realmgen.builder import RealmParams, RealmPipeline
from localmap.params import LocalParams
from region.params import RegionParams
from worldgen import GlobePipeline, GlobeParams


@dataclasses.dataclass
class AppParams:
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


class AppPipeline(Pipeline):
    """Top-level nested pipeline: physical world then human geography.

    Both sub-pipelines are ordinary :class:`~core.pipeline.Stage` objects, so a
    single ``build`` / ``apply_display`` walks the whole tree. Persistence stays
    scoped to the physical world via :attr:`globe_pipeline`.
    """

    def __init__(self, params: AppParams) -> None:
        self.globe_pipeline = GlobePipeline(params.globe)
        self.realm_pipeline = RealmPipeline(params.realm)
        super().__init__([self.globe_pipeline, self.realm_pipeline])


class AppParamsDialog(ParamsDialog):
    def __init__(self, params: AppParams, parent: "QWidget | None" = None) -> None:
        super().__init__(params, title="Parameters", parent=parent)

    def result_params(self, base: AppParams) -> AppParams:  # type: ignore[override]
        return super().result_params(base)


def main() -> None:
    debug = "--debug" in sys.argv
    file_arg = next((a for a in sys.argv[1:] if not a.startswith("-")), None)

    app = QApplication(sys.argv)

    window = MapWindow(debug=debug)
    window.setWindowTitle("Procedural World")

    current_params: list[AppParams] = [AppParams()]
    current_pipeline: list[Pipeline | None] = [None]

    def _build_and_apply(params: AppParams) -> None:
        window.set_rebuild_enabled(False)

        def _progress(msg: str) -> None:
            window.set_status(msg)
            QApplication.processEvents()

        globe_params = params.globe
        if globe_params.seed is None:
            globe_params = dataclasses.replace(globe_params, seed=random.randint(0, 2**31 - 1))

        params = dataclasses.replace(params, globe=globe_params)
        current_params[0] = params
        window.set_region_params(params.region)
        window.set_local_params(params.local)
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
        window.set_realm_params(params.realm)
        window.set_world_params(params.globe)

        window.set_globe(world)
        pipeline.apply_display(window, debug=debug)

        window.set_status("")
        window.set_rebuild_enabled(True)

    def _rerun() -> None:
        p = current_params[0]
        fresh = dataclasses.replace(p, globe=dataclasses.replace(p.globe, seed=None))
        _build_and_apply(fresh)

    def _open_params() -> None:
        dialog = AppParamsDialog(current_params[0], parent=window)
        if dialog.exec() == AppParamsDialog.DialogCode.Accepted:
            new_params = dialog.result_params(current_params[0])
            if new_params == current_params[0]:
                return
            # Region and local params only re-refine the views; anything else rebuilds the world.
            old = current_params[0]
            if dataclasses.replace(new_params, region=old.region, local=old.local) == old:
                current_params[0] = new_params
                window.set_region_params(new_params.region)
                window.set_local_params(new_params.local)
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
