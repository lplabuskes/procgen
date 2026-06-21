import dataclasses
import random
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFileDialog, QWidget

from core.pipeline import BuildContext
from core.world import World
from viz.mapview.mapview import MapWindow
from viz.params_dialog import ParamsDialog
from realmgen.builder import RealmParams, RealmPipeline
from worldgen import GlobePipeline, GlobeParams, load_globe, save_globe


@dataclasses.dataclass
class AppParams:
    globe: GlobeParams = dataclasses.field(
        default_factory=GlobeParams, metadata={"label": "World"}
    )
    realm: RealmParams = dataclasses.field(
        default_factory=RealmParams, metadata={"label": "Realm / Settlements"}
    )


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
    current_globe_pipeline: list[GlobePipeline | None] = [None]

    def _build_and_apply(params: AppParams) -> None:
        window.set_rebuild_enabled(False)

        def _progress(msg: str) -> None:
            window.set_status(msg)
            QApplication.processEvents()

        globe_params = params.globe
        if globe_params.seed is None:
            globe_params = dataclasses.replace(globe_params, seed=random.randint(0, 2**31 - 1))

        current_params[0] = dataclasses.replace(params, globe=globe_params)

        world = World(globe_params.mesh, seed=globe_params.seed, progress=_progress)
        ctx = BuildContext(world=world, progress=_progress)
        globe_pipeline = GlobePipeline(globe_params)
        globe_pipeline.build(ctx)
        current_globe_pipeline[0] = globe_pipeline

        realm_pipeline = RealmPipeline(params.realm)
        realm_pipeline.build(ctx)

        prev_scheme = window.active_scheme
        window.set_globe(ctx.world)
        globe_pipeline.apply_display(window, debug=debug)
        realm_pipeline.apply_display(window, debug=debug)
        if prev_scheme:
            window.set_active_scheme(prev_scheme)

        window.set_status("")
        window.set_rebuild_enabled(True)

    def _load_from_file(path: str) -> None:
        window.set_rebuild_enabled(False)
        window.set_status(f"Loading {Path(path).name}…")
        QApplication.processEvents()

        globe_pipeline, globe = load_globe(path)
        current_params[0] = AppParams(globe=globe_pipeline.params)
        current_globe_pipeline[0] = globe_pipeline

        window.set_globe(globe)
        globe_pipeline.apply_display(window, debug=debug)

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
            if new_params != current_params[0]:
                _build_and_apply(new_params)

    def _save() -> None:
        globe_pipeline = current_globe_pipeline[0]
        if globe_pipeline is None:
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
        save_globe(globe_pipeline, path)
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
