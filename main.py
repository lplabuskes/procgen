import dataclasses
import random
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFileDialog, QWidget

from viz.mapview.mapview import MapWindow
from viz.params_dialog import ParamsDialog
from worldgen.globe import GlobeBuilder, GlobeParams
from worldgen.globe.serialization import load_globe, save_globe


class GlobeParamsDialog(ParamsDialog):
    def __init__(self, params: GlobeParams, parent: "QWidget | None" = None) -> None:
        super().__init__(params, title="World Parameters", parent=parent)

    def result_params(self, base: GlobeParams) -> GlobeParams:  # type: ignore[override]
        return super().result_params(base)


def main() -> None:
    debug = "--debug" in sys.argv
    file_arg = next((a for a in sys.argv[1:] if not a.startswith("-")), None)

    app = QApplication(sys.argv)

    window = MapWindow()
    window.setWindowTitle("Procedural World")

    current_params: list[GlobeParams] = [GlobeParams()]
    current_builder: list[GlobeBuilder | None] = [None]

    def _build_and_apply(params: GlobeParams) -> None:
        window.set_rebuild_enabled(False)

        def _progress(msg: str) -> None:
            window.set_status(msg)
            QApplication.processEvents()

        if params.elevation.seed is None:
            ep = dataclasses.replace(params.elevation, seed=random.randint(0, 2**31 - 1))
            params = dataclasses.replace(params, elevation=ep)

        current_params[0] = params

        builder = GlobeBuilder(params)
        globe = builder.build(progress=_progress)
        current_builder[0] = builder

        window.set_globe(globe)
        builder.apply_display(window, debug=debug)

        window.set_status("")
        window.set_rebuild_enabled(True)

    def _load_from_file(path: str) -> None:
        window.set_rebuild_enabled(False)
        window.set_status(f"Loading {Path(path).name}…")
        QApplication.processEvents()

        builder, globe = load_globe(path)
        current_params[0] = builder.params
        current_builder[0] = builder

        window.set_globe(globe)
        builder.apply_display(window, debug=debug)

        window.set_status("")
        window.set_rebuild_enabled(True)

    def _rerun() -> None:
        fresh = dataclasses.replace(current_params[0], elevation=dataclasses.replace(
            current_params[0].elevation, seed=None
        ))
        _build_and_apply(fresh)

    def _open_params() -> None:
        dialog = GlobeParamsDialog(current_params[0], parent=window)
        if dialog.exec() == GlobeParamsDialog.DialogCode.Accepted:
            new_params = dialog.result_params(current_params[0])
            if new_params != current_params[0]:
                _build_and_apply(new_params)

    def _save() -> None:
        builder = current_builder[0]
        if builder is None:
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
        save_globe(builder, path)
        window.set_status("")

    window.set_save_callback(_save)
    window.set_rebuild_callback(_rerun)
    window.set_params_callback(_open_params)

    if file_arg:
        _load_from_file(file_arg)
    else:
        _build_and_apply(GlobeParams())

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
