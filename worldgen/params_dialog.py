from __future__ import annotations

import dataclasses
from typing import Optional, Union

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from worldgen.geology.elevation_builder import ElevationParams
from worldgen.globe.builder import GlobeParams

_SpinWidget = Union[QSpinBox, QDoubleSpinBox]

_ELEVATION_GROUPS: list[tuple[str, list[tuple[str, str, type, float, float]]]] = [
    ("Mesh Generation", [
        ("icosphere_depth", "Icosphere depth", int, 1, 8),
        ("n_swaps", "Edge swaps", int, 0, 10000),
        ("n_relax", "Relaxation steps", int, 0, 100),
        ("stiffness", "Stiffness", float, 0.01, 1.0),
        ("outer_iters", "Outer iterations", int, 1, 20),
    ]),
    ("Plate Layout", [
        ("n_plates", "Number of plates", int, 2, 50),
        ("plate_compactness", "Compactness", float, 0.0, 1.0),
        ("relax_max_passes", "Relax max passes", int, 0, 50),
        ("oceanic_fraction", "Oceanic fraction", float, 0.0, 1.0),
    ]),
    ("Plate Motion", [
        ("drift_scale", "Drift scale", float, 0.0, 10.0),
        ("spin_scale", "Spin scale", float, 0.0, 10.0),
    ]),
    ("Elevation", [
        ("elevation_rate_norm", "Rate normalization", float, 0.0, 10.0),
        ("elevation_decay_rate", "Decay rate", float, 0.0, 5.0),
        ("elevation_max_hops", "Max hops", int, 1, 30),
        ("elevation_tex_size", "Texture size", int, 8, 128),
    ]),
    ("Elevation Noise", [
        ("noise_amplitude", "Amplitude (km)", float, 0.0, 20.0),
        ("noise_coast_sigma", "Coast sigma (km)", float, 0.1, 10.0),
    ]),
    ("Pixel Erosion", [
        ("pixel_thermal_iterations", "Thermal iterations", int, 0, 30),
        ("pixel_thermal_talus", "Thermal talus (km)", float, 0.001, 1.0),
        ("pixel_thermal_erosion_rate", "Erosion rate", float, 0.0, 1.0),
    ]),
    ("Rivers", [
        ("river_flow_threshold", "Flow threshold (faces)", float, 1.0, 1000.0),
    ]),
    ("Volcanism", [
        ("n_hotspots", "Hotspots", int, 0, 30),
        ("volcanic_decay_rate", "Decay rate", float, 0.0, 5.0),
        ("volcanic_max_hops", "Max hops", int, 1, 20),
        ("volcanic_elevation_scale", "Scale continental (km)", float, 0.0, 20.0),
        ("volcanic_elevation_scale_oceanic", "Scale oceanic (km)", float, 0.0, 20.0),
    ]),
]


class GlobeParamsDialog(QDialog):
    def __init__(self, params: GlobeParams, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("World Parameters")
        self.setMinimumWidth(420)
        self.resize(420, 620)

        ep = params.elevation
        self._widgets: dict[str, _SpinWidget] = {}

        outer = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(6)

        for group_name, fields in _ELEVATION_GROUPS:
            box = QGroupBox(group_name)
            form = QFormLayout(box)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            for field_name, label, ftype, fmin, fmax in fields:
                val = getattr(ep, field_name)
                w: _SpinWidget
                if ftype is int:
                    w = QSpinBox()
                    w.setRange(int(fmin), int(fmax))
                    w.setValue(int(val))
                else:
                    w = QDoubleSpinBox()
                    w.setRange(fmin, fmax)
                    w.setDecimals(3)
                    w.setSingleStep(0.05)
                    w.setValue(float(val))
                self._widgets[field_name] = w
                form.addRow(label + ":", w)
            content_layout.addWidget(box)

        seed_box = QGroupBox("Reproducibility")
        seed_form = QFormLayout(seed_box)
        seed_row = QWidget()
        seed_row_layout = QHBoxLayout(seed_row)
        seed_row_layout.setContentsMargins(0, 0, 0, 0)
        self._seed_check = QCheckBox("Fixed")
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2**31 - 1)
        if ep.seed is None:
            self._seed_check.setChecked(False)
            self._seed_spin.setEnabled(False)
        else:
            self._seed_check.setChecked(True)
            self._seed_spin.setValue(ep.seed)
        self._seed_check.toggled.connect(self._seed_spin.setEnabled)
        seed_row_layout.addWidget(self._seed_check)
        seed_row_layout.addWidget(self._seed_spin, stretch=1)
        seed_form.addRow("Seed:", seed_row)
        content_layout.addWidget(seed_box)

        content_layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def result_params(self, base: GlobeParams) -> GlobeParams:
        """Return a new GlobeParams with widget values applied over base."""
        ep = dataclasses.replace(base.elevation)
        for field_name, widget in self._widgets.items():
            field_type = type(getattr(ep, field_name))
            setattr(ep, field_name, field_type(widget.value()))
        ep.seed = self._seed_spin.value() if self._seed_check.isChecked() else None
        return dataclasses.replace(base, elevation=ep)
