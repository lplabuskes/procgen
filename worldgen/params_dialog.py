from __future__ import annotations

import dataclasses
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from worldgen.globe.builder import GlobeParams


class _CollapsibleSection(QWidget):
    """A titled section whose content can be hidden via an arrow toggle."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow)
        self._toggle.clicked.connect(self._on_toggled)

        self._content = QFrame()
        self._content.setFrameShape(QFrame.Shape.StyledPanel)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(12, 6, 6, 6)
        self._content_layout.setSpacing(4)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        outer.addWidget(self._toggle)
        outer.addWidget(self._content)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(checked)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout


def _humanize(field_name: str) -> str:
    return field_name.replace("_", " ").capitalize()


class GlobeParamsDialog(QDialog):
    """Dialog that walks a (possibly nested) dataclass tree and builds collapsible UI.

    Each dataclass-typed field becomes a collapsible section. Leaf fields are read
    from their `metadata` dict: `label`, `min`, `max`, optional `step`. Setting
    `hidden=True` skips the field/section. `kind="optional_int"` adds a "Fixed"
    checkbox so the value can be None.
    """

    def __init__(self, params: GlobeParams, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("World Parameters")
        self.setMinimumWidth(420)
        self.resize(420, 620)

        # Map from path-tuple (e.g. ("elevation", "mesh", "icosphere_depth")) → widget/state.
        self._leaves: dict[tuple[str, ...], _LeafBinding] = {}

        outer = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(6)

        # Walk the top-level GlobeParams: each top field is itself a dataclass
        # section (currently just `elevation`). We render it inline rather than
        # nested under an extra "GlobeParams" header.
        for f in dataclasses.fields(params):
            if f.metadata.get("hidden"):
                continue
            value = getattr(params, f.name)
            if dataclasses.is_dataclass(value):
                section = _CollapsibleSection(f.metadata.get("label") or _humanize(f.name))
                self._build_dataclass(value, (f.name,), section.content_layout())
                content_layout.addWidget(section)
            else:
                form = QFormLayout()
                self._add_leaf(form, f, value, (f.name,))
                wrapper = QWidget()
                wrapper.setLayout(form)
                content_layout.addWidget(wrapper)

        content_layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _build_dataclass(
        self,
        instance: Any,
        path: tuple[str, ...],
        parent_layout: QVBoxLayout,
    ) -> None:
        """Recursively build UI for a dataclass instance into parent_layout."""
        # Separate leaves vs nested dataclasses so leaves group into a single form.
        form: Optional[QFormLayout] = None
        for f in dataclasses.fields(instance):
            if f.metadata.get("hidden"):
                continue
            value = getattr(instance, f.name)
            sub_path = path + (f.name,)
            if dataclasses.is_dataclass(value):
                form = None  # Break the current form group.
                section = _CollapsibleSection(f.metadata.get("label") or _humanize(f.name))
                self._build_dataclass(value, sub_path, section.content_layout())
                parent_layout.addWidget(section)
            else:
                if form is None:
                    form = QFormLayout()
                    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
                    wrapper = QWidget()
                    wrapper.setLayout(form)
                    parent_layout.addWidget(wrapper)
                self._add_leaf(form, f, value, sub_path)

    def _add_leaf(
        self,
        form: QFormLayout,
        f: dataclasses.Field,
        value: Any,
        path: tuple[str, ...],
    ) -> None:
        meta = f.metadata
        label = meta.get("label") or _humanize(f.name)
        kind = meta.get("kind")
        fmin = meta.get("min", 0)
        fmax = meta.get("max", 10_000)
        step = meta.get("step")

        if kind == "optional_int":
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            check = QCheckBox("Fixed")
            spin = QSpinBox()
            spin.setRange(int(fmin), int(fmax))
            if value is None:
                check.setChecked(False)
                spin.setEnabled(False)
            else:
                check.setChecked(True)
                spin.setValue(int(value))
            check.toggled.connect(spin.setEnabled)
            row_layout.addWidget(check)
            row_layout.addWidget(spin, stretch=1)
            form.addRow(label + ":", row)
            self._leaves[path] = _LeafBinding(kind="optional_int", widget=spin, check=check)
            return

        # Plain int/float field.
        if isinstance(value, bool):
            # Bool would slip through isinstance(value, int); explicit case for safety.
            check = QCheckBox()
            check.setChecked(bool(value))
            form.addRow(label + ":", check)
            self._leaves[path] = _LeafBinding(kind="bool", widget=check)
            return

        widget: QSpinBox | QDoubleSpinBox
        if isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(int(fmin), int(fmax))
            if step is not None:
                widget.setSingleStep(int(step))
            widget.setValue(int(value))
            self._leaves[path] = _LeafBinding(kind="int", widget=widget)
        else:
            widget = QDoubleSpinBox()
            widget.setRange(float(fmin), float(fmax))
            widget.setDecimals(3)
            widget.setSingleStep(float(step) if step is not None else 0.05)
            widget.setValue(float(value))
            self._leaves[path] = _LeafBinding(kind="float", widget=widget)

        form.addRow(label + ":", widget)

    def result_params(self, base: GlobeParams) -> GlobeParams:
        """Return a new GlobeParams with widget values applied over `base`."""
        return self._apply(base, ())

    def _apply(self, instance: Any, path: tuple[str, ...]) -> Any:
        """Return a new dataclass instance with leaf widgets applied recursively."""
        updates: dict[str, Any] = {}
        for f in dataclasses.fields(instance):
            sub_path = path + (f.name,)
            value = getattr(instance, f.name)
            if dataclasses.is_dataclass(value):
                updates[f.name] = self._apply(value, sub_path)
                continue
            binding = self._leaves.get(sub_path)
            if binding is None:
                # Hidden or otherwise not exposed — keep current value.
                continue
            updates[f.name] = binding.read()
        return dataclasses.replace(instance, **updates)


@dataclasses.dataclass
class _LeafBinding:
    kind: str  # "int" | "float" | "bool" | "optional_int"
    widget: Any
    check: Optional[QCheckBox] = None

    def read(self) -> Any:
        if self.kind == "optional_int":
            assert self.check is not None
            return self.widget.value() if self.check.isChecked() else None
        if self.kind == "bool":
            return self.widget.isChecked()
        if self.kind == "int":
            return int(self.widget.value())
        return float(self.widget.value())
