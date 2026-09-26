"""RegionView zoom/pan: the cursor anchor, clamping, and reset on a new region."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from region.geometry import RegionGeometry  # noqa: E402
from viz.mapview.regionview import RegionView  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _painted(view: RegionView) -> RegionView:
    """Run a paint so the view lays out its cell (offscreen, via ``render``)."""
    img = QImage(view.width(), view.height(), QImage.Format.Format_RGBA8888)
    view.render(img)
    return view


def _view(region_id: int = 40) -> RegionView:
    view = RegionView()
    view.resize(600, 400)
    view.set_region(RegionGeometry(region_id, 0.0))
    return _painted(view)


def _local_under(view: RegionView, mx: float, my: float) -> tuple[float, float]:
    """The cell's local ``(s, t)`` drawn at a screen point."""
    assert view._cell_layout is not None
    x0, y0, size, transform = view._cell_layout
    inverse, ok = transform.inverted()
    assert ok
    p = inverse.map(QPointF(mx, my))
    return (p.x() - x0) / size, 1.0 - (p.y() - y0) / size


def test_zoom_keeps_cursor_point_fixed(app):
    view = _view()
    before = _local_under(view, 350.0, 150.0)
    view.zoom_toward(350.0, 150.0, 4.0)
    _painted(view)
    assert view.zoom == pytest.approx(4.0)
    assert _local_under(view, 350.0, 150.0) == pytest.approx(before, abs=1e-9)


def test_zoom_is_clamped(app):
    view = _view()
    view.zoom_toward(300.0, 200.0, 1e6)
    assert view.zoom == RegionView.MAX_ZOOM
    view.zoom_toward(300.0, 200.0, 1e-6)
    assert view.zoom == 1.0
    assert tuple(view._pan) == (0.0, 0.0)


def test_pan_cannot_leave_the_cell(app):
    view = _view()
    view.zoom_toward(300.0, 200.0, 2.0)
    view._pan[:] = (5.0, -5.0)
    view._clamp_pan()
    assert tuple(view._pan) == pytest.approx((0.5, -0.5))


def test_new_region_resets_view(app):
    view = _view()
    view.zoom_toward(300.0, 200.0, 3.0)
    view.set_region(RegionGeometry(41, 0.0))
    assert view.zoom == 1.0
    assert tuple(view._pan) == (0.0, 0.0)
