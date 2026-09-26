"""RegionView zoom/pan: the cursor anchor, clamping, and reset on a new region."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from region.geometry import RegionGeometry  # noqa: E402
from viz.mapview.mapview import MapWindow  # noqa: E402
from viz.mapview.regionview import RegionView, _RegionProjector  # noqa: E402


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


def test_region_projector_matches_the_painted_cell(app):
    """Sphere points land where the cell's own local frame is drawn."""
    view = _view(45)
    assert view._geometry is not None and view._cell_layout is not None
    x0, y0, size, transform = view._cell_layout
    proj = _RegionProjector(view._geometry, x0, y0, size, transform, 1.0)
    ss = np.array([0.1, 0.5, 0.9])
    tt = np.array([0.2, 0.5, 0.7])
    sx, sy, keep = proj.points(view._geometry.local_to_ecef(ss, tt))
    assert keep.all()
    for s, t, x, y in zip(ss, tt, sx, sy):
        assert _local_under(view, float(x), float(y)) == pytest.approx((s, t), abs=1e-6)


def test_region_projector_culls_far_and_distant_features(app):
    view = _view(45)
    assert view._geometry is not None and view._cell_layout is not None
    proj = _RegionProjector(view._geometry, *view._cell_layout, 1.0)
    centre = view._geometry.center_ecef()
    distant = view._geometry.local_to_ecef(np.array([3.0]), np.array([0.5]))[0]
    _, _, keep = proj.points(np.stack([centre, -centre, distant]))
    assert keep.tolist() == [True, False, False]
    # A segment is drawn when either end is near the cell, never across the far side.
    *_, seg_keep = proj.segments(np.stack([centre, centre]), np.stack([distant, -centre]))
    assert seg_keep.tolist() == [True, False]


def test_overlay_toggle_drives_both_views(app):
    window = MapWindow()
    window.add_overlay("rivers", "Rivers", checked=True)
    assert window._map._overlay_visible["rivers"] is True
    assert window._region_view._overlay_visible["rivers"] is True
    window._overlay_actions["rivers"].setChecked(False)
    assert window._map._overlay_visible["rivers"] is False
    assert window._region_view._overlay_visible["rivers"] is False


class _StubTowns:
    positions = np.array([[1.0, 0.0, 0.0]])
    population = np.array([2_000.0])
    kind = np.array([0], dtype=np.int8)

    def __len__(self):
        return 1


class _StubGlobe:
    def get_feature(self, name):
        return _StubTowns() if name == "towns" else None


class _CountingProjector:
    zoom = 50.0

    def __init__(self, regional):
        self.regional = regional
        self.on_land_calls = 0

    def on_land(self, key, ecef):
        self.on_land_calls += 1
        return ecef

    def points(self, ecef):
        n = len(ecef)
        return np.zeros(n), np.zeros(n), np.ones(n, dtype=bool)


def test_towns_draw_only_in_the_region_view(app):
    from PySide6.QtGui import QImage, QPainter
    from viz.mapview.overlays import _draw_towns
    image = QImage(10, 10, QImage.Format.Format_RGBA8888)
    painter = QPainter(image)
    world_proj, region_proj = _CountingProjector(False), _CountingProjector(True)
    _draw_towns(painter, _StubGlobe(), world_proj)      # type: ignore[arg-type]
    _draw_towns(painter, _StubGlobe(), region_proj)     # type: ignore[arg-type]
    painter.end()
    assert world_proj.on_land_calls == 0 and region_proj.on_land_calls == 1
