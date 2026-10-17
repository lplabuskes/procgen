"""Measuring tape: great-circle distances, formatting, the click cycle, and clicks on the views."""
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from region.geometry import RegionGeometry  # noqa: E402
from viz.mapview.measure import MeasureTape, arc_points, distance_km, format_distance  # noqa: E402
from viz.mapview.regionview import RegionView  # noqa: E402
from worldgen.hydrology.discharge import EARTH_RADIUS_KM  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _unit(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon, lat = math.radians(lon_deg), math.radians(lat_deg)
    return np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])


def test_distance_is_great_circle():
    assert distance_km(_unit(0, 0), _unit(90, 0)) == pytest.approx(EARTH_RADIUS_KM * math.pi / 2)
    assert distance_km(_unit(0, 0), _unit(0, 0)) == 0.0
    # One degree of latitude is ~111 km.
    assert distance_km(_unit(10, 40), _unit(10, 41)) == pytest.approx(111.19, abs=0.01)


def test_arc_points_follow_the_great_circle():
    a, b = _unit(-30, 10), _unit(60, 50)
    arc = arc_points(a, b)
    assert np.allclose(arc[0], a) and np.allclose(arc[-1], b)
    assert np.allclose(np.linalg.norm(arc, axis=1), 1.0)
    steps = [distance_km(p, q) for p, q in zip(arc[:-1], arc[1:])]
    assert sum(steps) == pytest.approx(distance_km(a, b), rel=1e-9)


def test_format_distance():
    assert format_distance(0.0423) == "42 m"
    assert format_distance(0.9994) == "999 m"
    assert format_distance(3.456) == "3.46 km"
    assert format_distance(42.04) == "42.0 km"
    assert format_distance(12345.6) == "12,346 km"


def test_tape_click_cycle():
    tape = MeasureTape()
    a, b, c = _unit(0, 0), _unit(1, 0), _unit(2, 0)
    tape.click(a)
    assert tape.waiting and tape.ends() is None
    tape.cursor = b
    assert tape.ends() is not None
    tape.click(b)
    assert not tape.waiting and tape.cursor is None
    start, end = tape.ends()  # type: ignore[misc]
    assert np.allclose(start, a) and np.allclose(end, b)
    tape.click(c)                     # a third click starts over
    assert tape.waiting and np.allclose(tape.start, c) and tape.end is None  # type: ignore[arg-type]


def _painted(view: RegionView) -> RegionView:
    img = QImage(view.width(), view.height(), QImage.Format.Format_RGBA8888)
    view.render(img)
    return view


def test_region_view_clicks_measure_ground_distance(app):
    view = RegionView()
    view.resize(600, 400)
    geometry = RegionGeometry(40, 0.0)
    view.set_region(geometry)
    _painted(view)
    view.set_measure_mode(True)
    p0, p1 = QPointF(250.0, 150.0), QPointF(380.0, 260.0)
    for p in (p0, p1):
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p.toPoint())
    ends = view.tape.ends()
    assert ends is not None and view.tape.end is not None
    expected = [view.ecef_at_screen(p.x(), p.y()) for p in (p0.toPoint(), p1.toPoint())]
    assert expected[0] is not None and expected[1] is not None
    assert np.allclose(ends[0], expected[0]) and np.allclose(ends[1], expected[1])
    assert distance_km(*ends) > 0.0
    _painted(view)                    # draws without error
    # Leaving measure mode drops the tape; clicks outside the map place nothing.
    view.set_measure_mode(False)
    assert view.tape.start is None
    view.set_measure_mode(True)
    assert view.ecef_at_screen(2.0, 2.0) is None


def test_tape_moves_reuse_the_cached_map(app):
    view = RegionView()
    view.resize(600, 400)
    view.set_region(RegionGeometry(40, 0.0))
    _painted(view)
    view.set_measure_mode(True)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPointF(250, 150).toPoint())
    _painted(view)
    cached = view._base_cache
    assert cached is not None
    view.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove, QPointF(330, 220), QPointF(330, 220), Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
    assert view.tape.cursor is not None
    _painted(view)
    assert view._base_cache is cached          # the rubber band didn't repaint the map
    view.zoom_toward(300.0, 200.0, 2.0)
    assert view._base_cache is None            # a view change does


def test_escape_leaves_measure_mode(app):
    from viz.mapview.mapview import MapWindow
    window = MapWindow()
    window.show()
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window)
    window._measure_btn.setChecked(True)
    assert window._map._measure
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert not window._measure_btn.isChecked()
    assert not window._map._measure and not window._region_view._measure
    window.close()
