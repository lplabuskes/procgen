"""Local maps in the viewer: picking a hex in a region, the local view, its
cache, village hover, and the window's local → region → world navigation."""
import dataclasses
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from localmap.params import LocalParams  # noqa: E402
from localmap.test.conftest import local_params  # noqa: E402
from localmap.villages import VillageParams  # noqa: E402
from region.geometry import RegionGeometry  # noqa: E402
from region.params import RegionParams  # noqa: E402
from region.test.conftest import synthetic_globe  # noqa: E402, F401  (fixture)
from region.test.test_resources import globe  # noqa: E402, F401  (fixture)
from viz.mapview.mapview import MapWindow  # noqa: E402
from viz.mapview.regionview import RegionView  # noqa: E402

REGION_ID = 40
# Test-sized tiers (see localmap/test): coarse grids, villages scaled to match.
_REGION = dataclasses.replace(
    RegionParams(), upsample=dataclasses.replace(RegionParams().upsample, n_cols=6, tex_size=16)
)
_LOCAL = LocalParams(
    upsample=local_params(),
    villages=VillageParams(village_min=20_000.0, village_max=100_000.0, spacing_km=15.0),
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _painted(view: RegionView) -> RegionView:
    img = QImage(view.width(), view.height(), QImage.Format.Format_RGBA8888)
    view.render(img)
    return view


def _region_view(globe_) -> RegionView:
    view = RegionView()
    view.resize(600, 500)
    view.set_params(_REGION)
    view.set_region(RegionGeometry(REGION_ID, globe_.antimeridian), globe_)
    return _painted(view)


def _local_view() -> RegionView:
    view = RegionView()
    view.resize(600, 500)
    view.set_local_params(_LOCAL)
    return view


def _screen(view: RegionView, s: float, t: float) -> QPoint:
    """The screen pixel a local ``(s, t)`` is painted at."""
    assert view._cell_layout is not None
    x0, y0, size, transform = view._cell_layout
    p = transform.map(QPointF(x0 + s * size, y0 + (1.0 - t) * size))
    return QPoint(round(p.x()), round(p.y()))


def _land_hex(view: RegionView) -> int:
    fields = view.fields
    assert fields is not None
    land = (fields.textures["ocean_mask"][..., 0] < 0.5) & fields.in_hex
    return int(np.argmax(land.sum(axis=(1, 2))))


def test_a_click_picks_a_hex_but_a_drag_does_not(app, globe):  # noqa: F811
    view = _region_view(globe)
    picked: list[int] = []
    view.set_hex_click_callback(picked.append)
    h = _land_hex(view)
    at = _screen(view, *view.hex_local[h])

    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=at)
    assert picked == []                         # picking is off

    view.set_local_pick(True)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=at)
    assert picked == [h]

    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=at)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=at + QPoint(40, 0))
    assert picked == [h]


def test_the_local_view_shows_the_hexs_local_map(app, globe):  # noqa: F811
    region = _region_view(globe)
    h = _land_hex(region)
    view = _local_view()
    view.set_local(region, h)
    _painted(view)
    local = view.local_map
    assert local is not None and local.hex_index == h
    assert view.region_id == REGION_ID
    assert "Local map" in view.status_text() and "villages" in view.status_text()
    # The countryside drawn is net of the villages.
    fields = view.fields
    assert fields is not None
    np.testing.assert_array_equal(fields.textures["rural_density"], local.villages.rural_density)
    assert "villages" in view._refined_features()
    # The map is the parent hexagon: its square's corners are not part of it.
    assert view.texel_at_screen(*_xy(_screen(view, 0.5, 0.5))) is not None
    assert view.texel_at_screen(*_xy(_screen(view, 0.02, 0.98))) is None


def _xy(p: QPoint) -> tuple[float, float]:
    return float(p.x()), float(p.y())


def test_local_maps_are_cached_until_the_params_change(app, globe):  # noqa: F811
    region = _region_view(globe)
    h = _land_hex(region)
    view = _local_view()
    view.set_local(region, h)
    first = view.local_map
    view.set_local(region, h)
    assert view.local_map is first
    view.set_local_params(dataclasses.replace(_LOCAL, villages=dataclasses.replace(_LOCAL.villages, spacing_km=10.0)))
    view.set_local(region, h)
    assert view.local_map is not first
    # A re-refined region invalidates everything built from the old one.
    second = view.local_map
    region.set_region(RegionGeometry(REGION_ID, globe.antimeridian), globe)
    view.set_local(region, h)
    assert view.local_map is not second


def test_hovering_a_village_describes_it(app, globe):  # noqa: F811
    region = _region_view(globe)
    view = _local_view()
    view.set_local(region, _land_hex(region))
    _painted(view)
    local = view.local_map
    assert local is not None and len(local.villages) > 0
    s, t, _ = view._geometry.ecef_to_local(local.villages.positions[:1])  # type: ignore[union-attr]
    view._mouse = _xy(_screen(view, float(s[0]), float(t[0])))
    text = view.village_under_cursor()
    assert text is not None and "village" in text and "people" in text
    view._mouse = (-100.0, -100.0)
    assert view.village_under_cursor() is None


def test_the_window_steps_back_one_tier_at_a_time(app, globe):  # noqa: F811
    window = MapWindow()
    window.set_region_params(_REGION)
    window.set_local_params(_LOCAL)
    window._region_view.set_region(RegionGeometry(REGION_ID, globe.antimeridian), globe)
    window._stack.setCurrentWidget(window._region_view)

    window._enter_local(_land_hex(window._region_view))
    assert window._stack.currentWidget() is window._local_view
    assert window._back_btn.text() == "← Region"
    assert "villages" in window._overlay_actions

    window._go_back()
    assert window._stack.currentWidget() is window._region_view
    assert window._region_view.fields is not None          # the region is kept as it was
    window._go_back()
    assert window._stack.currentWidget() is window._map
