"""Local maps in the viewer: picking a hex in a region, the local view, its
cache, settlement hover, and the window's local → region → world navigation."""
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
from realmgen.test.conftest import _settlement  # noqa: E402
from realmgen.towns.types import TOWN_MARKET, TOWN_RIVER, Towns  # noqa: E402
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


def _local_view(region: RegionView) -> RegionView:
    """A local view sharing ``region``'s store, as the window's views do."""
    view = RegionView()
    view.resize(600, 500)
    view.set_store(region.store)
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
    view = _local_view(region)
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
    view = _local_view(region)
    view.set_local(region, h)
    first = view.local_map
    view.set_local(region, h)
    assert view.local_map is first
    view.set_local_params(dataclasses.replace(_LOCAL, villages=dataclasses.replace(_LOCAL.villages, spacing_km=10.0)))
    view.set_local(region, h)
    assert view.local_map is not first
    # Re-entering the region takes the store's maps: nothing is rebuilt.
    second, fields, builds = view.local_map, region.fields, len(region.store.builds)
    region.set_region(RegionGeometry(REGION_ID, globe.antimeridian), globe)
    view.set_local(region, h)
    assert region.fields is fields and view.local_map is second
    assert len(region.store.builds) == builds
    # New region params drop the local maps built from the old region.
    region.set_params(dataclasses.replace(_REGION, rivers=dataclasses.replace(_REGION.rivers, valley_seeking=1.0)))
    view.set_local(region, h)
    assert view.local_map is not second


def test_hovering_a_village_describes_it(app, globe):  # noqa: F811
    region = _region_view(globe)
    view = _local_view(region)
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


@pytest.fixture
def settled(globe):  # noqa: F811
    """Three towns and a city on land texels of the region's landiest hex,
    installed for one test; yields that hex."""
    probe = _region_view(globe)
    fields = probe.fields
    assert fields is not None
    h = _land_hex(probe)
    st = fields.grid.texture_local(fields.tex_size)[h]
    inside = ((st > 0.02) & (st < 0.98)).all(axis=-1)            # not over-hanging the cell
    land = (fields.textures["ocean_mask"][h, ..., 0] < 0.5) & fields.in_hex[h] & inside
    rows, cols = np.nonzero(land)
    pick = np.linspace(0, len(rows) - 1, 4).astype(int)
    pos = fields.grid.texture_positions(fields.tex_size)[h, rows[pick], cols[pick]]
    faces = fields.faces[h, rows[pick], cols[pick]].astype(np.int32)
    towns = Towns(
        positions=pos[:3].copy(), population=np.array([4_000.0, 3_000.0, 2_000.0]),
        face_id=faces[:3], parent=np.full(3, -1, dtype=np.int32),
        kind=np.array([TOWN_MARKET, TOWN_RIVER, TOWN_MARKET], dtype=np.int8),
    )
    city = _settlement(face_id=int(faces[3]), pop=50_000.0, position=pos[3].copy())
    globe.add_feature("towns", towns)
    globe.add_feature("settlements", [city])
    yield h
    globe.remove_feature("towns")
    globe.remove_feature("settlements")


def _show_settlements(view: RegionView) -> RegionView:
    view.set_overlay_visible("settlements", True)
    view.set_overlay_visible("towns", True)
    return _painted(view)


def _hover_at(view: RegionView, ecef) -> None:
    s, t, _ = view._geometry.ecef_to_local(np.asarray(ecef).reshape(1, 3))  # type: ignore[union-attr]
    view._mouse = _xy(_screen(view, float(s[0]), float(t[0])))


def test_region_hover_names_the_city_and_counts_the_hexs_towns(app, globe, settled):  # noqa: F811
    view = _show_settlements(_region_view(globe))
    city = globe.get_feature("settlements")[0]
    _hover_at(view, view._on_land("settlements", city.position[None]))
    text = view.city_under_cursor(None)
    assert text is not None and "people" in text and "50,000" in text
    view._mouse = (-100.0, -100.0)
    assert view.city_under_cursor(None) is None
    # The hexagon's towns, wherever the cursor is in it; none elsewhere.
    drawn = view._on_land("towns", globe.get_feature("towns").positions)
    s, t, _ = view._geometry.ecef_to_local(drawn)  # type: ignore[union-attr]
    hexes, _, _, inside = view.fields.grid.texels_at(s, t, view.fields.tex_size)  # type: ignore[union-attr]
    assert inside.any()
    h = int(np.bincount(hexes[inside]).argmax())     # refinement may move them off ``settled``
    n = int((inside & (hexes == h)).sum())
    count, kinds = view.towns_in_hex(h)
    assert count.startswith(f"Towns in hex: {n} (") and ("market" in kinds or "river" in kinds)
    other = next(i for i in range(len(view.hex_local)) if not (inside & (hexes == i)).any())
    assert view.towns_in_hex(other) == []


def test_local_hover_names_the_town_under_the_cursor(app, globe, settled):  # noqa: F811
    region = _region_view(globe)
    view = _local_view(region)
    view.set_local(region, settled)
    _show_settlements(view)
    towns = view._refined_features().get("towns")
    assert towns is not None and len(towns) > 0
    _hover_at(view, view._on_land("towns", towns.positions)[0])
    text = view.town_under_cursor()
    assert text is not None and "town" in text and "people" in text
    assert any(text == part for part in view.settlements_under_cursor(None))
    view._mouse = (-100.0, -100.0)
    assert view.town_under_cursor() is None
    # Towns are counted per hexagon only in a region.
    assert view.towns_in_hex(0) == []


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


def test_farmland_follows_its_settlements_overlays(app, globe):  # noqa: F811
    region = _region_view(globe)
    view = _local_view(region)
    view.set_local(region, _land_hex(region))
    _painted(view)
    local = view.local_map
    assert local is not None and local.farmland is not None
    assert (local.farmland.kind == 0).any()
    # Village fields show with the villages (on by default)…
    assert len(view._farm_images) == local.fields.n_hexes
    for name in ("villages", "towns", "settlements"):
        view.set_overlay_visible(name, False)
    assert view._farm_images == []
    # …and come back with them.
    view.set_overlay_visible("villages", True)
    assert len(view._farm_images) == local.fields.n_hexes
    _painted(view)


def test_region_and_local_hover_name_the_fort_and_its_garrison(app, globe, settled):  # noqa: F811
    from realmgen.polity.forts import Forts
    probe = _region_view(globe)
    fields = probe.fields
    assert fields is not None
    pos = fields.grid.texture_positions(fields.tex_size)[settled]
    land = fields.textures["ocean_mask"][settled, ..., 0] < 0.5
    rows, cols = np.nonzero(land & fields.in_hex[settled])
    at = pos[rows[len(rows) // 2], cols[len(cols) // 2]][None].copy()
    forts = Forts(
        positions=at, face_id=np.array([int(fields.faces[settled, rows[0], cols[0]])], dtype=np.int32),
        facing=np.array([0], dtype=np.int32), kind=np.array([0], dtype=np.int8),
        score=np.array([0.5]), garrison=np.array([321.0]),
    )
    globe.add_feature("forts", forts)
    try:
        region = _region_view(globe)
        local = _local_view(region)
        local.set_local(region, settled)
        for view in (region, local):
            view.set_overlay_visible("forts", True)
            _painted(view)
            _hover_at(view, view._on_land("forts", forts.positions)[0])
            assert view.fort_under_cursor() == "Fort (321 men)"
            assert "Fort (321 men)" in view.settlements_under_cursor(None)
            view._mouse = (-100.0, -100.0)
            assert view.fort_under_cursor() is None
    finally:
        globe.remove_feature("forts")
