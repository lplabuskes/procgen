"""ParamsDialog filtering (basic patterns, DERIVED) and the app's coupled params."""
import dataclasses
import os
from fnmatch import fnmatchcase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from main import BASIC_PARAMS, AppParams  # noqa: E402
from localmap.upsample import SHARED_WITH_REGION  # noqa: E402
from viz.params_dialog import ParamsDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _leaf_paths(instance, path=()):
    for f in dataclasses.fields(instance):
        value = getattr(instance, f.name)
        if dataclasses.is_dataclass(value):
            yield from _leaf_paths(value, path + (f.name,))
        else:
            yield ".".join(path + (f.name,))


def test_every_basic_pattern_names_a_param():
    """A renamed field would otherwise silently drop out of the default view."""
    leaves = list(_leaf_paths(AppParams()))
    for pattern in BASIC_PARAMS:
        assert any(fnmatchcase(p, pattern) for p in leaves), pattern


def test_basic_view_shows_only_matching_leaves(qapp):
    dialog = ParamsDialog(AppParams(), basic=BASIC_PARAMS)
    shown = {".".join(p) for p in dialog._leaves}
    assert "seed" in shown
    assert "globe.danger.enabled" in shown
    assert "globe.danger.troll_holds.enabled" not in shown
    assert "globe.danger.troll_holds.intensity" not in shown
    assert all(any(fnmatchcase(p, pat) for pat in BASIC_PARAMS) for p in shown)


def test_full_view_hides_derived_and_hidden(qapp):
    shown = {".".join(p) for p in ParamsDialog(AppParams())._leaves}
    assert "globe.hydrology.river_flow_threshold" in shown
    assert "globe.seed" not in shown                                 # hidden: AppParams.seed is the one
    assert "region.upsample.lapse_rate_c_per_km" not in shown        # derived from the climate
    assert "region.upsample.coast_band" in shown
    assert not any(f"local.upsample.{name}" in shown for name in SHARED_WITH_REGION)
    assert "local.upsample.slope_reference" in shown                 # the local tier's own


def test_result_keeps_unshown_values(qapp):
    params = AppParams(seed=5)
    params.globe.danger.troll_holds.intensity = 0.123
    result = ParamsDialog(params, basic=BASIC_PARAMS).result_params(params)
    assert result.globe.danger.troll_holds.intensity == pytest.approx(0.123)
    assert result.seed == 5


def test_app_params_copy_coupled_values_down():
    base = AppParams(seed=42)
    climate = dataclasses.replace(
        base.globe.climate, temperature=dataclasses.replace(base.globe.climate.temperature, lapse_rate_c_per_km=5.0))
    region = dataclasses.replace(base.region, upsample=dataclasses.replace(base.region.upsample, coast_band=0.4))
    params = dataclasses.replace(base, globe=dataclasses.replace(base.globe, climate=climate), region=region)
    assert params.globe.seed == 42
    assert params.region.upsample.lapse_rate_c_per_km == 5.0
    assert params.local.upsample.lapse_rate_c_per_km == 5.0
    assert params.local.upsample.coast_band == 0.4
    assert params.local.upsample.slope_reference != params.region.upsample.slope_reference
