"""Color schemes for the polity stage (per-face metadata callables)."""
from __future__ import annotations

from realmgen.polity.types import ControlClass, POLITY_COLORS, UNCLAIMED_COLOR
from viz.color_stops import interp_stops_scalar
from worldgen.geology.display import OCEAN_COLOR

_TENSION_STOPS = [
    (0.0, (0.30, 0.33, 0.36, 1.0)),  # calm interior — slate
    (0.4, (0.85, 0.70, 0.25, 1.0)),  # warming — amber
    (1.0, (0.90, 0.20, 0.15, 1.0)),  # contested — red
]

_AUTHORITY_STOPS = [
    (0.0, (0.10, 0.10, 0.14, 1.0)),  # saddle / no reach — near black
    (0.35, (0.25, 0.30, 0.55, 1.0)),
    (0.70, (0.45, 0.62, 0.80, 1.0)),
    (1.0, (0.95, 0.95, 0.70, 1.0)),  # capital core — bright
]

_CONTROL_COLORS = {
    int(ControlClass.DIRECT): (0.45, 0.68, 0.40, 1.0),       # green — province
    int(ControlClass.VASSAL): (0.85, 0.65, 0.25, 1.0),       # amber — vassal seat
    int(ControlClass.INDEPENDENT): (0.84, 0.37, 0.34, 1.0),  # red — independent
}


def _is_water(meta: dict) -> bool:
    return meta.get("is_water", meta.get("is_ocean", False))


def polity_color(meta: dict) -> tuple:
    if _is_water(meta):
        return OCEAN_COLOR
    pid = meta.get("polity_id", -1)
    if pid is None or int(pid) < 0:
        return UNCLAIMED_COLOR
    return POLITY_COLORS[int(pid) % len(POLITY_COLORS)]


def sovereign_color(meta: dict) -> tuple:
    if _is_water(meta):
        return OCEAN_COLOR
    sid = meta.get("sovereign_id", -1)
    if sid is None or int(sid) < 0:
        return UNCLAIMED_COLOR
    return POLITY_COLORS[int(sid) % len(POLITY_COLORS)]


def control_color(meta: dict) -> tuple:
    if _is_water(meta):
        return OCEAN_COLOR
    cc = meta.get("control_class", -1)
    if cc is None or int(cc) < 0:
        return UNCLAIMED_COLOR
    return _CONTROL_COLORS.get(int(cc), UNCLAIMED_COLOR)


def tension_color(meta: dict) -> tuple:
    if _is_water(meta):
        return OCEAN_COLOR
    t = min(max(float(meta.get("border_tension", 0.0)), 0.0), 1.0)
    return interp_stops_scalar(t, _TENSION_STOPS)


def make_authority_color(max_authority: float):
    mx = max(float(max_authority), 1e-9)

    def fn(meta: dict) -> tuple:
        if _is_water(meta):
            return OCEAN_COLOR
        a = float(meta.get("authority", 0.0))
        return interp_stops_scalar(min(a / mx, 1.0), _AUTHORITY_STOPS)

    return fn
