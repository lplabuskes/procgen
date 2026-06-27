"""Color schemes for the polity stage (per-face metadata callables)."""
from __future__ import annotations

import colorsys
from typing import TYPE_CHECKING, List

from realmgen.polity.types import ControlClass, POLITY_COLORS, UNCLAIMED_COLOR
from viz.color_stops import interp_stops_scalar
from worldgen.geology.display import OCEAN_COLOR, is_water_meta as _is_water

if TYPE_CHECKING:
    from realmgen.polity.types import Polity

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


def make_polity_tree_color(polities: List["Polity"]):
    """Return a color scheme that tints vassals away from their sovereign's base color.

    Sovereign territories render in the pure palette color.  Each vassal gets a
    hue jitter (±20°) and a lightness boost derived from its polity_id, so all
    subjects of the same sovereign are clearly related but individually distinct.
    """
    depth_of: dict[int, int] = {pol.polity_id: pol.depth for pol in polities}

    # Pre-convert each sovereign's palette color to HLS once.
    sov_hls: dict[int, tuple] = {}
    for pol in polities:
        if pol.liege is None:
            r, g, b, _ = POLITY_COLORS[pol.polity_id % len(POLITY_COLORS)]
            sov_hls[pol.polity_id] = colorsys.rgb_to_hls(r, g, b)

    def fn(meta: dict) -> tuple:
        if _is_water(meta):
            return OCEAN_COLOR
        sid_raw = meta.get("sovereign_id", -1)
        if sid_raw is None or int(sid_raw) < 0:
            return UNCLAIMED_COLOR
        sid = int(sid_raw)
        pid_raw = meta.get("polity_id", sid_raw)
        pid = int(pid_raw) if pid_raw is not None else sid

        h, lit, s = sov_hls.get(sid, colorsys.rgb_to_hls(0.5, 0.5, 0.5))
        depth = depth_of.get(pid, 0)

        if depth > 0:
            # Hue jitter ±10° — stable, unique per polity_id.
            t = ((pid * 2654435761) & 0xFFFF) / 0xFFFF  # 0..1
            h = (h + (t - 0.5) * (20.0 / 360.0)) % 1.0
            # Lighten toward white with depth so vassals are visually lighter.
            lit = min(lit + 0.08 + depth * 0.05, 0.82)

        r, g, b = colorsys.hls_to_rgb(h, lit, s)
        return (r, g, b, 1.0)

    return fn


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
