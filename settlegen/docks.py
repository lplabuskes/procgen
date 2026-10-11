"""A port's docks: quays along its waterfront and piers out into the water.

:func:`build_docks` works along the footprint's edge where it meets the water,
within ``reach`` of the chosen waterfront:

- **quays**: a paved strip ``quay_width_m`` deep along that shore, on the land
  (towns and cities; a village has none);
- **piers**: rectangles straight out from the shore every ``pier_spacing_m``
  (jittered), best first by nearness to the waterfront — on the sea or a lake
  ``pier_length_m`` (scaled by size), on a river at most a fifth of its width
  and none on one narrower than ``pier_min_river_m``.  A pier must stand over
  water (its outer half wholly), clear of the other piers.  Each of the site's
  landings in reach gets a broader, longer mole first.

Counts grow with people and how much the place ships or fishes.  Qt-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from settlegen.water import SiteWater


@dataclass
class Docks:
    quays: MultiPolygon = field(default_factory=MultiPolygon)
    piers: list[Polygon] = field(default_factory=list)


@dataclass
class DockSpec:
    """The numbers :func:`build_docks` needs (from :class:`settlegen.layout.SettlementLayoutParams`)."""

    quay_width_m: float = 12.0
    pier_spacing_m: float = 60.0
    pier_length_m: float = 60.0
    pier_width_m: float = 10.0
    pier_min_river_m: float = 40.0


def dock_reach_m(population: float, village: bool) -> float:
    """How far along the shore from the waterfront the docks run."""
    if village:
        return 300.0
    return float(np.clip(200.0 * math.sqrt(max(population, 1.0) / 5000.0), 150.0, 1500.0))


def pier_count(population: float, weight: float, tier: int) -> int:
    """Piers for ``population`` people shipping or fishing at ``weight`` (tier 0–3 city, 4 town, 5 village)."""
    if tier == 5:
        return 1 if weight >= 0.3 else 0
    if tier == 4:
        return int(np.clip(round(2.0 * weight * math.sqrt(population / 2000.0)), 1, 6))
    return int(np.clip(round(3.0 * weight * math.sqrt(population / 5000.0)), 2, 40))


def _rect(p: np.ndarray, out: np.ndarray, length: float, width: float, back: float = 2.0) -> Polygon:
    side = np.array([-out[1], out[0]]) * 0.5 * width
    a, b = p - back * out, p + length * out
    return Polygon([a - side, b - side, b + side, a + side])


def _river_width(water: SiteWater, p: np.ndarray) -> float:
    """Bankfull width of the river nearest ``p`` (0 without rivers)."""
    best, width = math.inf, 0.0
    pt = Point(float(p[0]), float(p[1]))
    for line in water.lines:
        if len(line.xy) < 2:
            continue
        segs = shapely.linestrings(np.stack([line.xy[:-1], line.xy[1:]], axis=1))
        d = np.asarray(shapely.distance(segs, pt), dtype=np.float64).reshape(-1)
        k = int(np.argmin(d))
        if d[k] < best:
            best, width = float(d[k]), float(line.width_m[k])
    return width


def build_docks(
    footprint: MultiPolygon, water: SiteWater, waterfront: np.ndarray, landings: Sequence[np.ndarray],
    population: float, weight: float, tier: int, rng: np.random.Generator, spec: DockSpec,
) -> Docks:
    """See the module docstring."""
    wet = water.bodies
    if footprint.is_empty or wet.is_empty:
        return Docks()
    village = tier == 5
    reach = dock_reach_m(population, village)
    near = Point(float(waterfront[0]), float(waterfront[1])).buffer(reach)
    # A village's jetty goes from the shore nearest it.
    edge = footprint.boundary if not village else wet.boundary
    shore = edge.intersection(wet.buffer(3.0)).intersection(near) if not village else edge.intersection(near)
    parts = [g for g in shapely.get_parts(shapely.line_merge(shore) if not shore.is_empty else shore)
             if isinstance(g, LineString) and g.length > 10.0]
    if not parts:
        return Docks()
    quays = MultiPolygon()
    if not village:
        strip = shapely.union_all([g.buffer(spec.quay_width_m) for g in parts]).intersection(footprint)
        quays = MultiPolygon([g for g in shapely.get_parts(strip) if isinstance(g, Polygon) and g.area > 20.0])
    still = shapely.union_all([water.sea, water.lakes])
    scale = float(np.clip(math.sqrt(max(population, 1.0) / 20000.0), 0.5, 1.5))
    if village:
        scale = 0.4

    def pier_at(p: np.ndarray, tangent: np.ndarray, length_mult: float,  # type: ignore[no-untyped-def]
                width_mult: float):
        normal = np.array([-tangent[1], tangent[0]])
        for sign in (1.0, -1.0):
            out = sign * normal
            probe = p + 8.0 * out
            if wet.contains(Point(float(probe[0]), float(probe[1]))):
                break
        else:
            return None
        if not still.is_empty and still.distance(Point(float(p[0]), float(p[1]))) < 10.0:
            length = spec.pier_length_m * scale * length_mult
        else:
            river = _river_width(water, p)
            if river < spec.pier_min_river_m:
                return None
            length = min(0.2 * river, 30.0) * length_mult
        pier = _rect(p, out, length, spec.pier_width_m * width_mult)
        outer = _rect(p + 0.5 * length * out, out, 0.5 * length, spec.pier_width_m * width_mult, back=0.0)
        if outer.difference(wet).area > 0.02 * outer.area or pier.intersection(wet).area < 0.75 * pier.area:
            return None
        return pier

    def tangent_at(line: LineString, s: float) -> np.ndarray:
        a = np.asarray(line.interpolate(max(s - 5.0, 0.0)).coords[0])
        b = np.asarray(line.interpolate(min(s + 5.0, line.length)).coords[0])
        d = b - a
        return d / max(float(np.linalg.norm(d)), 1e-9)

    piers: list[Polygon] = []

    def add(pier) -> bool:  # type: ignore[no-untyped-def]
        if pier is None or any(pier.distance(q) < 6.0 for q in piers):
            return False
        piers.append(pier)
        return True

    # Moles at the landings first.
    for la in landings:
        la = np.asarray(la, dtype=np.float64)
        if not near.contains(Point(float(la[0]), float(la[1]))):
            continue
        line = min(parts, key=lambda g: g.distance(Point(float(la[0]), float(la[1]))))
        s = line.project(Point(float(la[0]), float(la[1])))
        add(pier_at(np.asarray(line.interpolate(s).coords[0]), tangent_at(line, s), 1.4, 1.8))
    n = pier_count(population, weight, tier)
    slots = []
    for line in parts:
        s = float(rng.uniform(0.2, 0.8)) * spec.pier_spacing_m
        while s < line.length:
            slots.append((line, s))
            s += spec.pier_spacing_m * float(rng.uniform(0.75, 1.25))
    wf = Point(float(waterfront[0]), float(waterfront[1]))
    slots.sort(key=lambda ls: ls[0].interpolate(ls[1]).distance(wf))
    for line, s in slots:
        if len(piers) >= n + len(landings):
            break
        add(pier_at(np.asarray(line.interpolate(s).coords[0]), tangent_at(line, s),
                    float(rng.uniform(0.7, 1.3)), 1.0))
    return Docks(quays=quays, piers=piers)
