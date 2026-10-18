"""Landings: where a water leg pinned at a port first reaches water, a quay on its sea, lake or river."""
from __future__ import annotations

from typing import Literal

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point
from shapely.ops import nearest_points

from region.roads import RegionRoads
from region.upsample import EARTH_RADIUS_KM
from settlegen.frame import Frame
from settlegen.site.crossings import _points
from settlegen.site.types import Landing, RoadLine, SiteWater


def _split_at_coast(parent: RegionRoads, reach_m: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``parent``'s polylines and flows, split at any inner vertex that is a sea
    leg's coast end (two port hops from one city share their coast point: the
    region chains them city → coast → city)."""
    polylines = [np.asarray(p, dtype=np.float64) for p in parent.polylines]
    flows = [np.asarray(f, dtype=np.float64) for f in parent.polyline_flows]
    for a, _, _ in parent.sea:
        a = np.asarray(a, dtype=np.float64)
        best, best_d = None, reach_m
        for i, p in enumerate(polylines):
            d = np.linalg.norm(p - a, axis=1) * EARTH_RADIUS_KM * 1000.0
            v = int(np.argmin(d))
            if d[v] < best_d:
                best, best_d = (i, v), float(d[v])
        if best is None or best[1] in (0, len(polylines[best[0]]) - 1):
            continue
        i, v = best
        p, f = polylines.pop(i), flows.pop(i)
        polylines += [p[:v + 1], p[v:]]
        flows += [f[:v], f[v:]]
    # The two halves of such a loop are one road.
    keep: list[int] = []
    for i, p in enumerate(polylines):
        twin = next((j for j in keep if len(polylines[j]) == len(p) and (
            np.allclose(polylines[j], p) or np.allclose(polylines[j], p[::-1]))), None)
        if twin is None:
            keep.append(i)
        else:
            flows[twin] = flows[twin] + (flows[i] if np.allclose(polylines[twin], p) else flows[i][::-1])
    return [polylines[i] for i in keep], [flows[i] for i in keep]


def _landings(
    parent: RegionRoads, water: SiteWater, frame: Frame, reach_m: float, merge_m: float,
) -> tuple[list[Landing], list[tuple[np.ndarray, np.ndarray, float]], list[np.ndarray]]:
    """``(landings, sea legs in frame metres, parent polylines)``.

    A sea leg's coast end is the region's (its raster is ~9 km): the road that
    reached it (the local map snapped it onto its land; the road end nearest it,
    within ``reach_m``) lands where it passes nearest the shore in the frame, and
    is cut there.  A leg with no road lands at the shore nearest its coast end.
    """
    polylines, flows = _split_at_coast(parent, reach_m)
    parent.polyline_flows = flows
    # Matched before any road is cut at a landing: several legs can share a road's coast end.
    ends = np.array([[p[0], p[-1]] for p in polylines]).reshape(-1, 2, 3)
    legs: list[tuple[np.ndarray, np.ndarray, float]] = []
    landings: list[Landing] = []
    shore = water.shorelines
    frame_box = frame.box()
    landed: dict[int, np.ndarray] = {}            # polyline → its landing, once cut
    for a, b, f in parent.sea:
        a = np.asarray(a, dtype=np.float64)
        a_xy, b_xy = frame.to_xy(np.stack([a, np.asarray(b, dtype=np.float64)]))
        if shore.is_empty:
            legs.append((a_xy, b_xy, f))
            continue
        road, at_end = None, True
        if len(ends):
            d = np.linalg.norm(ends - a, axis=2) * EARTH_RADIUS_KM * 1000.0
            k = int(np.argmin(d))
            if d.flat[k] <= reach_m:
                road, at_end = divmod(k, 2)[0], bool(divmod(k, 2)[1])
        if road is not None and road in landed:
            land_xy = landed[road]
        else:
            if road is not None:
                xy = frame.to_xy(polylines[road])
                inside = LineString(xy).intersection(frame_box)
                if inside.is_empty:
                    legs.append((a_xy, b_xy, f))
                    continue
                on, near = nearest_points(shore, inside)
            else:
                if not frame_box.buffer(reach_m).contains(Point(a_xy)):
                    legs.append((a_xy, b_xy, f))
                    continue
                on, near = nearest_points(shore, Point(a_xy))
            land_xy = np.array([on.x, on.y])
            if on.distance(near) > reach_m:
                legs.append((a_xy, b_xy, f))
                continue
            if road is not None:
                # Cut the road at its vertex nearest the landing, then on to it.
                pos = polylines[road] if at_end else polylines[road][::-1]
                k = int(np.argmin(np.linalg.norm(frame.to_xy(pos) - land_xy, axis=1)))
                pos = np.concatenate([pos[:max(k, 1)], frame.to_ecef(land_xy)])
                polylines[road] = pos if at_end else pos[::-1].copy()
                flows = np.asarray(parent.polyline_flows[road], dtype=np.float64)
                n = len(pos) - 1
                parent.polyline_flows[road] = flows[:n] if at_end else flows[::-1][:n][::-1].copy()
                landed[road] = land_xy
        legs.append((land_xy, b_xy, f))
        for landing in landings:
            if np.linalg.norm(landing.xy - land_xy) < merge_m:
                landing.flow += float(f)
                break
        else:
            heading = b_xy - land_xy
            norm = float(np.linalg.norm(heading))
            on_pt = Point(land_xy)
            sea_d = water.sea.distance(on_pt) if not water.sea.is_empty else np.inf
            lake_d = water.lakes.distance(on_pt) if not water.lakes.is_empty else np.inf
            kind: Literal["sea", "lake", "river"] = "sea" if sea_d <= lake_d else "lake"
            landings.append(Landing(
                xy=land_xy, direction=heading / norm if norm > 0 else np.zeros(2), flow=float(f), water=kind))
    return landings, legs, polylines


def _quays(
    legs: list[RoadLine], pins: list[tuple[bool, bool]], water: SiteWater, channels: MultiPolygon, frame: Frame,
) -> list[Landing]:
    """Where each river or lake leg pinned at a port in the frame first reaches
    water (the nearest bank, for a port on the water)."""
    still = water.sea.union(water.lakes)
    wet = still.union(channels)
    frame_box = frame.box()
    out = []
    for leg, pin in zip(legs, pins):
        for pinned, xy, flows in ((pin[0], leg.xy, leg.flows), (pin[1], leg.xy[::-1], leg.flows[::-1])):
            if not pinned or not frame_box.contains(Point(xy[0])):
                continue
            if wet.contains(Point(xy[0])):
                # The port is drawn on water here (the finer channel covers the
                # local map's bank): its quay is the nearest bank.
                bank = nearest_points(wet.boundary, Point(xy[0]))[0]
                at, heading = np.array([bank.x, bank.y]), xy[0] - np.array([bank.x, bank.y])
                if float(np.linalg.norm(heading)) < 1e-9:
                    heading = xy[1] - xy[0]
                hit = 1
            else:
                hit = next((k for k in range(1, len(xy)) if wet.contains(Point(xy[k]))), None)
                if hit is None:
                    continue
                prev, first_wet = xy[hit - 1], xy[hit]
                entry = _points(LineString([prev, first_wet]).intersection(wet.boundary))
                at = min(entry, key=lambda p: float(np.linalg.norm(p - prev))) if entry else first_wet
                heading = first_wet - prev
            heading = heading / max(float(np.linalg.norm(heading)), 1e-12)
            on = Point(at + heading)
            kind: Literal["sea", "lake", "river"] = (
                "sea" if water.sea.contains(on) else "lake" if water.lakes.contains(on) else "river")
            out.append(Landing(xy=np.asarray(at, dtype=np.float64), direction=heading, flow=float(flows[hit - 1]),
                               water=kind))
    return out


def _merge_landings(landings: list[Landing], merge_m: float) -> list[Landing]:
    out: list[Landing] = []
    for landing in landings:
        same = next((q for q in out if np.linalg.norm(q.xy - landing.xy) < merge_m), None)
        if same is None:
            out.append(landing)
        else:
            same.flow += landing.flow
    return out


def _land_at(lines: list[RoadLine], landings: list[Landing], tol_m: float) -> None:
    """End each road that stops within ``tol_m`` of a landing exactly on it
    (re-tracing snaps a shore point a texel onto land)."""
    for landing in landings:
        for road in lines:
            for end in (0, -1):
                if float(np.linalg.norm(road.xy[end] - landing.xy)) < tol_m:
                    road.xy[end] = landing.xy
