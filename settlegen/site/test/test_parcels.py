"""The hinterland cut into parcels, on the synthetic lake frame."""
import dataclasses

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Point

from localmap.farmland import FarmlandParams
from settlegen.chain.builder import SettlementRef
from settlegen.site.hinterland import Neighbour, build_hinterland
from settlegen.site.parcels import CROP_STYLES, GARDEN_STYLE, PASTURE_STYLE, SettlementParcelParams, build_parcels
from settlegen.site.types import _as_multipolygon
from worldgen.resources.crops import CROP_NAMES


def _params(frame, **kw):
    """Parcels scaled as on a 4 km frame: the synthetic frame is far wider."""
    return SettlementParcelParams(size_scale=frame.side_m / 4000.0, **kw)


def _with_crop(fields, name):
    tex = np.full_like(fields.textures["dominant_crop"], CROP_NAMES.index(name))
    return dataclasses.replace(fields, textures={**fields.textures, "dominant_crop": tex})


@pytest.fixture(scope="module")
def farmed(site, globe):
    """A village by the frame's centre and a neighbour, with plenty of fields each."""
    fields, frame, water = site
    still = shapely.union_all([water.sea, water.lakes])
    spots = [np.array(s) * f * frame.half for f in np.linspace(0.3, 0.9, 13) for s in ((1, 1), (-1, -1), (1, -1))]
    dry = [p for p in spots if not still.contains(Point(p))]
    people = 0.08 * (frame.side_m ** 2 / 1e4) / FarmlandParams().ha_per_farmer
    footprint = _as_multipolygon(Point(0.0, 0.0).buffer(0.05 * frame.half).difference(still))
    neighbour = Neighbour(xy=dry[-1], kind="village", index=0, map=(0, 0), population=people, mining=False,
                          fields=MultiPolygon())
    hint = build_hinterland(globe, fields, frame, water, footprint, [], [], SettlementRef("village", 0, map=(0, 0)),
                            people, neighbours=[neighbour])
    assert not hint.own.is_empty and not hint.neighbours[0].fields.is_empty
    return hint


def _narrow_side(poly):
    rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
    sides = np.linalg.norm(np.diff(rect[:3], axis=0), axis=1)
    return float(sides.min())


@pytest.mark.parametrize("crop", ["Wheat", "Rice", "Maize", "Coffee"])
def test_parcels_tile_each_owners_fields_without_overlap(site, farmed, crop):
    fields, frame, _ = site
    parcels = build_parcels(_with_crop(fields, crop), frame, farmed, [4, 1], _params(frame))
    assert parcels is not None and parcels.style == crop
    assert len(parcels.polygons) == len(parcels.owner) == len(parcels.shade)
    assert parcels.shade.max() < len(parcels.palette) == len(parcels.edge)
    for k, geom in enumerate([farmed.own, farmed.neighbours[0].fields]):
        mine = np.array([p for p, o in zip(parcels.polygons, parcels.owner) if o == k], dtype=object)
        assert len(mine)
        # Inside the owner's fields, and covering nearly all of it (only slivers dropped).
        assert shapely.area(shapely.difference(mine, geom)).sum() < 1e-6 * geom.area
        assert shapely.area(mine).sum() > 0.95 * geom.area
        # No two parcels overlap, but for the centimetre snapping's slivers along their edges
        # (pairwise: a union of thousands of strips is not exact).
        a, b = shapely.STRtree(mine).query(mine, predicate="overlaps")
        assert shapely.area(shapely.intersection(mine[a], mine[b])).sum() < 1e-4 * geom.area


@pytest.mark.parametrize("crop", ["Wheat", "Maize", "Sugarcane"])
def test_parcels_are_the_crops_strip_width(site, farmed, crop):
    fields, frame, _ = site
    parcels = build_parcels(_with_crop(fields, crop), frame, farmed, [4, 1], _params(frame))
    assert parcels is not None
    # Whole strips (not cut by a furlong's edge) are about the style's width.
    widths = np.array([_narrow_side(p) for p in parcels.polygons])
    assert np.median(widths) == pytest.approx(CROP_STYLES[crop].strip_m * _params(frame).size_scale, rel=0.35)


def test_open_fields_lie_in_three_great_fields(site, farmed):
    fields, frame, _ = site
    parcels = build_parcels(_with_crop(fields, "Wheat"), frame, farmed, [4, 1], _params(frame))
    assert parcels is not None
    own = (parcels.owner == 0) & ~parcels.pasture
    states = parcels.shade[own] // 5
    counts = np.bincount(states, minlength=3)
    assert (counts > 0.1 * own.sum()).all()


def test_a_citys_own_ring_is_gardens(site, farmed):
    fields, frame, _ = site
    gardens = dataclasses.replace(farmed, own_kind="gardens")
    parcels = build_parcels(_with_crop(fields, "Wheat"), frame, gardens, [4, 1], _params(frame))
    assert parcels is not None and parcels.style == "Gardens"
    n_crop, n_garden = len(CROP_STYLES["Wheat"].states) * 5, len(GARDEN_STYLE.states) * 5
    assert len(parcels.palette) == n_crop + n_garden + len(PASTURE_STYLE.states) * 5
    # The settlement's parcels in the garden shades (no pasture of their own); its neighbour's
    # in the crop's, or the pasture's.
    own = parcels.owner == 0
    assert not parcels.pasture[own].any()
    assert ((parcels.shade[own] >= n_crop) & (parcels.shade[own] < n_crop + n_garden)).all()
    theirs = parcels.owner == 1
    assert (parcels.shade[theirs & ~parcels.pasture] < n_crop).all()
    assert (parcels.shade[theirs & parcels.pasture] >= n_crop + n_garden).all()


@pytest.mark.parametrize("share", [0.67, 1.0])
def test_a_share_of_the_fields_is_pasture_mostly_farthest_out(site, farmed, share):
    fields, frame, _ = site
    parcels = build_parcels(_with_crop(fields, "Wheat"), frame, farmed, [4, 1], _params(frame, arable_share=share))
    assert parcels is not None
    own = parcels.owner == 0
    polys = np.array(parcels.polygons, dtype=object)
    area = shapely.area(polys)
    pasture = own & parcels.pasture
    if share == 1.0:
        assert not parcels.pasture.any()
        return
    # Whole furlongs go to pasture: about (1 − share) of the area.
    assert area[pasture].sum() / area[own].sum() == pytest.approx(1.0 - share, abs=0.08)
    # Farther from the farmer, on average, than the cropland.
    dist = np.hypot(*shapely.get_coordinates(shapely.centroid(polys)).T)
    mean = lambda m: float(np.average(dist[m], weights=area[m]))  # noqa: E731
    assert mean(pasture) > mean(own & ~parcels.pasture)


def test_parcels_are_deterministic_and_can_be_switched_off(site, farmed):
    fields, frame, _ = site
    a = build_parcels(fields, frame, farmed, [4, 1], _params(frame))
    b = build_parcels(fields, frame, farmed, [4, 1], _params(frame))
    assert a is not None and b is not None
    assert np.array_equal(a.shade, b.shade)
    assert all(p.equals_exact(q, 0.0) for p, q in zip(a.polygons, b.polygons))
    assert build_parcels(fields, frame, farmed, [4, 1], _params(frame, enabled=False)) is None
