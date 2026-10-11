"""The region and local maps built so far for one world, shared by everything that needs them.

A region or local map is built only when it isn't already held: the viewer and
the settlement chain (:mod:`settlegen.chain`) ask the same store, so opening a
region in the viewer and then a settlement in it builds each map once.  Each map
is ~100 MB at defaults, so the store keeps the most recently used
``region_capacity`` regions and ``local_capacity`` local maps.

The store also owns the tiers' params and their reset rules (as the viewer had):

- new region params reset every town's and city's refinement to its stage site;
  new ``upsample`` params drop every region, other changes keep the regions'
  fields and only re-run their features (rivers, roads, settlements) on next use;
  either way every local map goes;
- new local params reset every local refinement and drop every local map;
- a new globe drops everything.

Realm and world params are the ones the world was built with (changes come with
a new globe), read when refining.  Qt-free.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

from localmap.builder import LocalMap, build_local_map
from localmap.params import LocalParams
from region.builder import RegionMap, refine_region_features
from region.params import RegionParams
from region.upsample import RegionFields, build_region_fields

if TYPE_CHECKING:
    from core.world import World
    from realmgen.builder import RealmParams
    from worldgen.builder import GlobeParams


class MapStore:
    """Region and local maps of one globe, built on demand.  See the module docstring."""

    REGION_CAPACITY = 3
    LOCAL_CAPACITY = 6

    def __init__(
        self, globe: Optional["World"] = None, *,
        region_capacity: int = REGION_CAPACITY, local_capacity: int = LOCAL_CAPACITY,
    ) -> None:
        self.globe = globe
        self.region_params = RegionParams()
        self.local_params = LocalParams()
        self.realm_params: Optional["RealmParams"] = None
        self.world_params: Optional["GlobeParams"] = None
        self.region_capacity = int(region_capacity)
        self.local_capacity = int(local_capacity)
        self._regions: "OrderedDict[int, RegionMap]" = OrderedDict()
        # Regions whose fields are still good but whose features must be re-run.
        self._stale: dict[int, RegionFields] = {}
        self._locals: "OrderedDict[tuple[int, int], LocalMap]" = OrderedDict()
        # Every build, in order: ("region", id) / ("local", (id, hex)) — for tests and timing.
        self.builds: list[tuple[str, object]] = []

    # ── params ───────────────────────────────────────────────────────────────

    def set_globe(self, globe: Optional["World"]) -> None:
        """Hold ``globe``'s maps from now on (a different globe drops everything)."""
        if globe is self.globe:
            return
        self.globe = globe
        self.clear()

    def set_tuning(self, realm: Optional["RealmParams"] = None, world: Optional["GlobeParams"] = None) -> None:
        """The realm and world params the globe was built with."""
        self.realm_params, self.world_params = realm, world

    def set_region_params(self, params: RegionParams) -> bool:
        """Replace the region tuning (see the module docstring); True if it changed."""
        if params == self.region_params:
            return False
        old, self.region_params = self.region_params, params
        if self.globe is not None:
            towns = self.globe.get_feature("towns")
            if towns is not None:
                towns.reset_refinement()
            for city in self.globe.get_feature("settlements") or []:
                city.reset_refinement()
        if params.upsample != old.upsample:
            self._regions.clear()
            self._stale.clear()
        else:
            self._stale.update({rid: m.fields for rid, m in self._regions.items()})
            self._regions.clear()
        self._locals.clear()
        return True

    def set_local_params(self, params: LocalParams) -> bool:
        """Replace the local tuning (see the module docstring); True if it changed."""
        if params == self.local_params:
            return False
        self.local_params = params
        if self.globe is not None:
            towns = self.globe.get_feature("towns")
            if towns is not None:
                towns.reset_local_refinement()
            for city in self.globe.get_feature("settlements") or []:
                city.reset_local_refinement()
        self._locals.clear()
        return True

    def clear(self) -> None:
        """Drop every map (settlement positions are left as they are)."""
        self._regions.clear()
        self._stale.clear()
        self._locals.clear()

    # ── maps ─────────────────────────────────────────────────────────────────

    def has_region(self, region_id: int) -> bool:
        return int(region_id) in self._regions

    def has_local(self, region_id: int, hex_index: int) -> bool:
        return (int(region_id), int(hex_index)) in self._locals

    def region(self, region_id: int) -> RegionMap:
        """Region ``region_id``'s map, built only if not held."""
        region_id = int(region_id)
        held = self._regions.pop(region_id, None)
        if held is None:
            held = self._build_region(region_id)
        self._regions[region_id] = held
        while len(self._regions) > self.region_capacity:
            self._regions.popitem(last=False)
        return held

    def local(self, region_id: int, hex_index: int) -> LocalMap:
        """The local map of regional hex ``hex_index`` of region ``region_id``, built
        (with its region, if needed) only if not held."""
        key = (int(region_id), int(hex_index))
        held = self._locals.pop(key, None)
        if held is None:
            held = self._build_local(*key)
        self._locals[key] = held
        while len(self._locals) > self.local_capacity:
            self._locals.popitem(last=False)
        return held

    def _globe(self) -> "World":
        if self.globe is None:
            raise ValueError("the store has no globe")
        return self.globe

    def _build_region(self, region_id: int) -> RegionMap:
        globe = self._globe()
        realm, world = self.realm_params, self.world_params
        fields = self._stale.pop(region_id, None)
        if fields is None:
            fields = build_region_fields(
                globe, region_id, self.region_params.upsample,
                resources=world.resources if world is not None else None,
                fertility_exponent=float(realm.population.fertility_exponent) if realm is not None else 2.0,
            )
        built = refine_region_features(
            globe, fields, self.region_params,
            land=world.cost.land if world is not None else None,
            danger_scale=float(realm.danger.movement_scale) if realm is not None else 2.0,
            town_params=realm.towns if realm is not None else None,
            population_params=realm.population if realm is not None else None,
            river_threshold_factor=realm.refine.river_threshold_factor if realm is not None else 1.5,
        )
        self.builds.append(("region", region_id))
        return built

    def _build_local(self, region_id: int, hex_index: int) -> LocalMap:
        globe = self._globe()
        region = self.region(region_id)
        realm, world = self.realm_params, self.world_params
        built = build_local_map(
            globe, region.fields, hex_index, self.local_params,
            region_rivers=region.rivers, region_roads=region.roads, region_deltas=region.deltas,
            region_streams=region.streams(globe, self.local_params.streams),
            resources=world.resources if world is not None else None,
            fertility_exponent=float(realm.population.fertility_exponent) if realm is not None else 2.0,
            land=world.cost.land if world is not None else None,
            danger_scale=float(realm.danger.movement_scale) if realm is not None else 2.0,
            town_params=realm.towns if realm is not None else None,
            population_params=realm.population if realm is not None else None,
            river_threshold_factor=realm.refine.river_threshold_factor if realm is not None else 1.5,
        )
        self.builds.append(("local", (region_id, hex_index)))
        return built
