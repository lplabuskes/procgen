"""Visual check on a settlement map — the site's layers, the layout, conservation.

Builds one settlement map (``docs/settlement_maps.md``) through the chain, as the
viewer does, and renders it with the viewer itself (``RegionView``, offscreen):
the frame's relief, water, roads, footprint, hinterland and layout. Prints:

  • the frame (side, texel) and each step's wall-clock time;
  • the **conservation report**: per local texel of every local map the frame
    reads, the settlement's mean elevation against the local value (float32 noise);
  • the site: water (sea/lake area, rivers and their widths, islands), roads
    (gates, bridges, fords, landings), footprint (people, density, area, pieces,
    sparse share, error against ``population / density``), anchor candidates,
    hinterland (own fields vs wanted, neighbours);
  • the profile (kind, rule, wealth, industries, materials, the history stubs)
    and the layout (summary line, district areas).

Usage:
    python3 scripts/debug_settlement.py                        # seed 4, its largest city
    python3 scripts/debug_settlement.py 4 --city 13
    python3 scripts/debug_settlement.py 536582844 --town 716 --zoom 3
    python3 scripts/debug_settlement.py 536582844 --village 50:115:3 --zoom 3
    python3 scripts/debug_settlement.py 4 --city 13 --anchors --buildable
    python3 scripts/debug_settlement.py 4 --city 13 --out /tmp/city.png

Writes ``scripts/results/settlement_<seed>_<kind><index>.png`` by default.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from settlegen.builder import SettlementMap, build_settlement_map  # noqa: E402
from settlegen.chain import SettlementRef, population  # noqa: E402
from settlegen.layout import summary  # noqa: E402
from settlegen.upsample import settlement_conservation_report  # noqa: E402
from _common import get_map_store, settlement_map_params  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
OVERLAYS = ("rivers", "trade_routes", "settlements", "towns", "villages")


def largest_city(world) -> int:  # type: ignore[no-untyped-def]
    n = len(world.get_feature("settlements"))
    return int(np.argmax([population(world, SettlementRef("city", i)) for i in range(n)]))


def parse_ref(args, world) -> SettlementRef:  # type: ignore[no-untyped-def]
    if args.town is not None:
        return SettlementRef("town", args.town)
    if args.village is not None:
        region, hex_index, index = (int(x) for x in args.village.split(":"))
        return SettlementRef("village", index, map=(region, hex_index))
    return SettlementRef("city", args.city if args.city is not None else largest_city(world))


def _parts(geom) -> int:  # type: ignore[no-untyped-def]
    return len(getattr(geom, "geoms", [geom])) if not geom.is_empty else 0


def report(m: SettlementMap, store, timings: dict[str, float]) -> None:  # type: ignore[no-untyped-def]
    site, profile, layout = m.site, m.profile, m.layout
    ref = m.ref
    where = f" on local map {ref.map[0]}/{ref.map[1]}" if ref.map is not None else ""
    print(f"{ref.kind} {ref.index}{where} · {summary(layout, profile)}")
    texel_m = site.frame.side_m / m.fields.tex_size
    print(f"  frame {site.frame.side_m / 1000:.2f} km at {texel_m:.1f} m ({m.fields.tex_size}² texels) · "
          f"local maps {m.chain.keys}")
    print("  time " + " · ".join(f"{k} {v:.2f} s" for k, v in timings.items())
          + f" · total {sum(timings.values()):.2f} s")

    print("  conservation residual per local texel (downsample must recover the local maps):")
    globe = store.globe
    for name, stats in settlement_conservation_report(m.fields, [mp.fields for mp in m.chain.maps],
                                                      globe.antimeridian).items():
        print(f"    {name:<14} max {stats['max']:.3e}  mean {stats['mean']:.3e}  over {stats['n_anchors']} anchors")

    w = site.water
    ha = 1e-4
    widths = [float(np.max(line.width_m)) for line in w.lines]
    islands = sum(len(p.interiors) for p in getattr(w.channels, "geoms", []))
    print(f"  water: sea {w.sea.area * ha:,.0f} ha · lakes {w.lakes.area * ha:,.0f} ha · "
          f"{len(w.lines)} rivers/streams" + (f" ({min(widths):.0f}–{max(widths):.0f} m wide)" if widths else "")
          + f" · {islands} islands")
    r = site.roads
    bridges = sum(c.bridge for c in r.crossings)
    landings = ", ".join(f"{la.water}" for la in r.landings) or "none"
    print(f"  roads: {len(r.lines)} roads, {len(r.water)} water legs, {len(r.sea)} sea legs · "
          f"{len(r.gates)} gates · {bridges} bridges, {len(r.crossings) - bridges} fords · landings: {landings}")
    pop = float(profile.population)
    want = pop / max(site.density_per_ha, 1e-9) / ha
    dense = site.footprint.difference(site.sparse) if not site.sparse.is_empty else site.footprint
    err = dense.area / want - 1.0 if want > 0 else float("nan")
    print(f"  footprint: {pop:,.0f} people at {site.density_per_ha:.0f}/ha · {site.footprint.area * ha:,.1f} ha "
          f"in {_parts(site.footprint)} piece(s) · sparse {site.sparse.area / max(site.footprint.area, 1e-9):.0%} "
          f"· dense area vs population/density {err:+.1%}")
    print("  anchors: " + ", ".join(f"{k} {len(v)}" for k, v in site.anchors.by_kind.items()))
    h = site.hinterland
    print(f"  hinterland: own {h.own_kind} {h.own.area * ha:,.0f} of {h.own_target_ha:,.0f} ha wanted · "
          f"{len(h.neighbours)} neighbours in the frame")

    pol = profile.polity
    inds = ", ".join(f"{i.name} {i.weight:.2f}" for i in profile.industries) or "none"
    mats = ", ".join(f"{k} {v:.2f}" for k, v in sorted(profile.materials.items(), key=lambda kv: -kv[1]) if v > 0)
    print(f"  profile: {profile.tier_name} ({profile.kind}; {', '.join(profile.site_kinds) or '-'}) · "
          f"{pol.control}{' capital' if pol.capital else ''} march {pol.march:.2f} · wealth {profile.wealth:.2f} · "
          f"walls {profile.wall_strength:.2f}" + (" walled" if profile.walls else ""))
    print(f"    industries: {inds}")
    print(f"    materials: {mats} · wind {np.linalg.norm(profile.wind_xy):.1f} m/s "
          f"(steadiness {profile.wind_steadiness:.2f}) · founded ~{profile.founding_age:.0f} y ago"
          + (" · planned" if profile.planned else ""))
    areas = sorted(layout.district_areas_ha().items(), key=lambda kv: -kv[1])
    print("  districts (ha): " + ", ".join(f"{k} {v:.1f}" for k, v in areas))
    if layout.walls is not None:
        print(f"  walls: {layout.walls.kind}, {layout.walls.line.length / 1000:.2f} km, "
              f"{len(layout.walls.towers)} towers, {len(layout.walls.gates)} gates")
    if layout.footprint is not None:
        extra = layout.footprint.difference(site.footprint).area * ha
        print(f"  walls took in {extra:.1f} ha beyond the site's footprint")
    if layout.piers:
        print(f"  docks: {len(layout.piers)} piers, quays {layout.quays.area * ha:.2f} ha")


def render(m: SettlementMap, store, params, zoom: float, out: Path,  # type: ignore[no-untyped-def]
           anchors: bool, buildable: bool) -> None:
    from PySide6.QtWidgets import QApplication
    from core.display import SchemeSpec
    from viz.mapview.regionview import RegionView
    from worldgen.geology.elevation_builder import ELEVATION_RECIPE
    app = QApplication.instance() or QApplication([])  # noqa: F841  (kept alive while rendering)
    view = RegionView()
    view.resize(1000, 1000)
    view.set_store(store)
    view.set_layer("Elevation", SchemeSpec(lambda _: (0, 0, 0, 1), "elevation_rgba", None, ELEVATION_RECIPE))
    view.set_settlement(m, store.globe, land_texels=int(params.upsample.water_spread_texels))
    for name in OVERLAYS:
        view.set_overlay_visible(name, True)
    view.set_overlay_visible("settlement_anchors", anchors)
    view.set_overlay_visible("buildable", buildable)
    view._zoom = float(zoom)
    out.parent.mkdir(parents=True, exist_ok=True)
    view.grab().save(str(out))
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seed", nargs="?", type=int, default=4)
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--city", type=int, help="city index (default: the largest)")
    which.add_argument("--town", type=int, help="town index")
    which.add_argument("--village", metavar="R:H:I", help="village I of local map (region R, hex H)")
    which.add_argument("--largest", action="store_true", help="the largest city (the default)")
    parser.add_argument("--zoom", type=float, default=1.0, help="view zoom (1: the whole frame)")
    parser.add_argument("--anchors", action="store_true", help="draw the anchor candidates (debug overlay)")
    parser.add_argument("--buildable", action="store_true", help="shade unbuildable ground (debug overlay)")
    parser.add_argument("--rebuild-cache", action="store_true", help="rebuild the cached physical world")
    parser.add_argument("--out", type=Path, help="output PNG (default scripts/results/settlement_<...>.png)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    store = get_map_store(args.seed, args.rebuild_cache)
    print(f"seed {args.seed} · world loaded in {time.perf_counter() - t0:.1f} s")
    ref = parse_ref(args, store.globe)
    params = settlement_map_params(store)
    timings: dict[str, float] = {}
    m = build_settlement_map(store, ref, params, timings)
    report(m, store, timings)
    tag = f"village{ref.map[0]}_{ref.map[1]}_{ref.index}" if ref.map is not None else f"{ref.kind}{ref.index}"
    out = args.out or RESULTS_DIR / f"settlement_{args.seed}_{tag}.png"
    render(m, store, params, args.zoom, out, args.anchors, args.buildable)


if __name__ == "__main__":
    main()
