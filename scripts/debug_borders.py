"""Visual border-drawing troubleshooter — fill vs. traced lines, at texel resolution.

Renders a zoomed raster of one region: the polity fill (``fill.PolityLabelField``)
and the border polylines contoured from it (``borders.py``).  Only texels inside
``mesh.face_texture_in_polygon`` are drawn, as the renderer clips the rest;
``--show-overhang`` adds the clipped over-hang (faded, red-edged).

Open endpoints in the printed diagnostics should all be coastal.

Usage:
    python3 scripts/debug_borders.py                      # seed 4, auto-pick a busy border region
    python3 scripts/debug_borders.py 31 --face 1234       # centre on a face
    python3 scripts/debug_borders.py --lat 12 --lon -40   # centre on lat/lon (degrees)
    python3 scripts/debug_borders.py --radius 8           # angular half-width of view (deg)
    python3 scripts/debug_borders.py --show-overhang      # reveal invisible over-hang texels
    python3 scripts/debug_borders.py --depth 0            # only sovereign borders (default: all)
    python3 scripts/debug_borders.py --seams --vertices   # overlay hexagon seams + border vertices
    python3 scripts/debug_borders.py --rebuild-cache --out /tmp/borders.png
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402

from core.pipeline import BuildContext  # noqa: E402
from realmgen.builder import RealmParams, RealmPipeline  # noqa: E402
from realmgen.refine.fill import WATER, UNCLAIMED, _polity_color_map  # noqa: E402
from realmgen.polity.types import UNCLAIMED_COLOR  # noqa: E402
from worldgen.geology.channels import OCEAN_MASK  # noqa: E402
from worldgen.geology.display import OCEAN_COLOR  # noqa: E402
from worldgen.hydrology.channels import LAKE_MASK  # noqa: E402

from _common import get_physical_world, add_seed_args, resolve_seeds  # noqa: E402

DEFAULT_SEEDS = [4]
DEPTH_COLORS = ["black", "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]


def build_realm(seed: int, rebuild: bool, lattice_offset: int | None = None):
    """Build the cached physical world + realm pipeline; return the texel labels
    and borders it produced (so they come from the current fill/border code)."""
    world = get_physical_world(seed, rebuild)
    params = RealmParams()
    if lattice_offset is not None:
        params.refine.border_lattice_offset = lattice_offset
    realm = RealmPipeline(params)
    realm.build(BuildContext(world=world, seed=seed))
    mesh = world.mesh

    polities = world.get_feature("polities") or []
    field = world.get_feature("polity_label_field")
    ocean_tex = OCEAN_MASK.read(mesh)
    lake_tex = LAKE_MASK.read(mesh) if LAKE_MASK.present(mesh) else None
    H, W = int(ocean_tex.shape[1]), int(ocean_tex.shape[2])
    labels = field.texel_labels(mesh.face_texture_positions(H, W), ocean_tex, lake_tex)
    borders = world.get_feature("polity_borders") or {}
    return world, mesh, polities, labels, borders


def label_colors(polities) -> dict:
    """``label -> rgb`` covering polity ids plus WATER/UNCLAIMED sentinels."""
    cmap = dict(_polity_color_map(polities))
    cmap[WATER] = np.asarray(OCEAN_COLOR[:3], dtype=np.float32)
    cmap[UNCLAIMED] = np.asarray(UNCLAIMED_COLOR[:3], dtype=np.float32)
    return cmap


def colors_for(labels_flat: np.ndarray, cmap: dict) -> np.ndarray:
    grey = np.asarray(UNCLAIMED_COLOR[:3], dtype=np.float32)
    out = np.tile(grey, (labels_flat.size, 1))
    for i, lab in enumerate(labels_flat):
        rgb = cmap.get(int(lab))
        if rgb is not None:
            out[i] = rgb
    return out


def local_basis(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """East/north tangent basis at a unit-vector ``center`` (orthographic frame)."""
    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, center)
    if np.linalg.norm(east) < 1e-6:        # center at a geographic pole
        east = np.array([1.0, 0.0, 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(center, east)
    north /= np.linalg.norm(north)
    return east, north


def pick_center_face(mesh, labels, in_poly) -> int:
    """Auto-pick a border-rich face: the one whose visible texels carry the most
    distinct sovereign (depth-0 label proxy) regions in its local neighbourhood."""
    n = labels.shape[0]
    diversity = np.zeros(n)
    for f in range(n):
        for g in [f, *mesh.face_adjacency[f]]:
            vis = labels[g][in_poly[g]]
            claimed = vis[vis >= 0]
            diversity[f] += len(np.unique(claimed))
    return int(np.argmax(diversity))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_seed_args(ap)
    ap.add_argument("--face", type=int, default=None,
                    help="centre the view on this face id (default: auto-pick a busy border)")
    ap.add_argument("--lat", type=float, default=None, help="centre latitude (deg)")
    ap.add_argument("--lon", type=float, default=None, help="centre longitude (deg)")
    ap.add_argument("--radius", type=float, default=6.0,
                    help="angular half-width of the view (deg, default 6)")
    ap.add_argument("--depth", type=int, default=None,
                    help="only draw this border depth (default: all)")
    ap.add_argument("--show-overhang", action="store_true",
                    help="also draw the invisible over-hang texels (faded, red-edged)")
    ap.add_argument("--seams", action="store_true", help="overlay hexagon seam outlines")
    ap.add_argument("--vertices", action="store_true", help="mark border segment vertices")
    ap.add_argument("--rebuild-cache", action="store_true", help="force rebuild worlds")
    ap.add_argument("--lattice-offset", type=int, default=None,
                    help="border lattice depth offset (default: RefineParams.border_lattice_offset)")
    ap.add_argument("--out", default=None, help="output PNG (default scripts/results/borders_<seed>.png)")
    args = ap.parse_args()

    seed = resolve_seeds(args, DEFAULT_SEEDS)[0]
    world, mesh, polities, labels, borders = build_realm(seed, args.rebuild_cache, args.lattice_offset)
    N, H, W = labels.shape
    in_poly = np.asarray(mesh.face_texture_in_polygon(H, W))
    positions = np.asarray(mesh.face_texture_positions(H, W), dtype=np.float64)
    centroids = np.asarray(mesh.face_centroids, dtype=np.float64)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # ── view centre ─────────────────────────────────────────────────────────
    if args.lat is not None or args.lon is not None:
        lat, lon = np.radians(args.lat or 0.0), np.radians(args.lon or 0.0)
        center = np.array([np.cos(lat) * np.cos(lon),
                           np.cos(lat) * np.sin(lon), np.sin(lat)])
        center_face = int(np.argmax(centroids @ center))
    else:
        center_face = args.face if args.face is not None else pick_center_face(mesh, labels, in_poly)
        center = centroids[center_face]
    center = center / np.linalg.norm(center)
    east, north = local_basis(center)
    rad = np.radians(args.radius)
    # Cull generously (the square's corners sit at ~rad×1.41); the axis limits do
    # the cropping.  Culling at `rad` would fake gaps where borders cross the edge.
    cull = np.cos(rad * 1.6)

    in_view = np.where(centroids @ center > cull)[0]
    cmap = label_colors(polities)

    def to_2d(P: np.ndarray) -> np.ndarray:
        return np.column_stack([P @ east, P @ north])

    # ── gather visible / over-hang texels as quads ──────────────────────────
    # Texels as corner quads: the texel grid is arbitrarily rotated, so square
    # markers would leave stripes.
    def face_corners(C: np.ndarray) -> np.ndarray:
        """(H, W, 3) texel centres → (H+1, W+1, 3) cell corners (edge-extrapolated)."""
        H_, W_, _ = C.shape
        ext = np.empty((H_ + 2, W_ + 2, 3))
        ext[1:-1, 1:-1] = C
        ext[0, 1:-1] = 2 * C[0] - C[1]
        ext[-1, 1:-1] = 2 * C[-1] - C[-2]
        ext[1:-1, 0] = 2 * C[:, 0] - C[:, 1]
        ext[1:-1, -1] = 2 * C[:, -1] - C[:, -2]
        ext[0, 0] = 2 * C[0, 0] - C[1, 1]
        ext[0, -1] = 2 * C[0, -1] - C[1, -2]
        ext[-1, 0] = 2 * C[-1, 0] - C[-2, 1]
        ext[-1, -1] = 2 * C[-1, -1] - C[-2, -2]
        return 0.25 * (ext[:-1, :-1] + ext[:-1, 1:] + ext[1:, :-1] + ext[1:, 1:])

    def gather_quads(visible: bool):
        quads, labs = [], []
        for f in in_view:
            mask = (in_poly[f] if visible else ~in_poly[f]) & (positions[f] @ center > cull)
            if not mask.any():
                continue
            crn = to_2d(face_corners(positions[f]).reshape(-1, 3)).reshape(H + 1, W + 1, 2)
            tl, tr = crn[:-1, :-1], crn[:-1, 1:]
            br, bl = crn[1:, 1:], crn[1:, :-1]
            q = np.stack([tl, tr, br, bl], axis=2).reshape(H * W, 4, 2)
            m = mask.reshape(-1)
            quads.append(q[m])
            labs.append(labels[f].reshape(-1)[m])
        if not quads:
            return np.empty((0, 4, 2)), np.empty(0, dtype=np.int64)
        return np.concatenate(quads), np.concatenate(labs)

    fig, ax = plt.subplots(figsize=(12.0, 12.0))

    if args.show_overhang:
        oh_q, oh_lab = gather_quads(False)
        if len(oh_q):
            ax.add_collection(PolyCollection(
                list(oh_q), facecolors=colors_for(oh_lab, cmap), alpha=0.25,
                edgecolors="red", linewidths=0.1))

    vis_q, vis_lab = gather_quads(True)
    if len(vis_q):
        ax.add_collection(PolyCollection(
            list(vis_q), facecolors=colors_for(vis_lab, cmap), linewidths=0))

    # ── seams (hexagon outlines) ────────────────────────────────────────────
    if args.seams:
        seam_segs = []
        for f in in_view:
            vids = list(mesh.faces[f])
            V = mesh.vertices[np.array(vids)].astype(np.float64)
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            for a, b in zip(V, np.roll(V, -1, axis=0)):
                if a @ center > cull or b @ center > cull:
                    seam_segs.append([to_2d(a[None])[0], to_2d(b[None])[0]])
        if seam_segs:
            ax.add_collection(LineCollection(seam_segs, colors="0.5",
                                             linewidths=0.5, linestyles=":"))

    # ── border polylines ────────────────────────────────────────────────────
    depths = sorted(borders) if args.depth is None else [args.depth]
    for depth in depths:
        segs = borders.get(depth)
        if segs is None or not len(segs):
            continue
        a, b = segs[:, :3], segs[:, 3:]
        # Either end in range, so segments crossing the view edge are kept.
        keep = (a @ center > cull) | (b @ center > cull)
        if not keep.any():
            continue
        lines = [[to_2d(a[i][None])[0], to_2d(b[i][None])[0]] for i in np.where(keep)[0]]
        col = DEPTH_COLORS[depth % len(DEPTH_COLORS)]
        ax.add_collection(LineCollection(lines, colors=col, linewidths=1.4,
                                         label=f"depth {depth} ({int(keep.sum())} segs)"))
        if args.vertices:
            vtx = np.vstack([a[keep], b[keep]])
            vxy = to_2d(vtx)
            ax.scatter(vxy[:, 0], vxy[:, 1], s=6, c=col, zorder=5)

    lim = np.sin(rad)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(
        f"seed {seed} — centre face {center_face} — radius {args.radius}° — "
        f"{'overhang shown' if args.show_overhang else 'visible texels only'}"
    )
    if depths:
        ax.legend(loc="upper right", fontsize=8)

    # ── textual diagnostics ─────────────────────────────────────────────────
    print(f"seed {seed}: {N} faces, {H}x{W} texels/face")
    vis_frac = in_poly.mean()
    print(f"  visible texels: {vis_frac * 100:.1f}% in-polygon, "
          f"{(1 - vis_frac) * 100:.1f}% over-hang (invisible)")
    for depth in sorted(borders):
        segs = borders[depth]
        # degree-1 (open) endpoints are coastal terminations OR unwelded gaps.
        keyc: Counter = Counter()
        for row in segs:
            keyc[tuple(np.round(row[:3], 5))] += 1
            keyc[tuple(np.round(row[3:], 5))] += 1
        open_ends = sum(1 for v in keyc.values() if v == 1)
        print(f"  depth {depth}: {len(segs):>6} segments, {open_ends:>5} open endpoints")

    out = args.out or str(Path(__file__).parent / "results" / f"borders_{seed}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
