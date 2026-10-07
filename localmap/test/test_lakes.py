"""Local maps inherit the region's small lakes and find smaller ones of their own."""
from typing import Any

import numpy as np
import pytest

from localmap.upsample import build_local_fields, path_segments
from localmap.test.conftest import local_params
from region.lakes import drain_nodes
from region.rivers import build_region_rivers
from region.upsample import SMALL_LAKE_KEY, RegionUpsampleParams, build_region_fields

REGION_ID = 51   # mostly land and thawing (see region/test/test_lakes.py)
OPEN: dict[str, Any] = dict(
    lake_breach_parent_texels=0.0, lake_min_depth_km=0.0, lake_min_texels=1, lake_min_aridity_index=0.0,
)


@pytest.fixture(scope="module")
def region(globe):  # noqa: F811
    return build_region_fields(globe, REGION_ID, RegionUpsampleParams(n_cols=6, tex_size=16, **OPEN))


@pytest.fixture(scope="module")
def pond_hex(region) -> int:
    """The regional hex with the most small-lake texels."""
    return int(np.argmax((region.textures[SMALL_LAKE_KEY][..., 0] > 0.5).sum(axis=(1, 2))))


def _parent(local, region, key):
    flat = region.textures[key][..., 0].reshape(-1)
    return (flat[local.parent_anchors.reshape(-1)] > 0.5).reshape(local.parent_anchors.shape)


def test_region_small_lakes_carry_down(globe, region, pond_hex):  # noqa: F811
    local = build_local_fields(globe, region, pond_hex, local_params(small_lakes=False))
    small = local.textures[SMALL_LAKE_KEY][..., 0] > 0.5
    parent = _parent(local, region, SMALL_LAKE_KEY) & local.domain
    assert parent.any()
    # Re-flooded at the region's surface: the shore moves, the lakes stay.
    assert (small & parent).sum() > 0.5 * parent.sum()
    assert not (small & ~(local.textures["lake_mask"][..., 0] > 0.5)).any()


def test_local_maps_add_their_own(globe, region, pond_hex):  # noqa: F811
    off = build_local_fields(globe, region, pond_hex, local_params(small_lakes=False))
    on = build_local_fields(globe, region, pond_hex, local_params(**OPEN))
    count = [(f.textures[SMALL_LAKE_KEY][..., 0] > 0.5).sum() for f in (off, on)]
    assert count[1] > count[0]


def test_no_small_lake_on_a_region_river(globe, region):  # noqa: F811
    rivers = build_region_rivers(region, globe)
    assert rivers.paths
    segments = path_segments(rivers.paths)
    assert segments.shape[1:] == (2, 3)
    # The most river-crossed hex among those with lake basins in them.
    crossed = np.bincount(
        np.argmin(np.linalg.norm(region.grid.centers_ecef()[None] - segments[:, 0, None], axis=-1), axis=1),
        minlength=region.n_hexes)
    ponds = (region.textures[SMALL_LAKE_KEY][..., 0] > 0.5).any(axis=(1, 2))
    hex_index = int(np.argmax(np.where(ponds, crossed, -1)))
    local = build_local_fields(globe, region, hex_index, local_params(**OPEN), drains=segments)
    inherited = build_local_fields(globe, region, hex_index, local_params(small_lakes=False))
    new = (local.textures[SMALL_LAKE_KEY][..., 0] > 0.5) & ~(inherited.textures[SMALL_LAKE_KEY][..., 0] > 0.5)
    assert new.any()
    tex = local.tex_size
    step = 2.0 * local.grid.radius / tex
    lo = -local.grid.radius
    n = int(np.ceil((1.0 - 2.0 * lo) / step)) + 1
    on_river = drain_nodes(local.grid, segments, lo, step, n).reshape(n, n)
    st = local.grid.texture_local(tex)
    c = np.clip(np.rint((st[..., 0] - lo) / step).astype(int), 0, n - 1)
    r = np.clip(np.rint((st[..., 1] - lo) / step).astype(int), 0, n - 1)
    assert on_river[r, c].any()
    assert not (new & on_river[r, c]).any()
