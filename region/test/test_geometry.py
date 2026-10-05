import numpy as np
import pytest

from region.geometry import (
    N_REGIONS,
    N_SUBDIV,
    RegionGeometry,
    cell_border_polylines,
    ea_forward,
    ea_inverse,
    ecef_to_cell,
)

# A few antimeridian rotations to prove assignment is orientation-stable.
ANTIMERIDIANS = [np.pi, 0.0, 2.3, -1.1]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_ea_forward_inverse_roundtrip():
    u = np.linspace(-1.0, 1.0, 101)
    assert np.allclose(ea_inverse(ea_forward(u)), u, atol=1e-9)
    p = np.linspace(0.0, 1.0, 101)
    assert np.allclose(ea_forward(ea_inverse(p)), p, atol=1e-9)


def test_ea_forward_endpoints():
    assert ea_forward(np.array([-1.0, 0.0, 1.0])) == pytest.approx([0.0, 0.5, 1.0])


@pytest.mark.parametrize("antimeridian", ANTIMERIDIANS)
def test_ecef_to_cell_in_range(antimeridian):
    rng = np.random.default_rng(0)
    pts = _unit(rng.standard_normal((5000, 3)))
    cells = ecef_to_cell(pts, antimeridian)
    assert cells.min() >= 0
    assert cells.max() < N_REGIONS


@pytest.mark.parametrize("antimeridian", ANTIMERIDIANS)
def test_all_regions_are_hit(antimeridian):
    """A dense sphere sampling must land in every one of the 54 regions."""
    rng = np.random.default_rng(1)
    pts = _unit(rng.standard_normal((200_000, 3)))
    cells = ecef_to_cell(pts, antimeridian)
    assert set(cells.tolist()) == set(range(N_REGIONS))


@pytest.mark.parametrize("antimeridian", ANTIMERIDIANS)
def test_local_to_cell_roundtrip(antimeridian):
    """Interior local points of a region map back to that region."""
    grid = np.linspace(0.08, 0.92, 6)
    ss, tt = np.meshgrid(grid, grid)
    ss, tt = ss.ravel(), tt.ravel()
    for region_id in range(N_REGIONS):
        geom = RegionGeometry(region_id, antimeridian)
        ecef = geom.local_to_ecef(ss, tt)
        assert np.allclose(np.linalg.norm(ecef, axis=1), 1.0, atol=1e-9)
        assert np.all(ecef_to_cell(ecef, antimeridian) == region_id)


@pytest.mark.parametrize("antimeridian", ANTIMERIDIANS)
def test_ecef_to_local_inverts_local_to_ecef(antimeridian):
    """Including past the cell's edges, where the frame keeps extending."""
    grid = np.linspace(-0.4, 1.4, 7)
    ss, tt = np.meshgrid(grid, grid)
    ss, tt = ss.ravel(), tt.ravel()
    for region_id in range(N_REGIONS):
        geom = RegionGeometry(region_id, antimeridian)
        s, t, valid = geom.ecef_to_local(geom.local_to_ecef(ss, tt))
        assert valid.all()
        np.testing.assert_allclose(s, ss, atol=1e-9)
        np.testing.assert_allclose(t, tt, atol=1e-9)


def test_ecef_to_local_far_side_is_invalid():
    geom = RegionGeometry(22, 0.3)
    _, _, valid = geom.ecef_to_local(-geom.center_ecef()[None, :])
    assert not valid[0]


def test_region_id_decomposition():
    for region_id in range(N_REGIONS):
        geom = RegionGeometry(region_id, np.pi)
        recomposed = geom.face * N_SUBDIV * N_SUBDIV + geom.ci * N_SUBDIV + geom.cj
        assert recomposed == region_id
        assert 0 <= geom.face < 6
        assert 0 <= geom.ci < N_SUBDIV
        assert 0 <= geom.cj < N_SUBDIV


def test_region_id_out_of_range():
    with pytest.raises(ValueError):
        RegionGeometry(N_REGIONS, np.pi)
    with pytest.raises(ValueError):
        RegionGeometry(-1, np.pi)


@pytest.mark.parametrize("antimeridian", ANTIMERIDIANS)
def test_center_ecef(antimeridian):
    for region_id in range(N_REGIONS):
        geom = RegionGeometry(region_id, antimeridian)
        c = geom.center_ecef()
        assert c.shape == (3,)
        assert np.linalg.norm(c) == pytest.approx(1.0, abs=1e-9)
        assert int(ecef_to_cell(c[None, :], antimeridian)[0]) == region_id


def test_sample_grid_shape_and_membership():
    geom = RegionGeometry(20, np.pi)
    grid = geom.sample_grid(res_s=7, res_t=5)
    assert grid.shape == (5, 7, 3)
    flat = grid.reshape(-1, 3)
    assert np.allclose(np.linalg.norm(flat, axis=1), 1.0, atol=1e-9)
    assert np.all(ecef_to_cell(flat, np.pi) == 20)


def test_local_to_lonlat_consistent_with_ecef():
    geom = RegionGeometry(7, np.pi)
    s = np.linspace(0.1, 0.9, 9)
    t = np.linspace(0.1, 0.9, 9)
    lonlat = geom.local_to_lonlat(s, t)
    ecef = geom.local_to_ecef(s, t)
    lon = np.arctan2(ecef[:, 1], ecef[:, 0])
    lat = np.arcsin(np.clip(ecef[:, 2], -1.0, 1.0))
    assert np.allclose(lonlat[:, 0], lon)
    assert np.allclose(lonlat[:, 1], lat)


@pytest.mark.parametrize("antimeridian", ANTIMERIDIANS)
def test_local_frame_is_right_handed(antimeridian):
    """``ds x dt`` points outward on every region: a mirrored face can't be fixed by screen rotation."""
    eps = 1e-4
    c = 0.5
    for region_id in range(N_REGIONS):
        geom = RegionGeometry(region_id, antimeridian)
        center = geom.local_to_ecef(np.array([c]), np.array([c]))[0]
        ds = (
            geom.local_to_ecef(np.array([c + eps]), np.array([c]))[0]
            - geom.local_to_ecef(np.array([c - eps]), np.array([c]))[0]
        )
        dt = (
            geom.local_to_ecef(np.array([c]), np.array([c + eps]))[0]
            - geom.local_to_ecef(np.array([c]), np.array([c - eps]))[0]
        )
        assert np.dot(np.cross(ds, dt), center) > 0, f"region {region_id} (face {geom.face}) is mirrored"


@pytest.mark.parametrize("antimeridian", ANTIMERIDIANS)
def test_up_direction_matches_true_north_or_antimeridian(antimeridian):
    """``up_direction()`` is true north, or the antimeridian tangent on a pole-centred cell."""
    eps = 1e-4
    c = 0.5
    for region_id in range(N_REGIONS):
        geom = RegionGeometry(region_id, antimeridian)
        center = geom.local_to_ecef(np.array([c]), np.array([c]))[0]
        ds = (
            geom.local_to_ecef(np.array([c + eps]), np.array([c]))[0]
            - geom.local_to_ecef(np.array([c - eps]), np.array([c]))[0]
        )
        dt = (
            geom.local_to_ecef(np.array([c]), np.array([c + eps]))[0]
            - geom.local_to_ecef(np.array([c]), np.array([c - eps]))[0]
        )
        z_hat = np.array([0.0, 0.0, 1.0])
        north = z_hat - np.dot(z_hat, center) * center
        if np.linalg.norm(north) < 1e-6:
            target = np.array([np.cos(antimeridian), np.sin(antimeridian), 0.0])
        else:
            target = _unit(north)

        a, b = geom.up_direction()
        rendered_up = _unit(a * ds + b * dt)
        assert np.dot(rendered_up, target) == pytest.approx(1.0, abs=1e-6)


def test_up_direction_equatorial_faces_need_no_rotation():
    """The 4 equatorial cube faces already have +t pointing at true north."""
    for region_id in range(0, 36):  # faces 0-3
        geom = RegionGeometry(region_id, np.pi)
        ds, dt = geom.up_direction()
        assert ds == pytest.approx(0.0, abs=1e-6)
        assert dt > 0.0


def test_up_direction_pole_regions_agree_on_the_antimeridian():
    """Both pole-centred regions render the same real-world direction as up.

    Only the sphere direction must match, not the raw ``(ds, dt)`` (axis conventions differ).
    """
    eps = 1e-4
    c = 0.5
    for region_id in (40, 49):  # face 4 / face 5 centre cell
        geom = RegionGeometry(region_id, np.pi)
        assert np.linalg.norm(np.abs(geom.center_ecef()) - np.array([0, 0, 1])) < 1e-9

    def rendered_up(region_id: int) -> np.ndarray:
        geom = RegionGeometry(region_id, np.pi)
        ds = (
            geom.local_to_ecef(np.array([c + eps]), np.array([c]))[0]
            - geom.local_to_ecef(np.array([c - eps]), np.array([c]))[0]
        )
        dt = (
            geom.local_to_ecef(np.array([c]), np.array([c + eps]))[0]
            - geom.local_to_ecef(np.array([c]), np.array([c - eps]))[0]
        )
        a, b = geom.up_direction()
        return _unit(a * ds + b * dt)

    assert rendered_up(40) == pytest.approx(rendered_up(49), abs=1e-6)


def test_border_polylines_on_unit_sphere():
    polys = cell_border_polylines(np.pi, samples=50)
    # 6 faces * (N+1 constant-pu + N+1 constant-pv) gridlines.
    assert len(polys) == 6 * 2 * (N_SUBDIV + 1)
    for poly in polys:
        assert poly.shape == (50, 3)
        assert np.allclose(np.linalg.norm(poly, axis=1), 1.0, atol=1e-9)
