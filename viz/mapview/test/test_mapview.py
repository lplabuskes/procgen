import numpy as np
from viz.mapview.projections import (
    EquirectangularProjection,
    MercatorProjection,
    MollweideProjection,
    RobinsonProjection,
    ALL_PROJECTIONS,
)
from viz.mapview.mapview import _split_at_antimeridian


def _lonlat_to_ecef(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _make_ecef(*lonlat_pairs) -> np.ndarray:
    return np.array([_lonlat_to_ecef(lon, lat) for lon, lat in lonlat_pairs])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _check_equator_at_zero(proj):
    """Any point on the equator should project to v=0."""
    for lon_deg in (-180, -90, 0, 90):
        pts = np.radians([[lon_deg, 0.0]])
        uv = proj.project_lonlat(pts)
        assert abs(uv[0, 1]) < 1e-6, f"{proj.name}: equator v={uv[0, 1]} at lon={lon_deg}"


def _check_prime_meridian_at_zero(proj):
    """Any point on the prime meridian should project to u=0."""
    for lat_deg in (-60, -30, 0, 30, 60):
        pts = np.radians([[0.0, lat_deg]])
        uv = proj.project_lonlat(pts)
        assert abs(uv[0, 0]) < 1e-6, f"{proj.name}: prime meridian u={uv[0, 0]} at lat={lat_deg}"


def _check_ns_symmetry(proj):
    """(lon, lat) → (u, v) should imply (lon, -lat) → (u, -v)."""
    lons = np.radians([-150, -90, -45, 0, 45, 90, 150])
    lats = np.radians([10, 30, 60])
    for lat in lats:
        pts_n = np.column_stack([lons, np.full_like(lons, lat)])
        pts_s = np.column_stack([lons, np.full_like(lons, -lat)])
        uv_n = proj.project_lonlat(pts_n)
        uv_s = proj.project_lonlat(pts_s)
        np.testing.assert_allclose(uv_n[:, 0], uv_s[:, 0], atol=1e-5,
                                   err_msg=f"{proj.name}: u broken N/S symmetry at lat={np.degrees(lat)}")
        np.testing.assert_allclose(uv_n[:, 1], -uv_s[:, 1], atol=1e-5,
                                   err_msg=f"{proj.name}: v broken N/S symmetry at lat={np.degrees(lat)}")


def _check_ew_symmetry(proj):
    """(lon, lat) → (u, v) should imply (-lon, lat) → (-u, v)."""
    lons = np.radians([30, 60, 90, 120])
    lats = np.radians([0, 20, 45, 70])
    for lat in lats:
        pts_e = np.column_stack([lons, np.full_like(lons, lat)])
        pts_w = np.column_stack([-lons, np.full_like(lons, lat)])
        uv_e = proj.project_lonlat(pts_e)
        uv_w = proj.project_lonlat(pts_w)
        np.testing.assert_allclose(uv_e[:, 0], -uv_w[:, 0], atol=1e-5,
                                   err_msg=f"{proj.name}: u broken E/W symmetry at lat={np.degrees(lat)}")
        np.testing.assert_allclose(uv_e[:, 1], uv_w[:, 1], atol=1e-5,
                                   err_msg=f"{proj.name}: v broken E/W symmetry at lat={np.degrees(lat)}")


def _check_project_vs_project_lonlat(proj):
    """project() from ECEF and project_lonlat() from explicit lon/lat must agree."""
    test_cases = [
        (0.0, 0.0), (90.0, 45.0), (-90.0, -45.0), (135.0, 70.0), (-45.0, -20.0),
    ]
    ecef = _make_ecef(*test_cases)
    lonlat = np.radians(test_cases)
    uv_from_ecef = proj.project(ecef)
    uv_from_lonlat = proj.project_lonlat(lonlat)
    np.testing.assert_allclose(uv_from_ecef, uv_from_lonlat, atol=1e-6,
                               err_msg=f"{proj.name}: project() vs project_lonlat() mismatch")


# ---------------------------------------------------------------------------
# Equirectangular
# ---------------------------------------------------------------------------

class TestEquirectangularProjection:
    proj = EquirectangularProjection()

    def test_equator_at_v_zero(self):
        _check_equator_at_zero(self.proj)

    def test_prime_meridian_at_u_zero(self):
        _check_prime_meridian_at_zero(self.proj)

    def test_ns_symmetry(self):
        _check_ns_symmetry(self.proj)

    def test_ew_symmetry(self):
        _check_ew_symmetry(self.proj)

    def test_project_vs_project_lonlat(self):
        _check_project_vs_project_lonlat(self.proj)

    def test_north_pole_at_v_one(self):
        pts = np.array([[0.0, np.pi / 2]])
        uv = self.proj.project_lonlat(pts)
        assert abs(uv[0, 1] - 1.0) < 1e-6

    def test_south_pole_at_v_minus_one(self):
        pts = np.array([[0.0, -np.pi / 2]])
        uv = self.proj.project_lonlat(pts)
        assert abs(uv[0, 1] + 1.0) < 1e-6

    def test_antimeridian_east_at_u_one(self):
        pts = np.array([[np.pi, 0.0]])
        uv = self.proj.project_lonlat(pts)
        assert abs(uv[0, 0] - 1.0) < 1e-6

    def test_antimeridian_west_at_u_minus_one(self):
        pts = np.array([[-np.pi, 0.0]])
        uv = self.proj.project_lonlat(pts)
        assert abs(uv[0, 0] + 1.0) < 1e-6

    def test_output_in_unit_square(self):
        lons = np.linspace(-np.pi, np.pi, 36)
        lats = np.linspace(-np.pi / 2, np.pi / 2, 18)
        lon_g, lat_g = np.meshgrid(lons, lats)
        pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])
        uv = self.proj.project_lonlat(pts)
        assert np.all(uv >= -1.0 - 1e-6) and np.all(uv <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Mercator
# ---------------------------------------------------------------------------

class TestMercatorProjection:
    proj = MercatorProjection()

    def test_equator_at_v_zero(self):
        _check_equator_at_zero(self.proj)

    def test_prime_meridian_at_u_zero(self):
        _check_prime_meridian_at_zero(self.proj)

    def test_ns_symmetry(self):
        _check_ns_symmetry(self.proj)

    def test_ew_symmetry(self):
        _check_ew_symmetry(self.proj)

    def test_project_vs_project_lonlat(self):
        _check_project_vs_project_lonlat(self.proj)

    def test_v_monotone_with_latitude(self):
        lats = np.radians([-80, -60, -30, 0, 30, 60, 80])
        pts = np.column_stack([np.zeros_like(lats), lats])
        uv = self.proj.project_lonlat(pts)
        v = uv[:, 1]
        assert np.all(np.diff(v) > 0), "Mercator v must be strictly increasing with lat"

    def test_lat_limit_clamps_output(self):
        lim = MercatorProjection._LAT_LIMIT
        pts_at_limit = np.array([[0.0, lim]])
        pts_beyond = np.array([[0.0, lim + 0.01]])
        uv_at = self.proj.project_lonlat(pts_at_limit)
        uv_beyond = self.proj.project_lonlat(pts_beyond)
        assert abs(uv_at[0, 1] - uv_beyond[0, 1]) < 1e-6

    def test_output_bounded(self):
        # Within the lat limit, v must stay in [-1, 1]
        lats = np.linspace(-MercatorProjection._LAT_LIMIT, MercatorProjection._LAT_LIMIT, 100)
        lons = np.linspace(-np.pi, np.pi, 100)
        pts = np.column_stack([lons, lats])
        uv = self.proj.project_lonlat(pts)
        assert np.all(np.abs(uv[:, 0]) <= 1.0 + 1e-6)
        assert np.all(np.abs(uv[:, 1]) <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Mollweide
# ---------------------------------------------------------------------------

class TestMollweideProjection:
    proj = MollweideProjection()

    def test_equator_at_v_zero(self):
        _check_equator_at_zero(self.proj)

    def test_prime_meridian_at_u_zero(self):
        _check_prime_meridian_at_zero(self.proj)

    def test_ns_symmetry(self):
        _check_ns_symmetry(self.proj)

    def test_ew_symmetry(self):
        _check_ew_symmetry(self.proj)

    def test_project_vs_project_lonlat(self):
        _check_project_vs_project_lonlat(self.proj)

    def test_poles_project_to_u_zero(self):
        # Poles are degenerate; all longitudes should map to u=0
        for lon_deg in (-90, 0, 90):
            for lat_deg in (90, -90):
                pts = np.radians([[lon_deg, lat_deg]])
                uv = self.proj.project_lonlat(pts)
                assert abs(uv[0, 0]) < 1e-5, (
                    f"Mollweide pole u={uv[0, 0]} at lon={lon_deg}, lat={lat_deg}"
                )

    def test_equatorial_antimeridian_u(self):
        # At lat=0, theta=0, u = lon / (pi/sqrt(2)) = pi / (pi/sqrt(2)) = sqrt(2)
        pts = np.array([[np.pi, 0.0]])
        uv = self.proj.project_lonlat(pts)
        np.testing.assert_allclose(uv[0, 0], np.sqrt(2), atol=1e-5)

    def test_output_within_ellipse(self):
        # The Mollweide ellipse: (u/(sqrt(2)))^2 + (v/sqrt(2))^2 <= 1
        lons = np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 36)
        lats = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 18)
        lon_g, lat_g = np.meshgrid(lons, lats)
        pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])
        uv = self.proj.project_lonlat(pts)
        # semi-axes: a=sqrt(2) in u, b=sqrt(2) in v
        ellipse_val = (uv[:, 0] / np.sqrt(2)) ** 2 + (uv[:, 1] / np.sqrt(2)) ** 2
        assert np.all(ellipse_val <= 1.0 + 1e-4), "Points outside Mollweide ellipse"


# ---------------------------------------------------------------------------
# Robinson
# ---------------------------------------------------------------------------

class TestRobinsonProjection:
    proj = RobinsonProjection()

    def test_equator_at_v_zero(self):
        _check_equator_at_zero(self.proj)

    def test_prime_meridian_at_u_zero(self):
        _check_prime_meridian_at_zero(self.proj)

    def test_ns_symmetry(self):
        _check_ns_symmetry(self.proj)

    def test_ew_symmetry(self):
        _check_ew_symmetry(self.proj)

    def test_project_vs_project_lonlat(self):
        _check_project_vs_project_lonlat(self.proj)

    def test_poles_at_v_one(self):
        pts_n = np.array([[0.0, np.pi / 2]])
        pts_s = np.array([[0.0, -np.pi / 2]])
        uv_n = self.proj.project_lonlat(pts_n)
        uv_s = self.proj.project_lonlat(pts_s)
        assert abs(uv_n[0, 1] - 1.0) < 1e-6, f"Robinson north pole v={uv_n[0, 1]}"
        assert abs(uv_s[0, 1] + 1.0) < 1e-6, f"Robinson south pole v={uv_s[0, 1]}"

    def test_v_monotone_with_latitude(self):
        lats = np.radians([-90, -60, -30, 0, 30, 60, 90])
        pts = np.column_stack([np.zeros_like(lats), lats])
        uv = self.proj.project_lonlat(pts)
        v = uv[:, 1]
        assert np.all(np.diff(v) > 0), "Robinson v must be strictly increasing with lat"

    def test_plen_decreases_toward_poles(self):
        # Parallel length (plen) shrinks as lat → 90°, so equal-lon spacing → smaller u
        lats = np.radians([0, 30, 60, 89])
        pts = np.column_stack([np.full_like(lats, np.pi / 2), lats])
        uv = self.proj.project_lonlat(pts)
        u = uv[:, 0]
        assert np.all(np.diff(u) < 0), "Robinson u (at fixed lon) should decrease toward pole"


# ---------------------------------------------------------------------------
# ALL_PROJECTIONS
# ---------------------------------------------------------------------------

class TestAllProjections:
    def test_list_length(self):
        assert len(ALL_PROJECTIONS) == 4

    def test_all_have_names(self):
        assert all(isinstance(p.name, str) and p.name for p in ALL_PROJECTIONS)

    def test_all_return_float32(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.5]])
        for proj in ALL_PROJECTIONS:
            uv = proj.project_lonlat(pts)
            assert uv.dtype == np.float32, f"{proj.name} returned {uv.dtype}, expected float32"

    def test_all_return_correct_shape(self):
        pts = np.radians([[0, 0], [90, 45], [-90, -45]])
        for proj in ALL_PROJECTIONS:
            uv = proj.project_lonlat(pts)
            assert uv.shape == (3, 2), f"{proj.name} output shape {uv.shape}"


# ---------------------------------------------------------------------------
# Antimeridian splitting
# ---------------------------------------------------------------------------

class TestSplitAtAntimeridian:
    def test_no_crossing_returns_single_polygon(self):
        face = _make_ecef((10, 20), (30, 40), (50, 10))
        result = _split_at_antimeridian(face)
        assert len(result) == 1
        assert result[0].shape == (3, 2)

    def test_western_face_no_crossing(self):
        face = _make_ecef((-10, 20), (-30, 40), (-50, 10))
        result = _split_at_antimeridian(face)
        assert len(result) == 1

    def test_crossing_returns_two_polygons(self):
        # Two crossing edges: lone eastern vertex + two western vertices
        face = _make_ecef((170, 20), (-170, 30), (-160, -10))
        result = _split_at_antimeridian(face)
        assert len(result) == 2

    def test_both_halves_have_enough_vertices(self):
        face = _make_ecef((170, 20), (-170, 30), (-160, -10))
        result = _split_at_antimeridian(face)
        for poly in result:
            assert poly.shape[0] >= 3

    def test_crossing_halves_have_correct_lon_signs(self):
        # Eastern half should have lon ≥ 0, western half lon ≤ 0
        face = _make_ecef((170, 20), (-170, 30), (-160, -10))
        result = _split_at_antimeridian(face)
        lons = [poly[:, 0] for poly in result]
        east_lons = [arr for arr in lons if np.all(arr >= -1e-9)]
        west_lons = [arr for arr in lons if np.all(arr <= 1e-9)]
        assert len(east_lons) == 1
        assert len(west_lons) == 1

    def test_antimeridian_boundary_lat_matches(self):
        # The latitude at the split boundary should be the same in both halves.
        face = _make_ecef((170, 0), (-170, 0), (90, 45))
        result = _split_at_antimeridian(face)
        assert len(result) == 2
        # Boundary points are where lon == ±π
        east = result[0] if result[0][0, 0] >= 0 else result[1]
        west = result[0] if result[0][0, 0] < 0 else result[1]
        east_boundary_lats = east[east[:, 0] > np.pi - 1e-6, 1]
        west_boundary_lats = west[west[:, 0] < -np.pi + 1e-6, 1]
        assert len(east_boundary_lats) > 0
        assert len(west_boundary_lats) > 0
        np.testing.assert_allclose(
            np.sort(east_boundary_lats), np.sort(west_boundary_lats), atol=1e-6
        )

    def test_output_polygons_are_lonlat(self):
        face = _make_ecef((10, 20), (30, 40), (50, 10))
        result = _split_at_antimeridian(face)
        poly = result[0]
        lons, lats = poly[:, 0], poly[:, 1]
        assert np.all(lons >= -np.pi - 1e-6) and np.all(lons <= np.pi + 1e-6)
        assert np.all(lats >= -np.pi / 2 - 1e-6) and np.all(lats <= np.pi / 2 + 1e-6)
