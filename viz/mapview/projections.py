import numpy as np


class Projection:
    name: str

    def _to_lonlat(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # ECEF convention: Z is north, lon=0 at +X axis
        lat = np.arcsin(np.clip(vertices[:, 2], -1.0, 1.0))
        lon = np.arctan2(vertices[:, 1], vertices[:, 0])
        return lon, lat

    def _from_lonlat(self, _lon: np.ndarray, _lat: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def project(self, vertices: np.ndarray) -> np.ndarray:
        lon, lat = self._to_lonlat(vertices)
        return self._from_lonlat(lon, lat)

    def project_lonlat(self, lonlat: np.ndarray) -> np.ndarray:
        return self._from_lonlat(lonlat[:, 0], lonlat[:, 1])

    def unproject_lonlat(self, uv: np.ndarray) -> np.ndarray:
        """Inverse of project_lonlat.  uv: (N, 2).
        Returns (N, 2) [shifted_lon, lat] with nan for out-of-bounds points."""
        return np.full((len(uv), 2), np.nan, dtype=np.float64)


class EquirectangularProjection(Projection):
    name = "Equirectangular"

    def _from_lonlat(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        u = lon / np.pi
        v = lat / (np.pi / 2)
        return np.column_stack([u, v]).astype(np.float32)

    def unproject_lonlat(self, uv: np.ndarray) -> np.ndarray:
        lon = uv[:, 0] * np.pi
        lat = uv[:, 1] * (np.pi / 2)
        result = np.column_stack([lon, lat]).astype(np.float64)
        oob = (np.abs(uv[:, 0]) > 1.0) | (np.abs(uv[:, 1]) > 1.0)
        result[oob] = np.nan
        return result


class MercatorProjection(Projection):
    name = "Mercator"

    _LAT_LIMIT = np.radians(85.051129)
    _Y_MAX = float(np.log(np.tan(np.pi / 4 + np.radians(85.051129) / 2)))

    def _from_lonlat(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        u = lon / np.pi
        lat_c = np.clip(lat, -self._LAT_LIMIT, self._LAT_LIMIT)
        v = np.log(np.tan(np.pi / 4 + lat_c / 2)) / self._Y_MAX
        return np.column_stack([u, v]).astype(np.float32)

    def unproject_lonlat(self, uv: np.ndarray) -> np.ndarray:
        lon = uv[:, 0] * np.pi
        lat = 2.0 * np.arctan(np.exp(uv[:, 1] * self._Y_MAX)) - np.pi / 2.0
        result = np.column_stack([lon, lat]).astype(np.float64)
        oob = (np.abs(uv[:, 0]) > 1.0) | (np.abs(uv[:, 1]) > 1.0)
        result[oob] = np.nan
        return result


class MollweideProjection(Projection):
    name = "Mollweide"

    def _from_lonlat(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        # Iterative Newton solve for auxiliary angle theta
        theta = lat.copy()
        sin_lat = np.sin(lat)
        near_pole = np.abs(lat) > np.pi / 2 - 1e-6
        for _ in range(10):
            denom = 2 + 2 * np.cos(2 * theta)
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            theta -= (2 * theta + np.sin(2 * theta) - np.pi * sin_lat) / denom
        theta = np.where(near_pole, np.sign(lat) * (np.pi / 2), theta)

        # Normalize so the map ellipse fits in [-1, 1] x [-1, 1]
        u = (lon * np.cos(theta)) / (np.pi / np.sqrt(2))
        v = np.sin(theta) / (1.0 / np.sqrt(2))
        return np.column_stack([u, v]).astype(np.float32)

    def unproject_lonlat(self, uv: np.ndarray) -> np.ndarray:
        # v = √2 sin(θ) → θ = arcsin(v/√2); valid for |v| ≤ √2
        sin_theta = uv[:, 1] / np.sqrt(2.0)
        valid = np.abs(sin_theta) <= 1.0
        theta = np.where(valid, np.arcsin(np.clip(sin_theta, -1.0, 1.0)), 0.0)
        lat_arg = np.clip((2.0 * theta + np.sin(2.0 * theta)) / np.pi, -1.0, 1.0)
        lat = np.arcsin(lat_arg)
        cos_theta = np.cos(theta)
        safe_cos = np.where(np.abs(cos_theta) < 1e-6, 1e-6, cos_theta)
        lon = uv[:, 0] * (np.pi / np.sqrt(2.0)) / safe_cos
        result = np.column_stack([lon, lat]).astype(np.float64)
        # OOB: outside the Mollweide circle of radius √2, i.e. u²+v² > 2
        oob = ~valid | (uv[:, 0] ** 2 + uv[:, 1] ** 2 > 2.0)
        result[oob] = np.nan
        return result


_ROBINSON_LATS = np.arange(0, 91, 5, dtype=np.float64)
_ROBINSON_PLEN = np.array([
    1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600,
    0.9427, 0.9216, 0.8962, 0.8679, 0.8350, 0.7986, 0.7597,
    0.7186, 0.6732, 0.6213, 0.5722, 0.5322,
], dtype=np.float64)
_ROBINSON_PDFE = np.array([
    0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720,
    0.4340, 0.4958, 0.5571, 0.6176, 0.6769, 0.7346, 0.7903,
    0.8435, 0.8936, 0.9394, 0.9761, 1.0000,
], dtype=np.float64)


class RobinsonProjection(Projection):
    name = "Robinson"

    def _from_lonlat(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        abs_lat_deg = np.abs(lat) * (180.0 / np.pi)
        plen = np.interp(abs_lat_deg, _ROBINSON_LATS, _ROBINSON_PLEN)
        pdfe = np.interp(abs_lat_deg, _ROBINSON_LATS, _ROBINSON_PDFE)
        u = plen * (lon / np.pi)
        v = np.sign(lat) * pdfe
        return np.column_stack([u, v]).astype(np.float32)

    def unproject_lonlat(self, uv: np.ndarray) -> np.ndarray:
        # Invert pdfe: table is monotonically increasing, np.interp works in both directions
        abs_v = np.abs(uv[:, 1])
        lat_deg = np.interp(abs_v, _ROBINSON_PDFE, _ROBINSON_LATS)
        lat = np.sign(uv[:, 1]) * lat_deg * (np.pi / 180.0)
        plen = np.interp(lat_deg, _ROBINSON_LATS, _ROBINSON_PLEN)
        safe_plen = np.where(plen < 1e-6, 1e-6, plen)
        lon = uv[:, 0] * np.pi / safe_plen
        result = np.column_stack([lon, lat]).astype(np.float64)
        oob = (abs_v > 1.0) | (np.abs(lon) > np.pi * 1.01)
        result[oob] = np.nan
        return result


ALL_PROJECTIONS: list[Projection] = [
    RobinsonProjection(),
    MollweideProjection(),
    MercatorProjection(),
    EquirectangularProjection(),
]
