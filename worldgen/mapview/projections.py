import numpy as np


class Projection:
    name: str

    def _to_lonlat(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # ECEF convention: Z is north, lon=0 at +X axis
        lat = np.arcsin(np.clip(vertices[:, 2], -1.0, 1.0))
        lon = np.arctan2(vertices[:, 1], vertices[:, 0])
        return lon, lat

    def project(self, vertices: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class EquirectangularProjection(Projection):
    name = "Equirectangular"

    def project(self, vertices: np.ndarray) -> np.ndarray:
        lon, lat = self._to_lonlat(vertices)
        u = lon / np.pi
        v = lat / (np.pi / 2)
        return np.column_stack([u, v]).astype(np.float32)


class MercatorProjection(Projection):
    name = "Mercator"

    _LAT_LIMIT = np.radians(85.051129)
    _Y_MAX = float(np.log(np.tan(np.pi / 4 + np.radians(85.051129) / 2)))

    def project(self, vertices: np.ndarray) -> np.ndarray:
        lon, lat = self._to_lonlat(vertices)
        u = lon / np.pi
        lat_c = np.clip(lat, -self._LAT_LIMIT, self._LAT_LIMIT)
        v = np.log(np.tan(np.pi / 4 + lat_c / 2)) / self._Y_MAX
        return np.column_stack([u, v]).astype(np.float32)


class MollweideProjection(Projection):
    name = "Mollweide"

    def project(self, vertices: np.ndarray) -> np.ndarray:
        lon, lat = self._to_lonlat(vertices)

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

    def project(self, vertices: np.ndarray) -> np.ndarray:
        lon, lat = self._to_lonlat(vertices)
        abs_lat_deg = np.abs(lat) * (180.0 / np.pi)
        plen = np.interp(abs_lat_deg, _ROBINSON_LATS, _ROBINSON_PLEN)
        pdfe = np.interp(abs_lat_deg, _ROBINSON_LATS, _ROBINSON_PDFE)
        u = plen * (lon / np.pi)
        v = np.sign(lat) * pdfe
        return np.column_stack([u, v]).astype(np.float32)


ALL_PROJECTIONS: list[Projection] = [
    EquirectangularProjection(),
    MercatorProjection(),
    MollweideProjection(),
    RobinsonProjection(),
]
