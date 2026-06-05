import numpy as np
import pytest

from worldgen.geology.elevation import (
    compute_elevation_textures,
    compute_noise_elevation_deltas,
    face_elevations_from_texture,
)
from worldgen.geology.elevation_builder import ElevationBuilder, ElevationParams
from core.mesh import Mesh
from core.pipeline import BuildContext
from core.world import World


def _bilinear_sample(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear-sample a (H, W, ...) array at a continuous UV in [0, 1]²."""
    H, W = texture.shape[:2]
    u = float(np.clip(uv[0], 0.0, 1.0))
    v = float(np.clip(uv[1], 0.0, 1.0))
    x = u * (W - 1)
    y = v * (H - 1)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)
    fx = x - x0
    fy = y - y0
    return ((1 - fx) * (1 - fy) * texture[y0, x0]
            + fx * (1 - fy) * texture[y0, x1]
            + (1 - fx) * fy * texture[y1, x0]
            + fx * fy * texture[y1, x1])


def _iter_shared_edges(mesh):
    """Yield (face_a, face_b, v_i, v_j) for every edge-adjacent face pair."""
    for fa, neighbors in enumerate(mesh.face_adjacency):
        for fb in neighbors:
            if fb <= fa:
                continue
            shared = [v for v in mesh.faces[fa] if v in mesh.faces[fb]]
            assert len(shared) == 2, (
                f"faces {fa},{fb} share {len(shared)} vertices, expected 2"
            )
            yield fa, fb, shared[0], shared[1]


def _edge_uvs(mesh, face_idx: int, v_i: int, v_j: int):
    """UV endpoints of edge (v_i, v_j) in `face_idx`'s tangent-plane parameterization."""
    face = mesh.faces[face_idx]
    ia = face.index(v_i)
    ib = face.index(v_j)
    return mesh._face_vertex_uvs[face_idx][ia], mesh._face_vertex_uvs[face_idx][ib]


class TestFaceElevationsFromTexture:
    def test_output_shape_and_dtype(self):
        tex = np.zeros((5, 8, 8, 1), dtype=np.float32)
        result = face_elevations_from_texture(tex)
        assert result.shape == (5,)
        assert result.dtype == np.float64

    def test_constant_texture_returns_that_value(self):
        tex = np.full((3, 4, 4, 1), 2.5, dtype=np.float32)
        result = face_elevations_from_texture(tex)
        np.testing.assert_allclose(result, 2.5)

    def test_mean_across_spatial_dims(self):
        tex = np.zeros((2, 2, 2, 1), dtype=np.float32)
        tex[0, :, :, 0] = [[1.0, 3.0], [5.0, 7.0]]  # mean = 4.0
        tex[1, :, :, 0] = [[0.0, 0.0], [0.0, 2.0]]  # mean = 0.5
        result = face_elevations_from_texture(tex)
        np.testing.assert_allclose(result, [4.0, 0.5])

    def test_per_face_values_are_independent(self):
        rng = np.random.default_rng(0)
        tex = rng.standard_normal((10, 6, 6, 1)).astype(np.float32)
        result = face_elevations_from_texture(tex)
        for i in range(10):
            expected = float(tex[i, :, :, 0].mean())
            assert abs(result[i] - expected) < 1e-5, f"face {i} mismatch"


class TestNoiseShape:
    def test_shape_and_dtype(self):
        mesh = Mesh.icosphere(1).mesh_dual()
        result = compute_noise_elevation_deltas(mesh, tex_size=8, seed=42)
        assert result.shape == (len(mesh.faces), 8, 8, 1)
        assert result.dtype == np.float32

    def test_channel_populated(self):
        mesh = Mesh.icosphere(1).mesh_dual()
        result = compute_noise_elevation_deltas(mesh, tex_size=8, seed=0)
        assert np.any(result[:, :, :, 0] != 0.0)


class TestNoiseDeterminism:
    def test_same_seed_same_result(self):
        mesh = Mesh.icosphere(1).mesh_dual()
        a = compute_noise_elevation_deltas(mesh, tex_size=8, seed=7)
        b = compute_noise_elevation_deltas(mesh, tex_size=8, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_result(self):
        mesh = Mesh.icosphere(1).mesh_dual()
        a = compute_noise_elevation_deltas(mesh, tex_size=8, seed=1)
        b = compute_noise_elevation_deltas(mesh, tex_size=8, seed=2)
        assert not np.array_equal(a, b)


class TestNoiseAmplitude:
    def test_within_amplitude_bound(self):
        mesh = Mesh.icosphere(1).mesh_dual()
        amp = 1.5
        result = compute_noise_elevation_deltas(
            mesh, tex_size=16, amplitude=amp, seed=0
        )
        # Normalized fBm output is roughly in [-1, 1]; allow a small slack.
        assert np.max(np.abs(result[:, :, :, 0])) <= amp * 1.05

    def test_amplitude_scales_linearly(self):
        mesh = Mesh.icosphere(1).mesh_dual()
        a = compute_noise_elevation_deltas(mesh, tex_size=8, amplitude=1.0, seed=3)
        b = compute_noise_elevation_deltas(mesh, tex_size=8, amplitude=2.0, seed=3)
        np.testing.assert_allclose(b[:, :, :, 0], 2.0 * a[:, :, :, 0], atol=1e-5)


class TestPositionContinuity:
    """Underlying invariant for seamless noise: face_texture_positions must give
    the same 3D point on either side of a shared edge."""

    def test_positions_match_at_shared_vertices(self):
        # At exact polygon corners both faces should agree (pixel-perfect).
        mesh = Mesh.icosphere(1).mesh_dual()
        H = W = 32
        positions = mesh.face_texture_positions(H, W)
        max_err = 0.0
        for fa, fb, vi, vj in _iter_shared_edges(mesh):
            uv_a_i, _ = _edge_uvs(mesh, fa, vi, vj)
            uv_b_i, _ = _edge_uvs(mesh, fb, vi, vj)
            p_a = _bilinear_sample(positions[fa], uv_a_i)
            p_b = _bilinear_sample(positions[fb], uv_b_i)
            max_err = max(max_err, float(np.linalg.norm(p_a - p_b)))
        # Shared vertices are exact pixel positions on the [0,1]² grid corners
        # in tangent-plane param space, so tolerance can be tight.
        assert max_err < 1e-4, f"vertex position discrepancy: {max_err:.6f}"

    def test_positions_match_along_shared_edge_interior(self):
        mesh = Mesh.icosphere(1).mesh_dual()
        H = W = 32
        positions = mesh.face_texture_positions(H, W)
        ts = np.array([0.13, 0.27, 0.41, 0.59, 0.73, 0.87])
        max_err = 0.0
        worst = None
        for fa, fb, vi, vj in _iter_shared_edges(mesh):
            uv_a_i, uv_a_j = _edge_uvs(mesh, fa, vi, vj)
            uv_b_i, uv_b_j = _edge_uvs(mesh, fb, vi, vj)
            for t in ts:
                uv_a = (1 - t) * uv_a_i + t * uv_a_j
                uv_b = (1 - t) * uv_b_i + t * uv_b_j
                p_a = _bilinear_sample(positions[fa], uv_a)
                p_b = _bilinear_sample(positions[fb], uv_b)
                err = float(np.linalg.norm(p_a - p_b))
                if err > max_err:
                    max_err = err
                    worst = (fa, fb, float(t))
        # Faces in an icosphere(1) dual have angular radius ~30°; sub-percent
        # absolute distance on the unit sphere ≈ ~0.5° angular agreement.
        assert max_err < 0.01, (
            f"edge-interior position discrepancy {max_err:.4f} at "
            f"faces/t={worst}"
        )


class TestNoiseContinuity:
    """The actual seam test: bilinear-sampled noise values must agree along
    shared edges between adjacent faces.

    Tolerances are chosen for a depth-2 dual mesh (320 dual faces) — coarser
    than production (depth-4, 2562 faces) so the test runs fast, but the
    residual seam after band-limiting scales down further on finer meshes
    (~3 mm on depth-4). Tolerance budget here is the visibility threshold
    relative to the regression we're guarding against (~45 mm pre-band-limit).
    """

    SEAM_TOL_KM = 0.025  # well under the ~45 mm visible seam from the original bug

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_noise_matches_at_shared_vertices(self, seed):
        mesh = Mesh.icosphere(2).mesh_dual()
        H = 32
        amplitude = 1.5
        noise = compute_noise_elevation_deltas(
            mesh, tex_size=H, amplitude=amplitude, seed=seed
        )[:, :, :, 0]
        max_diff = 0.0
        for fa, fb, vi, vj in _iter_shared_edges(mesh):
            uv_a_i, _ = _edge_uvs(mesh, fa, vi, vj)
            uv_b_i, _ = _edge_uvs(mesh, fb, vi, vj)
            v_a = float(_bilinear_sample(noise[fa], uv_a_i))
            v_b = float(_bilinear_sample(noise[fb], uv_b_i))
            max_diff = max(max_diff, abs(v_a - v_b))
        assert max_diff < self.SEAM_TOL_KM, (
            f"vertex noise discrepancy too large: {max_diff:.4f} km"
        )

    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_noise_matches_along_shared_edge_interior(self, seed):
        mesh = Mesh.icosphere(2).mesh_dual()
        H = 32
        amplitude = 1.5
        noise = compute_noise_elevation_deltas(
            mesh, tex_size=H, amplitude=amplitude, seed=seed
        )[:, :, :, 0]
        ts = np.array([0.13, 0.27, 0.41, 0.59, 0.73, 0.87])
        max_diff = 0.0
        worst = None
        for fa, fb, vi, vj in _iter_shared_edges(mesh):
            uv_a_i, uv_a_j = _edge_uvs(mesh, fa, vi, vj)
            uv_b_i, uv_b_j = _edge_uvs(mesh, fb, vi, vj)
            for t in ts:
                uv_a = (1 - t) * uv_a_i + t * uv_a_j
                uv_b = (1 - t) * uv_b_i + t * uv_b_j
                v_a = float(_bilinear_sample(noise[fa], uv_a))
                v_b = float(_bilinear_sample(noise[fb], uv_b))
                diff = abs(v_a - v_b)
                if diff > max_diff:
                    max_diff = diff
                    worst = (fa, fb, float(t))
        assert max_diff < self.SEAM_TOL_KM, (
            f"edge-interior noise discrepancy {max_diff:.4f} km at "
            f"faces/t={worst} (amplitude={amplitude})"
        )


class TestElevationTextureContinuity:
    """Elevation texture values must agree at shared edges.

    compute_elevation_textures stores absolute elevation (not a delta), so we
    compare the sampled texture value directly on both sides of each shared edge.
    """

    # Vertex tolerance accounts for bilinear sampling error (UV corners don't land
    # exactly on pixel centers in a tex_size=32 grid).
    # Edge tolerance reflects the near-zero residual from barycentric interpolation
    # (measured ~0.001 km; pre-fix UV-space IDW reached ~1.8 km).
    VERTEX_TOL_KM = 0.1
    EDGE_TOL_KM = 0.01

    def _make_test_case(self, seed=42):
        mesh = Mesh.icosphere(2).mesh_dual()
        rng = np.random.default_rng(seed)
        elevations = rng.uniform(-5.0, 5.0, size=len(mesh.faces))
        return mesh, elevations

    def test_elevation_matches_at_shared_vertices(self):
        mesh, elevations = self._make_test_case()
        tex_size = 32
        textures = compute_elevation_textures(
            mesh, elevations, tex_size=tex_size
        )[:, :, :, 0]

        max_diff = 0.0
        for fa, fb, vi, vj in _iter_shared_edges(mesh):
            uv_a_i, _ = _edge_uvs(mesh, fa, vi, vj)
            uv_b_i, _ = _edge_uvs(mesh, fb, vi, vj)
            elev_a = float(_bilinear_sample(textures[fa], uv_a_i))
            elev_b = float(_bilinear_sample(textures[fb], uv_b_i))
            max_diff = max(max_diff, abs(elev_a - elev_b))

        assert max_diff < self.VERTEX_TOL_KM, (
            f"elevation mismatch at shared vertices: {max_diff:.4f} km"
        )

    def test_elevation_matches_along_shared_edge(self):
        mesh, elevations = self._make_test_case()
        tex_size = 32
        textures = compute_elevation_textures(
            mesh, elevations, tex_size=tex_size
        )[:, :, :, 0]

        ts = np.array([0.13, 0.27, 0.41, 0.59, 0.73, 0.87])
        max_diff = 0.0
        worst = None
        for fa, fb, vi, vj in _iter_shared_edges(mesh):
            uv_a_i, uv_a_j = _edge_uvs(mesh, fa, vi, vj)
            uv_b_i, uv_b_j = _edge_uvs(mesh, fb, vi, vj)
            for t in ts:
                uv_a = (1 - t) * uv_a_i + t * uv_a_j
                uv_b = (1 - t) * uv_b_i + t * uv_b_j
                elev_a = float(_bilinear_sample(textures[fa], uv_a))
                elev_b = float(_bilinear_sample(textures[fb], uv_b))
                diff = abs(elev_a - elev_b)
                if diff > max_diff:
                    max_diff = diff
                    worst = (fa, fb, float(t))

        assert max_diff < self.EDGE_TOL_KM, (
            f"elevation mismatch along shared edge: {max_diff:.4f} km at {worst}"
        )


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class TestOceanMaskComponents:
    """Verify every connected component of the ocean mask reaches genuine ocean depth.

    Rationale: the flood-fill in compute_ocean_mask_texture seeds from faces whose
    max pixel elevation is at or below sea level (0 km). A correctly-functioning
    mask should therefore contain no spurious component that exists entirely in
    the [-0.5, 0] km shallow band — every real ocean component must include at
    least one pixel below -0.5 km. The connectivity used here is independent of
    the flood-fill implementation: intra-face 4-connectivity plus exact
    pixel-pixel cross-face matching from _iter_boundary_pixel_pairs.
    """

    def test_each_component_reaches_deep_ocean(self):
        params = ElevationParams()
        params.seed = 72264827
        builder = ElevationBuilder(params)
        builder.build(BuildContext(world=World()))

        mesh = builder.mesh
        elev = builder.elevation_textures[:, :, :, 0]   # (N, H, W)
        ocean = builder.ocean_mask[:, :, :, 0] > 0.5     # (N, H, W) bool
        N, H, W = ocean.shape
        tex_size = H

        def node_id(f: int, r: int, c: int) -> int:
            return (f * H + r) * W + c

        uf = _UnionFind(N * H * W)

        # Intra-face 4-connectivity over ocean pixels.
        ocean_f, ocean_r, ocean_c = np.where(ocean)
        for f, r, c in zip(ocean_f.tolist(), ocean_r.tolist(), ocean_c.tolist()):
            if r + 1 < H and ocean[f, r + 1, c]:
                uf.union(node_id(f, r, c), node_id(f, r + 1, c))
            if c + 1 < W and ocean[f, r, c + 1]:
                uf.union(node_id(f, r, c), node_id(f, r, c + 1))

        # Cross-face: exact pixel-pixel partners along shared mesh edges.
        for fa, fb, ra, ca, rb, cb in mesh.iter_boundary_pixel_pairs(tex_size):
            if ocean[fa, ra, ca] and ocean[fb, rb, cb]:
                uf.union(node_id(fa, ra, ca), node_id(fb, rb, cb))

        # Bucket ocean pixels by component root and track min elevation per root.
        component_min_elev: dict[int, float] = {}
        component_size: dict[int, int] = {}
        component_example: dict[int, tuple[int, int, int]] = {}
        for f, r, c in zip(ocean_f.tolist(), ocean_r.tolist(), ocean_c.tolist()):
            root = uf.find(node_id(f, r, c))
            e = float(elev[f, r, c])
            if root not in component_min_elev or e < component_min_elev[root]:
                component_min_elev[root] = e
                component_example[root] = (f, r, c)
            component_size[root] = component_size.get(root, 0) + 1

        # Every component must contain at least one pixel below -0.5 km.
        SEED_ELEVATION = -0.5
        offenders = [
            (root, component_min_elev[root], component_size[root], component_example[root])
            for root in component_min_elev
            if component_min_elev[root] >= SEED_ELEVATION
        ]
        if offenders:
            lines = [
                f"{len(offenders)} ocean component(s) never drop below {SEED_ELEVATION} km:"
            ]
            for root, min_e, size, (f, r, c) in offenders[:10]:
                lines.append(
                    f"  component @ face={f} (r={r},c={c}) size={size} px "
                    f"min_elev={min_e:.3f} km"
                )
            pytest.fail("\n".join(lines))
