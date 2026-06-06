from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.mesh import Mesh, PolygonMesh
from core.params import param_field as _f, paramclass

_CACHE_DIR = Path(__file__).parent.parent.parent / "mesh_cache"
_N_CACHED = 5


@paramclass
class MeshParams:
    """Knobs for generating the shared mesh cache (icosphere + deregularization)."""
    icosphere_depth: int = _f(5, label="Icosphere depth", min=1, max=8)
    n_swaps: int = _f(4000, label="Edge swaps", min=0, max=10000)
    n_relax: int = _f(10, label="Relaxation steps", min=0, max=100)
    stiffness: float = _f(0.1, label="Stiffness", min=0.01, max=1.0)
    outer_iters: int = _f(3, label="Outer iterations", min=1, max=20)


def _mesh_path(idx: int, cache_dir: Path) -> Path:
    return cache_dir / f"mesh_{idx}.npz"


def _save(mesh: PolygonMesh, path: Path) -> None:
    face_sizes = np.array([len(f) for f in mesh.faces], dtype=np.int32)
    face_data = np.array([v for f in mesh.faces for v in f], dtype=np.int32)
    np.savez_compressed(path, vertices=mesh.vertices, face_sizes=face_sizes, face_data=face_data)


def _load(path: Path) -> PolygonMesh:
    data = np.load(path)
    face_sizes = data["face_sizes"].tolist()
    flat = data["face_data"].tolist()
    faces = []
    offset = 0
    for size in face_sizes:
        faces.append(tuple(flat[offset:offset + size]))
        offset += size
    return PolygonMesh(data["vertices"], faces)


def is_populated(cache_dir: Path = _CACHE_DIR, n: int = _N_CACHED) -> bool:
    return all(_mesh_path(i, cache_dir).exists() for i in range(n))


def build(
    params: MeshParams,
    cache_dir: Path = _CACHE_DIR,
    n: int = _N_CACHED,
    progress: Optional[Callable[[str], None]] = None,
) -> None:
    """Generate n random meshes and write them to cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        if progress:
            progress(f"Caching mesh {i + 1}/{n}…")
        tri = Mesh.icosphere(params.icosphere_depth).deregularize(
            n_swaps=params.n_swaps,
            n_relax=params.n_relax,
            stiffness=params.stiffness,
            outer_iters=params.outer_iters,
        )
        _save(tri.mesh_dual(), _mesh_path(i, cache_dir))


def load_random(seed: Optional[int] = None, cache_dir: Path = _CACHE_DIR, n: int = _N_CACHED) -> PolygonMesh:
    idx = (seed % n) if seed is not None else random.randrange(n)
    return _load(_mesh_path(idx, cache_dir))
