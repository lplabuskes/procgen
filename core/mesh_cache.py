from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.mesh import Mesh, PolygonMesh

_CACHE_DIR = Path(__file__).parent.parent.parent / "mesh_cache"
_N_CACHED = 5


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
    icosphere_depth: int,
    n_swaps: int,
    n_relax: int,
    stiffness: float,
    outer_iters: int,
    cache_dir: Path = _CACHE_DIR,
    n: int = _N_CACHED,
    progress: Optional[Callable[[str], None]] = None,
) -> None:
    """Generate n random meshes and write them to cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        if progress:
            progress(f"Caching mesh {i + 1}/{n}…")
        tri = Mesh.icosphere(icosphere_depth).deregularize(
            n_swaps=n_swaps,
            n_relax=n_relax,
            stiffness=stiffness,
            outer_iters=outer_iters,
        )
        _save(tri.mesh_dual(), _mesh_path(i, cache_dir))


def load_random(seed: Optional[int] = None, cache_dir: Path = _CACHE_DIR, n: int = _N_CACHED) -> PolygonMesh:
    idx = (seed % n) if seed is not None else random.randrange(n)
    return _load(_mesh_path(idx, cache_dir))
