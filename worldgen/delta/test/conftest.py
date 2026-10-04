"""Shared fixtures: a depth-4 globe and its texel index."""
from __future__ import annotations

import pytest

from core.mesh import Mesh
from worldgen.delta.deposit import TexelIndex

TEX = 16


@pytest.fixture(scope="module")
def mesh():
    return Mesh.icosphere(4).mesh_dual()


@pytest.fixture(scope="module")
def index(mesh):
    return TexelIndex(mesh, TEX, TEX)
