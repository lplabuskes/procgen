"""Unit tests for realmgen.refine.fill — speck removal."""
from __future__ import annotations

import numpy as np


# ── specks ──────────────────────────────────────────────────────────────────


def _lattice_with_speck():
    from realmgen.refine.borders import _icosphere_lattice
    verts, _, edges, _ = _icosphere_lattice(3)
    labels = np.where(verts[:, 2] > 0, 0, 1).astype(np.int64)    # two hemispheres
    speck = int(np.argmax(verts[:, 2]))                           # polity 1 at polity 0's pole
    labels[speck] = 1
    return verts, edges, labels, speck


def test_find_specks_hands_a_tiny_enclave_to_its_surroundings():
    from realmgen.refine.fill import find_specks
    verts, edges, labels, speck = _lattice_with_speck()
    specks = find_specks(verts, edges, labels, max_verts=3)
    assert specks is not None and len(specks) == 1
    out = specks.apply(verts, labels)
    assert out[speck] == 0
    assert np.array_equal(np.delete(out, speck), np.delete(labels, speck))     # nothing else moves
    near = verts[speck] + 0.2 * specks.spacing * np.array([1.0, 0.0, 0.0])
    assert specks.apply(near[None] / np.linalg.norm(near), np.array([1]))[0] == 0   # off-lattice too


def test_find_specks_keeps_settlements_and_islands():
    from realmgen.refine.fill import WATER, find_specks
    verts, edges, labels, speck = _lattice_with_speck()
    assert find_specks(verts, edges, labels, max_verts=3, keep=verts[speck][None]) is None
    nbrs = np.unique(edges[(edges == speck).any(axis=1)])
    island = labels.copy()
    island[nbrs[nbrs != speck]] = WATER                        # ringed by water: an island
    assert find_specks(verts, edges, island, max_verts=3) is None
