"""Unit tests for the core pipeline skeleton: World, channels, BuildContext,
Pipeline, and DisplayRegistry. No Qt — pure logic."""
import numpy as np

from core.channels import FeatureChannel, MetaChannel, TextureChannel
from core.display import DisplayRegistry
from core.pipeline import BuildContext, Pipeline, Stage
from core.world import World


class _FakeMesh:
    """Minimal stand-in exposing the bits the channels touch."""

    def __init__(self, n_faces):
        self.face_metadata = [dict() for _ in range(n_faces)]
        self.face_textures = {}

    def apply_metadata_channel(self, key, values):
        assert len(values) == len(self.face_metadata)
        for i, v in enumerate(values):
            self.face_metadata[i][key] = v

    def set_face_textures(self, name, tex, n_channels=4):
        assert tex.shape[-1] == n_channels
        self.face_textures[name] = tex


# ── World ──────────────────────────────────────────────────────────────────────

def test_world_feature_registry():
    w = World.from_mesh(_FakeMesh(1))
    assert w.mesh is not None
    assert not w.has_feature("rivers")
    w.add_feature("rivers", [1, 2, 3])
    assert w.has_feature("rivers")
    assert w.get_feature("rivers") == [1, 2, 3]
    assert w.get_feature("missing", default=42) == 42
    assert set(w.feature_names()) == {"rivers"}
    w.remove_feature("rivers")
    assert not w.has_feature("rivers")


# ── MetaChannel / TextureChannel (mesh-based) ────────────────────────────────────

def test_meta_channel_round_trip():
    mesh = _FakeMesh(4)
    ch = MetaChannel("fertility", dtype=np.float64)
    assert not ch.present(mesh)
    # Absent → default-filled read.
    np.testing.assert_array_equal(ch.read(mesh), np.zeros(4))
    ch.write(mesh, np.array([0.1, 0.2, 0.3, 0.4]))
    assert ch.present(mesh)
    np.testing.assert_allclose(ch.read(mesh), [0.1, 0.2, 0.3, 0.4])


def test_texture_channel_round_trip():
    mesh = _FakeMesh(2)
    ch = TextureChannel("elevation", n_channels=1)
    assert not ch.present(mesh)
    tex = np.zeros((2, 4, 4, 1), dtype=np.float32)
    tex[1] = 3.0
    ch.write(mesh, tex)
    assert ch.present(mesh)
    np.testing.assert_array_equal(ch.read(mesh), tex)


# ── FeatureChannel (world-based) ─────────────────────────────────────────────────

def test_feature_channel_round_trip():
    w = World.from_mesh(_FakeMesh(1))
    ch = FeatureChannel("rivers")
    assert not ch.present(w)
    assert ch.read(w) is None
    ch.write(w, [1, 2, 3])
    assert ch.present(w)
    assert ch.read(w) == [1, 2, 3]


# ── BuildContext ────────────────────────────────────────────────────────────────

def test_context_mesh_exposes_world_mesh():
    mesh = _FakeMesh(1)
    ctx = BuildContext(world=World.from_mesh(mesh))
    assert ctx.mesh is mesh is ctx.world.mesh


def test_rng_is_deterministic_and_name_separated():
    a = BuildContext(world=World.from_mesh(_FakeMesh(1)), seed=42)
    b = BuildContext(world=World.from_mesh(_FakeMesh(1)), seed=42)
    # Same (seed, name) → identical streams, regardless of context instance.
    assert a.rng("climate").random() == b.rng("climate").random()
    # Order independence: drawing "hydrology" first doesn't change "climate".
    c = BuildContext(world=World.from_mesh(_FakeMesh(1)), seed=42)
    _ = c.rng("hydrology").random()
    assert c.rng("climate").random() == a.rng("climate").random()
    # Different names → different streams.
    assert a.rng("climate").random() != a.rng("hydrology").random()


def test_rng_unseeded_varies_but_self_consistent():
    ctx = BuildContext(world=World.from_mesh(_FakeMesh(1)))  # seed=None → random base
    # Same name twice → same stream within one context.
    assert ctx.rng("x").random() == ctx.rng("x").random()


def test_progress_callback():
    seen = []
    ctx = BuildContext(world=World.from_mesh(_FakeMesh(1)), progress=seen.append)
    ctx.report("hello")
    assert seen == ["hello"]


# ── Pipeline ────────────────────────────────────────────────────────────────────

def test_pipeline_runs_stages_in_order():
    order = []

    class _S(Stage):
        def __init__(self, tag):
            self.name = tag

        def build(self, ctx):
            order.append(self.name)
            ctx.world.add_feature(self.name, True)

    w = World.from_mesh(_FakeMesh(1))
    ctx = BuildContext(world=w)
    Pipeline([_S("a"), _S("b"), _S("c")]).build(ctx)
    assert order == ["a", "b", "c"]
    assert set(w.feature_names()) == {"a", "b", "c"}


def test_stage_default_hooks_are_noops():
    class _S(Stage):
        name = "s"

        def build(self, ctx):
            pass

    s = _S()
    reg = DisplayRegistry()
    assert s.serialize() == {}
    assert s.register_display(World.from_mesh(_FakeMesh(1)), reg, debug=True) is None
    assert s.load({}, BuildContext(world=World.from_mesh(_FakeMesh(1)))) is None


# ── DisplayRegistry ─────────────────────────────────────────────────────────────

def test_display_registry_accumulates():
    reg = DisplayRegistry()
    reg.add_scheme("Elevation", lambda m: (0, 0, 0, 1), texture="elevation_rgba",
                   legend=("gradient", []))
    reg.add_overlay("rivers", "Rivers", checked=False)
    assert "Elevation" in reg.schemes
    assert reg.schemes["Elevation"].texture == "elevation_rgba"
    assert reg.overlays[0].feature == "rivers"
    assert reg.overlays[0].checked is False


def test_display_registry_hover_concatenates():
    reg = DisplayRegistry()
    reg.add_hover(lambda f, uv, w: [f"biome:{f}"])
    reg.add_hover(lambda f, uv, w: ["wet"] if f == 1 else [])
    assert reg.hover(0, None, None) == ["biome:0"]
    assert reg.hover(1, None, None) == ["biome:1", "wet"]
