"""Realm pipeline — human-geography layer over a fully-built world.

``RealmPipeline`` composes four stages:

1. :class:`~realmgen.scoring.builder.ScoringStage` — suitability + situational value
2. :class:`~realmgen.placement.builder.PlacementStage` — settlement seeding
3. :class:`~realmgen.population.builder.PopulationStage` — trade + population calibration
4. :class:`~realmgen.polity.builder.PolityStage` — territory growth + border tension

Usage::

    from realmgen.builder import RealmParams, RealmPipeline
    from core.pipeline import BuildContext

    pipeline = RealmPipeline(RealmParams())
    pipeline.build(BuildContext(world=globe))
    pipeline.apply_display(window)
"""
from __future__ import annotations

from typing import Optional

from core.display import DisplayRegistry
from core.params import field, paramclass
from core.pipeline import BuildContext, Pipeline
from core.world import World
from realmgen.placement.builder import PlacementStage, SeederParams
from realmgen.placement.types import TIER_NAMES
from realmgen.polity.builder import PolityParams, PolityStage
from realmgen.population.builder import PopulationParams, PopulationStage, TradeParams
from realmgen.scoring.builder import ScoringStage, SituationalValueParams, SuitabilityParams


# ── Top-level params ──────────────────────────────────────────────────────────


@paramclass
class RealmParams:
    suitability: SuitabilityParams = field(
        default_factory=SuitabilityParams, metadata={"label": "Suitability"}
    )
    situational: SituationalValueParams = field(
        default_factory=SituationalValueParams, metadata={"label": "Situational value"}
    )
    seeder: SeederParams = field(
        default_factory=SeederParams, metadata={"label": "Settlement seeder"}
    )
    trade: TradeParams = field(
        default_factory=TradeParams, metadata={"label": "Trade network"}
    )
    population: PopulationParams = field(
        default_factory=PopulationParams, metadata={"label": "Population"}
    )
    polity: PolityParams = field(
        default_factory=PolityParams, metadata={"label": "Polities"}
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────


class RealmPipeline(Pipeline):
    """Four-stage pipeline for the human-geography layer.

    Stages in order: :class:`~realmgen.scoring.builder.ScoringStage` →
    :class:`~realmgen.placement.builder.PlacementStage` →
    :class:`~realmgen.population.builder.PopulationStage` →
    :class:`~realmgen.polity.builder.PolityStage`.  After ``build(ctx)``, each
    stage stores its results as attributes; ``apply_display`` wires all of them
    into a ``MapWindow``.
    """

    def __init__(self, params: RealmParams | None = None) -> None:
        self._params = params or RealmParams()
        p = self._params
        self.scoring_stage = ScoringStage(p.suitability, p.situational)
        self.placement_stage = PlacementStage(p.seeder)
        self.population_stage = PopulationStage(p.trade, p.population, p.seeder)
        self.polity_stage = PolityStage(p.polity)
        super().__init__([
            self.scoring_stage,
            self.placement_stage,
            self.population_stage,
            self.polity_stage,
        ])
        self.world: Optional[World] = None

    @property
    def params(self) -> RealmParams:
        return self._params

    @property
    def suitability(self):
        return self.scoring_stage.suitability

    @property
    def situational_value(self):
        return self.scoring_stage.situational_value

    def build(self, ctx: BuildContext) -> None:
        super().build(ctx)
        self.world = ctx.world

    def apply_display(self, window, debug: bool = False) -> None:
        """Merge realm color schemes, overlays, and hover into a MapWindow after build()."""
        if self.world is None:
            return
        reg = DisplayRegistry()
        for stage in self.stages:
            stage.register_display(self.world, reg, debug=debug)

        if reg.schemes:
            merged = dict(window._map._color_schemes)
            for name, spec in reg.schemes.items():
                merged[name] = spec.fn
            window.set_color_schemes(merged)

        for ov in reg.overlays:
            window.add_overlay(ov.feature, label=ov.label, checked=ov.checked)

        mesh = self.world.mesh if self.world else None
        if mesh is None:
            return

        prev_hover = window._map._hover_callback

        def _realm_hover(face_src, uv=None):
            if prev_hover is not None:
                prev_hover(face_src, uv)
            if face_src is None:
                return
            meta = mesh.face_metadata[face_src]
            if meta.get("is_water", meta.get("is_ocean", False)):
                return
            parts: list[str] = []
            suit = meta.get("suitability")
            if suit is not None:
                parts.append(f"Suit: {float(suit):.2f}")
            tier = meta.get("settlement_tier")
            if tier is not None and int(tier) >= 0:
                t = int(tier)
                parts.append(TIER_NAMES[min(t, len(TIER_NAMES) - 1)])
                pop = meta.get("population")
                if pop is not None and float(pop) > 0:
                    parts.append(f"Pop: {float(pop):.3g}")
            pid = meta.get("polity_id")
            if pid is not None and int(pid) >= 0:
                parts.append(f"Polity {int(pid)}")
                tension = meta.get("border_tension")
                if tension is not None and float(tension) > 0.01:
                    parts.append(f"Tension: {float(tension):.2f}")
            if parts:
                cur = window._status_label.text()
                fragment = "  ".join(parts)
                window.set_status(f"{cur}  |  {fragment}" if cur else fragment)

        window.set_hover_callback(_realm_hover)
