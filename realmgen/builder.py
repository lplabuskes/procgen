"""Realm pipeline — human-geography layer over a fully-built world.

``RealmPipeline`` composes six stages:

1. :class:`~realmgen.scoring.builder.ScoringStage` — suitability + situational value
2. :class:`~realmgen.placement.builder.PlacementStage` — settlement seeding
3. :class:`~realmgen.population.builder.PopulationStage` — trade + population calibration
4. :class:`~realmgen.polity.builder.PolityStage` — territory growth + border tension
5. :class:`~realmgen.refine.builder.RefineStage` — texel-level siting + borders
6. :class:`~realmgen.towns.builder.TownStage` — sub-city towns (display-only)

Usage::

    from realmgen.builder import RealmParams, RealmPipeline
    from core.pipeline import BuildContext

    pipeline = RealmPipeline(RealmParams())
    pipeline.build(BuildContext(world=globe))
    pipeline.apply_display(window)
"""
from __future__ import annotations

from core.params import field, param_field as _f, paramclass
from core.pipeline import Pipeline
from realmgen.placement.builder import PlacementStage, SeederParams
from realmgen.polity.builder import PolityParams, PolityStage
from realmgen.population.builder import PopulationParams, PopulationStage, TradeParams
from realmgen.refine.builder import RefineParams, RefineStage
from realmgen.scoring.builder import ScoringStage, SituationalValueParams, SuitabilityParams
from realmgen.towns.builder import TownParams, TownStage


# ── Top-level params ──────────────────────────────────────────────────────────


@paramclass
class DangerInfluenceParams:
    suitability_weight: float = _f(
        0.5, label="Suitability penalty weight", min=0.0, max=1.0
    )
    block_threshold: float = _f(
        0.65, label="Settlement block threshold", min=0.0, max=1.0
    )
    movement_scale: float = _f(
        2.0, label="Movement cost added (+ danger)", min=0.0, max=20.0
    )


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
    refine: RefineParams = field(
        default_factory=RefineParams, metadata={"label": "Texel refinement"}
    )
    towns: TownParams = field(
        default_factory=TownParams, metadata={"label": "Towns"}
    )
    danger: DangerInfluenceParams = field(
        default_factory=DangerInfluenceParams, metadata={"label": "Danger influence"}
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────


class RealmPipeline(Pipeline):
    """Six-stage pipeline for the human-geography layer.

    Stages in order: :class:`~realmgen.scoring.builder.ScoringStage` →
    :class:`~realmgen.placement.builder.PlacementStage` →
    :class:`~realmgen.population.builder.PopulationStage` →
    :class:`~realmgen.polity.builder.PolityStage` →
    :class:`~realmgen.refine.builder.RefineStage` →
    :class:`~realmgen.towns.builder.TownStage`.  After ``build(ctx)``, each
    stage stores its results as attributes; ``apply_display`` wires all of them
    into a ``MapWindow``.
    """

    def __init__(self, params: RealmParams | None = None) -> None:
        self._params = params or RealmParams()
        p = self._params
        self.scoring_stage = ScoringStage(p.suitability, p.situational, p.danger)
        self.placement_stage = PlacementStage(p.seeder, p.danger)
        self.population_stage = PopulationStage(p.trade, p.population, p.seeder, p.danger, p.polity.defensibility)
        self.polity_stage = PolityStage(p.polity, p.danger)
        self.refine_stage = RefineStage(p.refine)
        self.town_stage = TownStage(p.towns, p.population, p.refine)
        super().__init__([
            self.scoring_stage,
            self.placement_stage,
            self.population_stage,
            self.polity_stage,
            self.refine_stage,
            self.town_stage,
        ])

    @property
    def params(self) -> RealmParams:
        return self._params

    @property
    def suitability(self):
        return self.scoring_stage.suitability

    @property
    def situational_value(self):
        return self.scoring_stage.situational_value
