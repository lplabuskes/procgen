"""Lightweight checks on the realm pipeline stage protocol surface (no world build)."""
import numpy as np

from core.display import DisplayRegistry
from core.pipeline import Stage
from realmgen.builder import RealmParams, RealmPipeline
from realmgen.placement.builder import PlacementStage
from realmgen.polity.builder import PolityStage
from realmgen.population.builder import PopulationStage
from realmgen.refine.builder import RefineStage
from realmgen.scoring.builder import ScoringStage
from realmgen.towns.builder import TownStage
from realmgen.channels import (
    BORDER_TENSION,
    FORT_POPULATION,
    HINTERLAND_ID,
    POLITY_ID,
    POPULATION,
    SITUATIONAL_VALUE,
    SUITABILITY,
    TRADE_FLOW,
)
from realmgen.trade.network import TradeResult


def test_all_stages_are_stage_subclasses():
    assert issubclass(ScoringStage, Stage)
    assert issubclass(PlacementStage, Stage)
    assert issubclass(PopulationStage, Stage)
    assert issubclass(PolityStage, Stage)


def test_stage_channels():
    p = RealmParams()
    scoring = ScoringStage(p.suitability, p.situational)
    assert SUITABILITY in scoring.provides
    assert SITUATIONAL_VALUE in scoring.provides

    population = PopulationStage(p.trade, p.population, p.seeder)
    assert TRADE_FLOW in population.provides
    assert POPULATION in population.provides
    assert HINTERLAND_ID in population.provides

    polity = PolityStage(p.polity)
    assert POLITY_ID in polity.provides
    assert BORDER_TENSION in polity.provides
    assert FORT_POPULATION in polity.provides


def test_realm_pipeline_has_six_stages():
    rp = RealmPipeline()
    assert len(rp.stages) == 6
    assert isinstance(rp.stages[0], ScoringStage)
    assert isinstance(rp.stages[1], PlacementStage)
    assert isinstance(rp.stages[2], PopulationStage)
    assert isinstance(rp.stages[3], PolityStage)
    assert isinstance(rp.stages[4], RefineStage)
    assert isinstance(rp.stages[5], TownStage)


def test_scoring_register_display_gated_on_results():
    p = RealmParams()
    stage = ScoringStage(p.suitability, p.situational)
    reg = DisplayRegistry()
    stage.register_display(None, reg)  # type: ignore[arg-type]
    assert reg.schemes == {}

    stage.suitability = np.array([0.5], dtype=np.float32)
    stage.situational_value = np.array([0.5], dtype=np.float32)
    reg2 = DisplayRegistry()
    stage.register_display(None, reg2, debug=True)  # type: ignore[arg-type]
    assert set(reg2.schemes) == {"Suitability", "Situational Value"}


def test_population_register_display_gated_on_results():
    p = RealmParams()
    stage = PopulationStage(p.trade, p.population, p.seeder)
    reg = DisplayRegistry()
    stage.register_display(None, reg)  # type: ignore[arg-type]
    assert reg.schemes == {}
    assert reg.overlays == []

    face_flow = np.zeros(4, dtype=np.float32)
    stage.trade_result = TradeResult(face_flow=face_flow, hub_mask=np.zeros(4, dtype=bool), routes=[])
    stage.settlements = []
    stage.hinterland = np.full(4, -1, dtype=np.int32)
    reg2 = DisplayRegistry()
    stage.register_display(None, reg2, debug=True)  # type: ignore[arg-type]
    assert "Trade Flow" in reg2.schemes
    assert "Hinterland" in reg2.schemes
    assert any(ov.feature == "settlements" for ov in reg2.overlays)


def test_polity_register_display_gated_on_results():
    p = RealmParams()
    stage = PolityStage(p.polity)
    reg = DisplayRegistry()
    stage.register_display(None, reg)  # type: ignore[arg-type]
    assert reg.schemes == {}

    from realmgen.polity.types import Polity
    stage.polities = [
        Polity(polity_id=0, capital_face=0, capital_position=np.array([1.0, 0.0, 0.0]),
               power=10.0, n_faces=3, member_settlement_faces=[0]),
    ]
    stage.face_tension = np.zeros(4, dtype=np.float64)
    reg2 = DisplayRegistry()
    stage.register_display(None, reg2, debug=True)  # type: ignore[arg-type]
    assert "Polities" in reg2.schemes
    assert "Border Tension" in reg2.schemes
    assert not any(ov.feature == "forts" for ov in reg2.overlays)

    from realmgen.polity.forts import Forts
    stage.forts = Forts(
        positions=np.array([[1.0, 0.0, 0.0]]), face_id=np.array([2], dtype=np.int32),
        facing=np.array([3], dtype=np.int32), kind=np.array([0], dtype=np.int8),
        score=np.array([0.5]), garrison=np.array([300.0]),
    )
    reg3 = DisplayRegistry()
    stage.register_display(None, reg3)  # type: ignore[arg-type]
    assert any(ov.feature == "forts" and ov.checked for ov in reg3.overlays)
