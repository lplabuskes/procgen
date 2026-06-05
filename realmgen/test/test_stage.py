"""Lightweight checks on the RealmStage protocol surface (no world build)."""
import numpy as np

from core.display import DisplayRegistry
from core.pipeline import Stage
from realmgen.builder import RealmStage
from realmgen.channels import SITUATIONAL_VALUE, SUITABILITY


def test_realm_stage_is_a_stage():
    assert issubclass(RealmStage, Stage)
    s = RealmStage()
    assert s.name == "realm"
    assert SUITABILITY in s.provides and SITUATIONAL_VALUE in s.provides


def test_register_display_is_gated_on_computed_results():
    stage = RealmStage()
    reg = DisplayRegistry()
    # Nothing computed yet → no schemes contributed.
    stage.register_display(None, reg)
    assert reg.schemes == {}
    # Once results exist, both schemes are contributed.
    stage.suitability = np.array([0.5], dtype=np.float32)
    stage.situational_value = np.array([0.5], dtype=np.float32)
    stage.register_display(None, reg)
    assert set(reg.schemes) == {"Suitability", "Situational Value"}
