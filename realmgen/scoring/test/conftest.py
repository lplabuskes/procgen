import pytest

from core.pipeline import BuildContext
from core.world import World
from realmgen.builder import RealmPipeline
from worldgen import GlobePipeline, GlobeParams


@pytest.fixture(scope="session")
def scoring_globe():
    p = GlobeParams()
    p.seed = 7
    ctx = BuildContext(world=World(p.mesh, seed=p.seed))
    GlobePipeline(p).build(ctx)
    return ctx.world


@pytest.fixture(scope="session")
def scoring_realm(scoring_globe):
    realm = RealmPipeline()
    realm.build(BuildContext(world=scoring_globe))
    return realm
