"""Layout params (``SettlementMapParams.layout``)."""
from __future__ import annotations

from core.params import param_field as _f, paramclass


@paramclass
class SettlementLayoutParams:
    """Tuning for the layout."""

    citadel_march: float = _f(0.3, label="March that builds a citadel", min=0.0, max=1.0, step=0.05)
    citadel_wall_strength: float = _f(0.6, label="Wall strength that builds a citadel", min=0.0, max=2.0, step=0.05)
    core_share: float = _f(0.45, label="Old core share of the footprint", min=0.05, max=1.0, step=0.05)
    core_wall_gain: float = _f(0.25, label="Core share per wall strength", min=0.0, max=1.0, step=0.05)
    core_young_gain: float = _f(0.2, label="Core share for a young place", min=0.0, max=1.0, step=0.05)
    young_years: float = _f(300.0, label="Young below (years)", min=10.0, max=3000.0, step=10.0)
    core_wobble: float = _f(0.12, label="Core outline wobble", min=0.0, max=0.5, step=0.01)
    stone_wall_strength: float = _f(0.4, label="Stone walls from strength", min=0.0, max=2.0, step=0.05)
    wall_m: float = _f(5.0, label="Wall width (m)", min=1.0, max=30.0, step=0.5)
    wall_water_m: float = _f(15.0, label="No wall within (m) of water", min=0.0, max=200.0, step=1.0)
    tower_spacing_m: float = _f(70.0, label="Tower spacing at strength 1 (m)", min=10.0, max=500.0, step=5.0)
    # Walls are dear: the line spans notches narrower than this × the core's radius.
    wall_closing: float = _f(0.35, label="Wall spans notches (× core radius)", min=0.0, max=1.0, step=0.05)
    # Open grounds scale with people, within _PLAZA_M2 / _CITADEL_M2 / _PRECINCT_M2.
    plaza_m2_per_person: float = _f(0.08, label="Market plaza (m² per person)", min=0.0, max=2.0, step=0.01)
    citadel_m2_per_person: float = _f(0.15, label="Citadel (m² per person)", min=0.0, max=5.0, step=0.01)
    precinct_m2_per_person: float = _f(0.05, label="Sacred precinct (m² per person)", min=0.0, max=2.0, step=0.01)
    route_step_m: float = _f(20.0, label="Arterial routing grid (m)", min=5.0, max=100.0, step=1.0)
    slope_weight: float = _f(10.0, label="Arterial slope cost", min=0.0, max=100.0, step=1.0)
    water_cost: float = _f(40.0, label="Arterial water cost (×)", min=1.0, max=500.0, step=1.0)
    road_follow: float = _f(0.6, label="Arterial cost along roads (×)", min=0.1, max=1.0, step=0.05)
    outside_cost: float = _f(1.5, label="Arterial cost off the footprint (×)", min=1.0, max=10.0, step=0.1)
    arterial_m: float = _f(10.0, label="Arterial width (m)", min=2.0, max=40.0, step=0.5)
    street_m: float = _f(5.0, label="Street width (m)", min=1.0, max=20.0, step=0.5)
    block_ha: float = _f(0.35, label="Core block (ha)", min=0.02, max=5.0, step=0.01)
    suburb_block_mult: float = _f(2.0, label="Suburb block (×)", min=1.0, max=10.0, step=0.1)
    sparse_block_mult: float = _f(4.0, label="Sparse-ground block (×)", min=1.0, max=20.0, step=0.1)
    block_jitter: float = _f(0.35, label="Block size jitter (lognormal σ)", min=0.0, max=1.0, step=0.05)
    organic_offset: float = _f(0.25, label="Organic cut offset (share)", min=0.0, max=0.5, step=0.01)
    organic_angle_deg: float = _f(14.0, label="Organic cut angle jitter (°)", min=0.0, max=45.0, step=1.0)
    lot_frontage_m: float = _f(8.0, label="Lot frontage (m)", min=2.0, max=50.0, step=0.5)
    lot_depth_m: float = _f(22.0, label="Lot depth (m)", min=5.0, max=100.0, step=1.0)
    suburb_lot_mult: float = _f(1.5, label="Suburb lot (×)", min=1.0, max=5.0, step=0.1)
    garden_lot_mult: float = _f(3.0, label="Garden lot (×)", min=1.0, max=10.0, step=0.1)
    # Quay, market and elite lots run back to fill their blocks, up to this deep (merchant plots).
    merchant_lot_depth_m: float = _f(45.0, label="Quay / market / elite lots up to (m deep)", min=5.0, max=150.0,
                                     step=1.0)
    # Lanes into blocks too deep for their frontage: closes (dead ends) and alleys (through).
    lane_m: float = _f(3.0, label="Close / alley width (m)", min=1.0, max=10.0, step=0.5)
    lane_spacing_block_m: float = _f(60.0, label="Closes along a deep block every (m)", min=15.0, max=500.0,
                                     step=5.0)
    alley_share: float = _f(0.3, label="Share of lanes running through (alleys)", min=0.0, max=1.0, step=0.05)
    elite_share: float = _f(0.12, label="Elite share of core blocks", min=0.0, max=0.5, step=0.01)
    craft_share: float = _f(0.06, label="Craft quarter share of blocks (per weight)", min=0.0, max=0.5, step=0.01)
    quay_m: float = _f(40.0, label="Quay reach from the water (m)", min=5.0, max=200.0, step=5.0)
    graveyard_people: float = _f(20000.0, label="People per graveyard", min=500.0, max=100000.0, step=500.0)
    square_people: float = _f(12000.0, label="People per small square", min=500.0, max=100000.0, step=500.0)
    shrine_people: float = _f(1500.0, label="People per shrine", min=100.0, max=50000.0, step=100.0)
    well_ha: float = _f(1.5, label="Core hectares per well", min=0.1, max=20.0, step=0.1)
    radial_streets: int = _f(3, label="Main streets out of a city's market (at least)", min=0, max=12)
    radial_spacing_m: float = _f(500.0, label="City main streets leave at least this far apart (m)", min=50.0,
                                 max=5000.0, step=25.0)
    # Secondary arterials: through streets filling the gaps between the main ones, and the
    # street inside the walls.
    secondary_m: float = _f(7.0, label="Secondary arterial / wall street width (m)", min=2.0, max=40.0, step=0.5)
    arterial_spacing_m: float = _f(500.0, label="City arterials at most this far apart (m)", min=100.0,
                                   max=5000.0, step=25.0)
    town_arterial_spacing_m: float = _f(800.0, label="Town arterials at most this far apart (m)", min=100.0,
                                        max=5000.0, step=25.0)
    # Towns: a main street through the market, burgage plots along it.
    town_block_mult: float = _f(1.5, label="Town block (×)", min=0.5, max=10.0, step=0.1)
    main_street_gate_deg: float = _f(35.0, label="Main street takes a gate within (°)", min=0.0, max=90.0, step=1.0)
    burgage_depth_m: float = _f(45.0, label="Burgage plot depth (m)", min=10.0, max=150.0, step=1.0)
    burgage_frontage_m: float = _f(7.0, label="Burgage plot frontage (m)", min=2.0, max=30.0, step=0.5)
    lane_spacing_m: float = _f(140.0, label="Cross-lane spacing (m)", min=40.0, max=500.0, step=5.0)
    market_street_m: float = _f(24.0, label="Town market street width (m)", min=5.0, max=80.0, step=1.0)
    # Villages: tofts along a street, round a green or along a few lanes.
    household_people: float = _f(5.0, label="People per household", min=1.0, max=20.0, step=0.5)
    toft_share: float = _f(0.85, label="Tofts' share of a village footprint", min=0.1, max=2.0, step=0.05)
    village_street_m: float = _f(6.0, label="Village street width (m)", min=2.0, max=20.0, step=0.5)
    # Households that don't farm (settlegen.layout.households): everyday trades on small plots,
    # fishers' and miners' cottages; towns and cities spread their trades through the houses.
    trade_people_mult: float = _f(1.0, label="Everyday trades (× per-capita counts)", min=0.0, max=5.0, step=0.1)
    max_trade_share: float = _f(0.35, label="Trades at most this share of a village", min=0.0, max=1.0, step=0.05)
    craft_plot_scale: float = _f(0.4, label="Craft plot (× farm toft)", min=0.05, max=1.0, step=0.05)
    cottage_plot_scale: float = _f(0.35, label="Cottage plot (× farm toft)", min=0.05, max=1.0, step=0.05)
    urban_trade_share: float = _f(0.25, label="Trades at most this share of a town's house plots", min=0.0,
                                  max=1.0, step=0.05)
    # Docks.
    quay_width_m: float = _f(12.0, label="Quay width (m)", min=2.0, max=60.0, step=1.0)
    pier_spacing_m: float = _f(60.0, label="Pier spacing (m)", min=10.0, max=500.0, step=5.0)
    pier_length_m: float = _f(60.0, label="Pier length on still water (m)", min=5.0, max=300.0, step=5.0)
    pier_width_m: float = _f(10.0, label="Pier width (m)", min=2.0, max=40.0, step=1.0)
    pier_min_river_m: float = _f(40.0, label="No piers on rivers narrower than (m)", min=0.0, max=500.0, step=5.0)
