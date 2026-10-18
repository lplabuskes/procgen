"""Site params (``SettlementMapParams``: ``rivers``, ``roads``, ``footprint``, ``anchors``)."""
from __future__ import annotations

from core.params import param_field as _f, paramclass
from localmap.rivers import LocalRiverParams
from localmap.roads import LocalRoadParams


@paramclass
class SettlementRiverParams(LocalRiverParams):
    """:class:`LocalRiverParams` at settlement scale (raster in settlement texels, corridor in local texels)."""

    # ~45 m: the 15 m raster keeps the router's 45° turns; this and 12 passes smooth them out.
    raster_texels: float = _f(3.0, label="River raster spacing (texels)", min=0.5, max=8.0, step=0.5)
    # Past the frame, in frame units (the frame is 1).
    margin: float = _f(0.02, label="River raster margin", min=0.0, max=0.2, step=0.01)
    corridor_parent_texels: float = _f(
        1.0, label="Corridor half-width (local texels)", min=0.25, max=3.0, step=0.25)
    smoothing_iterations: int = _f(12, label="Path smoothing iterations", min=0, max=40)
    # The channel's shape (natural_channels).
    width_wander: float = _f(0.15, label="Channel width wander (lognormal σ)", min=0.0, max=0.6, step=0.01)
    bank_wander: float = _f(0.1, label="Channel bank wander (× width)", min=0.0, max=0.5, step=0.01)
    wander_widths: float = _f(4.0, label="Channel wander length (× width)", min=0.5, max=20.0, step=0.5)
    island_min_width_m: float = _f(60.0, label="Narrowest river with islands (m)", min=5.0, max=1000.0, step=5.0)
    islands_per_km: float = _f(0.6, label="Islands per km of river", min=0.0, max=5.0, step=0.05)
    # Mean island breadth (× the river's width) and length (× its breadth).
    island_width: float = _f(0.3, label="Island breadth (× river width)", min=0.05, max=2.0, step=0.05)
    island_length: float = _f(7.0, label="Island length (× breadth)", min=1.5, max=20.0, step=0.5)
    # The channel widens round an island by this share of its breadth (1: both arms keep the river's width).
    island_widening: float = _f(0.5, label="Channel widening at an island (× breadth)", min=0.0, max=1.0, step=0.05)
    # Land between a channel and the water it runs into narrower than this goes under water.
    min_land_m: float = _f(75.0, label="Narrowest land beside a channel (m)", min=0.0, max=500.0, step=5.0)
    # The channel's course is the line smoothed over this many widths.
    course_smoothing: float = _f(1.0, label="Channel course smoothing (× width)", min=0.0, max=5.0, step=0.1)


@paramclass
class SettlementRoadParams(LocalRoadParams):
    """:class:`LocalRoadParams` at settlement scale (raster in settlement texels, corridor in
    local texels), plus where sea legs land and which crossings are bridged."""

    # ~45 m, as the rivers'.
    raster_texels: float = _f(3.0, label="Road raster spacing (texels)", min=1.0, max=8.0, step=0.5)
    margin: float = _f(0.02, label="Road raster margin", min=0.0, max=0.2, step=0.01)
    corridor_parent_texels: float = _f(
        1.0, label="Corridor half-width (local texels)", min=0.25, max=3.0, step=0.25)
    smoothing_iterations: int = _f(8, label="Road smoothing iterations", min=0, max=20)
    # The region puts a sea leg's coast end on its ~9 km raster: the shore can be a way off.
    landing_reach_km: float = _f(2.0, label="Landing reach (km)", min=0.1, max=10.0, step=0.1)
    bridge_width_m: float = _f(20.0, label="Bridge from channel width (m)", min=0.0, max=500.0, step=1.0)
    # A river here is many raster nodes wide: roads cross it the short way and keep to its banks.
    # Each step along a channel costs as much as climbing this slope (km/km): the
    # ruggedness of the street-scale relief would otherwise send roads down the flat river bed.
    channel_climb: float = _f(1.0, label="Channel step cost (as slope)", min=0.0, max=5.0, step=0.1)
    bank_m: float = _f(60.0, label="River bank band (m)", min=0.0, max=500.0, step=5.0)
    water_corridor_parent_texels: float = _f(
        1.5, label="Water corridor half-width (local texels)", min=0.25, max=6.0, step=0.25)


@paramclass
class SettlementFootprintParams:
    """Tuning for the density, footprint and buildable ground."""

    density_per_ha: float = _f(70.0, label="Built-up density at the reference size (people/ha)",
                               min=5.0, max=500.0, step=5.0)
    density_reference_population: float = _f(1000.0, label="Density reference population",
                                             min=100.0, max=100_000.0, step=100.0)
    density_size_exponent: float = _f(0.2, label="Density size exponent", min=0.0, max=1.0, step=0.05)
    walled_density_mult: float = _f(1.3, label="Walled density (×)", min=1.0, max=3.0, step=0.05)
    density_jitter: float = _f(0.3, label="Density jitter (lognormal σ)", min=0.0, max=1.0, step=0.05)
    constraint_exponent: float = _f(0.5, label="Hemmed-in density exponent", min=0.0, max=2.0, step=0.05)
    constraint_max_mult: float = _f(2.0, label="Hemmed-in density max (×)", min=1.0, max=5.0, step=0.1)
    min_density_per_ha: float = _f(10.0, label="Min density (people/ha)", min=1.0, max=100.0, step=1.0)
    max_density_per_ha: float = _f(450.0, label="Max density (people/ha)", min=50.0, max=2000.0, step=10.0)
    road_pull: float = _f(0.3, label="Road pull", min=0.0, max=2.0, step=0.05)
    road_pull_m: float = _f(120.0, label="Road pull reach (m)", min=10.0, max=1000.0, step=10.0)
    water_pull: float = _f(0.2, label="Waterfront pull", min=0.0, max=2.0, step=0.05)
    water_pull_m: float = _f(150.0, label="Waterfront pull reach (m)", min=10.0, max=1000.0, step=10.0)
    edge_roughness: float = _f(0.5, label="Footprint edge roughness", min=0.0, max=3.0, step=0.05)
    # Correlation length of the edge noise.
    roughness_m: float = _f(100.0, label="Edge roughness scale (m)", min=5.0, max=500.0, step=5.0)
    slope_free: float = _f(0.05, label="Footprint slope free (km/km)", min=0.0, max=1.0, step=0.01)
    # The slope the penalty reads, smoothed: hillsides, not the street-scale bumps.
    slope_smoothing_m: float = _f(120.0, label="Footprint slope smoothing (m)", min=0.0, max=500.0, step=5.0)
    # Detached pieces smaller than this share of the footprint go back to its main body.
    min_piece_fraction: float = _f(0.03, label="Smallest footprint piece (share)", min=0.0, max=0.5, step=0.01)
    slope_weight: float = _f(2.0, label="Footprint slope penalty", min=0.0, max=50.0, step=0.5)
    hole_density_mult: float = _f(0.3, label="Enclosed-hole density (×)", min=0.0, max=1.0, step=0.05)
    hole_min_enclosed: float = _f(0.25, label="Hole edge on the footprint (share)", min=0.0, max=1.0, step=0.05)
    # Concavities narrower than a disc this share of the footprint's radius are filled (sparsely built).
    compact: float = _f(0.15, label="Footprint concavity fill (× radius)", min=0.0, max=1.0, step=0.01)
    # Each metre of water on the way from the centre counts this many metres of distance.
    crossing_cost: float = _f(3.0, label="Water crossing cost (× distance)", min=1.0, max=20.0, step=0.5)
    island_min_ha: float = _f(2.0, label="Smallest island built on (ha)", min=0.0, max=100.0, step=0.5)
    buildable_max_slope: float = _f(0.3, label="Buildable max slope (km/km)", min=0.01, max=2.0, step=0.01)


@paramclass
class SettlementAnchorParams:
    """Tuning for the anchor candidates."""

    per_kind: int = _f(6, label="Candidates per kind", min=1, max=30)
    separation_m: float = _f(150.0, label="Candidate separation (m)", min=10.0, max=1000.0, step=10.0)
    smoothing_m: float = _f(30.0, label="Relief smoothing (m)", min=0.0, max=200.0, step=5.0)
    surround_m: float = _f(300.0, label="High-ground surround (m)", min=50.0, max=2000.0, step=25.0)
    sample_m: float = _f(30.0, label="Road / shore sample spacing (m)", min=5.0, max=200.0, step=5.0)
    market_reach_m: float = _f(75.0, label="Market road reach (m)", min=10.0, max=500.0, step=5.0)
    waterfront_reach_m: float = _f(25.0, label="Waterfront reach (m)", min=1.0, max=200.0, step=1.0)
    # Slope (km/km) at which ground stops counting as flat.
    flat_slope: float = _f(0.1, label="Flat-ground slope (km/km)", min=0.01, max=1.0, step=0.01)
