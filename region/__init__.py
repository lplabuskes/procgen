"""Regional maps: equal-area cubed-sphere subdivision and per-region geometry.

The globe is divided into ``6 · N_SUBDIV**2`` equal-area regions via a gnomonic
cubed-sphere parameterization with a separable equal-area CDF stretch.  This
package owns the (Qt-free) geometry: forward/inverse maps between the sphere and
each region's local ``[0, 1]**2`` coordinate frame.  The local frame is the
foundation for upsampling — sampling global fields at regional resolution.
"""
from region.geometry import (
    N_SUBDIV,
    N_REGIONS,
    RegionGeometry,
    cell_border_polylines,
    ea_forward,
    ea_inverse,
    ecef_to_cell,
)

__all__ = [
    "N_SUBDIV",
    "N_REGIONS",
    "RegionGeometry",
    "cell_border_polylines",
    "ea_forward",
    "ea_inverse",
    "ecef_to_cell",
]
