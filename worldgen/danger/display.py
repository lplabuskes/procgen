"""Color scheme for the danger stage."""
from __future__ import annotations

from worldgen.geology.display import ocean_gradient_scheme

# Neutral land → amber → red → deep red
_DANGER_STOPS = [
    (0.00, (0.72, 0.52, 0.25, 1.0)),
    (0.25, (0.88, 0.55, 0.10, 1.0)),
    (0.50, (0.85, 0.18, 0.05, 1.0)),
    (0.75, (0.65, 0.05, 0.05, 1.0)),
    (1.00, (0.30, 0.00, 0.05, 1.0)),
]

danger_color = ocean_gradient_scheme("danger", _DANGER_STOPS, default=0.0)


def danger_legend_entries() -> list[tuple[str, tuple]]:
    labels = ["safe", "0.25", "0.50", "0.75", "lethal"]
    return [(lbl, c) for lbl, (_, c) in zip(labels, _DANGER_STOPS)]
