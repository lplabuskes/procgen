"""Polity formation — final stage of the realm (human-geography) layer.

A polity is a tree, not a flat partition: sovereigns are roots, vassals are
children. Capitals project an authority field over the land + sea cost surfaces;
faces are claimed by the strongest projector; each settlement is classified
direct / vassal / independent by projected authority vs local power. Modules:
``seeding`` (Stage 0), ``authority`` (Stages 1-3 pure functions), ``forest`` (the
tree algorithm), ``tension`` (border tension), ``builder`` (the pipeline stage).
"""
