"""Base traversal-cost surfaces — the final worldgen stage.

Converts already-computed physical data (biome, ruggedness, ocean mask, rivers,
coast distance) into the *physical base* land and sea movement-cost layers that
the downstream realm layer (realmgen) consumes. Purely physical and reusable;
the RPG-specific danger field is layered on by realmgen, not here.

See docs/settlement_generation_design.md (Stage 0) for the rationale.
"""
