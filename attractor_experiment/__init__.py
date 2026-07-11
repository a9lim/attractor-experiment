"""attractor-experiment — the meta-register (MR) basin research line.

Spun off from ``llmoji-experiment`` on 2026-05-15. The kaomoji-introspection
work stays in ``llmoji-experiment``; this repo owns the attractor-trajectory
work: the MR basin, its cross-content / cross-post-training invariance,
continuation-basis geometry, and basin steering.

The shared hidden-state engine (model registry, capture, sidecar IO,
the quadrant taxonomy) is imported from ``llmoji_experiment.*`` — install
that package editable alongside this one. This package itself owns only
the basin-specific prompt families (bliss / doom / conspiracy /
sycophancy / archaic / pre-1930 / plain / saturated-mundane).
"""

__version__ = "0.1.0"
