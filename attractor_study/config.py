"""Path + registry config for attractor-study.

The shared hidden-state engine config lives in ``llmoji_study.config``
(model registry, probe layers, generation constants, ``resolve_model``).
This module re-exports all of it, then overrides the three path roots
so that attractor-study scripts read and write *this* repo's tree
(``data/``, ``figures/``) rather than llmoji-study's.

Import pattern in scripts:

    from attractor_study.config import DATA_DIR, FIGURES_DIR, MODEL_REGISTRY

Everything that used to come from ``llmoji_study.config`` is available
here unchanged except ``REPO_ROOT`` / ``DATA_DIR`` / ``FIGURES_DIR``.
"""

from __future__ import annotations

from pathlib import Path

# Re-export the shared engine config (model registry, resolve_model,
# probe definitions, generation constants, ...). No __all__ upstream,
# so this pulls every public name.
from llmoji_study.config import *  # noqa: F401,F403

# Override the path roots: attractor-study outputs land in this repo.
# Sidecars still go to DATA_DIR/local/hidden/<experiment>/ because the
# shared capture engine derives that subpath from the hidden_dir arg.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
