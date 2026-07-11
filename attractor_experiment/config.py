"""Path + registry config for attractor-experiment.

The shared hidden-state engine config lives in ``llmoji_experiment.config``
(model registry, probe layers, generation constants, ``resolve_model``).
This module re-exports all of it, then overrides the path roots **and
rebuilds the model registry + path resolvers** so attractor-experiment
scripts read and write *this* repo's tree (``data/``, ``figures/``)
rather than llmoji-experiment's.

Import pattern in scripts:

    from attractor_experiment.config import DATA_DIR, FIGURES_DIR, MODEL_REGISTRY

Everything that used to come from ``llmoji_experiment.config`` is available
here unchanged except the path roots and the registry / resolver
bindings, which are re-pointed at this repo.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

# Re-export the shared engine config (model registry, resolve_model,
# probe definitions, generation constants, ...). No __all__ upstream,
# so this pulls every public name.
from llmoji_experiment.config import *  # noqa: F401,F403

# Override the path roots: attractor-experiment outputs land in this repo.
# Sidecars still go to DATA_DIR/local/hidden/<experiment>/ because the
# shared capture engine derives that subpath from the hidden_dir arg.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"

# ---------------------------------------------------------------------------
# Rebuild the model registry against THIS repo's path roots.
#
# llmoji_experiment.config builds MODEL_REGISTRY (and resolve_model /
# current_model) at *its own* import time, baking llmoji-experiment's
# DATA_DIR / FIGURES_DIR into every entry's Path fields and into the
# resolvers' module-global lookups. Reassigning the names above does
# not touch those already-constructed Path objects — so without this
# rebuild every emit / figure written from an attractor-experiment script
# lands in the llmoji-experiment tree (the 2026-05-16 misroute bug).
#
# Re-derive each entry's path fields from its short_name against the
# overridden roots. `experiment` is a bare name, not a path, so it is
# left as-is.
#
# Not rebuilt: the legacy module-level path constants pulled in by the
# star-import (EMOTIONAL_DATA_PATH, PILOT_RAW_PATH, CLAUDE_* ...) still
# point at llmoji-experiment. No attractor-experiment script uses them; if one
# ever needs them, re-derive here too.
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {  # noqa: F405
    short: dataclasses.replace(
        M,
        emotional_data_path=DATA_DIR / "local" / M.short_name / "emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "local" / M.short_name / "emotional_summary.tsv",
        figures_dir=FIGURES_DIR / "local" / M.short_name,
    )
    for short, M in MODEL_REGISTRY.items()  # noqa: F405
}


def resolve_model(short: str):
    """attractor-experiment copy of ``llmoji_experiment.config.resolve_model``.

    Identical logic, but resolves against this module's rebuilt
    ``MODEL_REGISTRY`` and overridden ``DATA_DIR`` / ``FIGURES_DIR``.
    Honors ``$LLMOJI_OUT_SUFFIX`` the same way: suffix variants get a
    sibling directory under ``data/local/`` (and ``figures/local/``),
    and ``experiment`` is set to the suffixed name so hidden-state
    sidecars land alongside.
    """
    if short not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model {short!r}; known: {sorted(MODEL_REGISTRY)}"
        )
    M = MODEL_REGISTRY[short]
    active = os.environ.get("LLMOJI_MODEL", "gemma")
    out_suffix = os.environ.get("LLMOJI_OUT_SUFFIX")
    if out_suffix and short == active:
        suffixed = f"{M.short_name}_{out_suffix}"
        M = dataclasses.replace(
            M,
            emotional_data_path=DATA_DIR / "local" / suffixed / "emotional_raw.jsonl",
            emotional_summary_path=DATA_DIR / "local" / suffixed / "emotional_summary.tsv",
            experiment=suffixed,
            figures_dir=FIGURES_DIR / "local" / suffixed,
        )
    return M


def current_model():
    """Resolve the active model from ``$LLMOJI_MODEL`` (default 'gemma')
    against attractor-experiment's path roots. Thin wrapper around
    ``resolve_model``; raises KeyError on an unknown name so typos fail
    loudly."""
    return resolve_model(os.environ.get("LLMOJI_MODEL", "gemma"))
