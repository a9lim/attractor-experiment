"""attractor-experiment paths layered over workspace-shared configuration."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

from transformer_experiments.kaomoji import settings as _shared_settings
from transformer_experiments.models import MODEL_REGISTRY as SHARED_MODEL_REGISTRY


# Compatibility re-exports. The values themselves are owned at workspace root.
PROBES = _shared_settings.PROBES
PROBE_CATEGORIES = _shared_settings.PROBE_CATEGORIES
STEERED_AXIS = _shared_settings.STEERED_AXIS
TEMPERATURE = _shared_settings.TEMPERATURE


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
EMOTIONAL_CONDITION = "kaomoji_prompted"


@dataclass(frozen=True)
class ModelPaths:
    """Shared model metadata plus attractor-experiment-owned paths."""

    model_id: str
    short_name: str
    emotional_data_path: Path
    emotional_summary_path: Path
    experiment: str
    figures_dir: Path
    use_saklas: bool = True
    trust_remote_code: bool = False
    probe_calibrated: bool = True


def _paths(short: str) -> ModelPaths:
    spec = SHARED_MODEL_REGISTRY[short]
    return ModelPaths(
        model_id=spec.model_id,
        short_name=spec.short_name,
        emotional_data_path=DATA_DIR / "local" / short / "emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "local" / short / "emotional_summary.tsv",
        experiment=short,
        figures_dir=FIGURES_DIR / "local" / short,
        use_saklas=spec.use_saklas,
        trust_remote_code=spec.trust_remote_code,
        probe_calibrated=spec.probe_calibrated,
    )


MODEL_REGISTRY: dict[str, ModelPaths] = {
    short: _paths(short) for short in SHARED_MODEL_REGISTRY
}


def resolve_model(short: str) -> ModelPaths:
    """Resolve a model slug, applying this experiment's output suffix."""
    if short not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {short!r}; known: {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[short]
    active = os.environ.get("LLMOJI_MODEL", "gemma")
    out_suffix = os.environ.get("LLMOJI_OUT_SUFFIX")
    if out_suffix and short == active:
        suffixed = f"{model.short_name}_{out_suffix}"
        model = dataclasses.replace(
            model,
            emotional_data_path=DATA_DIR / "local" / suffixed / "emotional_raw.jsonl",
            emotional_summary_path=DATA_DIR / "local" / suffixed / "emotional_summary.tsv",
            experiment=suffixed,
            figures_dir=FIGURES_DIR / "local" / suffixed,
        )
    return model


def current_model() -> ModelPaths:
    """Resolve ``$LLMOJI_MODEL`` (default ``gemma``)."""
    return resolve_model(os.environ.get("LLMOJI_MODEL", "gemma"))
