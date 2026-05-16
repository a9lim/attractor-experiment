"""Saklas-profile write helpers.

Extracted from llmoji-study's ``scripts/local/22c_register_centroid_probes.py``
for the continuation-probe registration script. Holds the four helpers
that write centroid profiles into a saklas vector namespace plus the
per-concept ``pack.json`` synthesis pass:

- ``_saklas_namespace_dir`` — resolve/create ``<saklas_home>/vectors/<ns>/``
- ``_save_centroid_profile`` — write one centroid profile + sidecar
- ``_profile_dict_from_layerstack`` — build the ``{layer: tensor}`` dict
- ``_write_pack_jsons`` — synthesize ``pack.json`` manifests per concept

``_NAMESPACE`` is fixed to ``"llmoji"`` here — that is the namespace the
continuation probes register under.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


_PACK_TAGS = ("llmoji-study", "centroid", "v4-9cell")
_PACK_SOURCE = "llmoji-study/scripts/local/22c_register_centroid_probes.py"
_PACK_LICENSE = "CC-BY-4.0"
_PACK_RECOMMENDED_ALPHA = 0.3  # centroid magnitudes are large; start low


# Namespace slug the continuation probes register under.
_NAMESPACE: str = "llmoji"


def _saklas_namespace_dir() -> Path:
    """Return ``<saklas_home>/vectors/<namespace>/`` (created if missing).

    Namespace is ``_NAMESPACE`` (``llmoji``). Sibling trees keep
    different prompt-set centroids discoverable but separately
    addressable in saklas's UI / grammar.
    """
    from saklas.io.paths import vectors_dir
    out = vectors_dir() / _NAMESPACE
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_centroid_profile(
    profile_dict: dict[int, torch.Tensor],
    *,
    concept: str,
    model_id: str,
    method: str,
    components: dict | None = None,
    n_rows_per_layer: int | None = None,
) -> Path:
    """Save a centroid profile via ``saklas.core.vectors.save_profile``.

    ``concept`` is the saklas concept slug (e.g. ``q_HPD``,
    ``HPD.NB``). ``method`` goes into the JSON sidecar so a downstream
    reader can branch on centroid flavor without inferring from the
    name. ``components`` carries provenance for difference probes
    (e.g. ``{"plus": "HPD", "minus": "NB"}``).
    """
    from saklas.core.vectors import save_profile
    from saklas.io.paths import safe_model_id

    namespace_dir = _saklas_namespace_dir()
    concept_dir = namespace_dir / concept
    concept_dir.mkdir(parents=True, exist_ok=True)
    path = concept_dir / f"{safe_model_id(model_id)}.safetensors"
    metadata: dict = {"method": method}
    if components is not None:
        metadata["components"] = components
    if n_rows_per_layer is not None:
        metadata["components"] = (
            {**(metadata.get("components", {})),
             "n_rows": int(n_rows_per_layer)}
        )
    save_profile(profile_dict, str(path), metadata)
    return path


def _profile_dict_from_layerstack(
    centroid: np.ndarray,
    layer_idxs: list[int],
) -> dict[int, torch.Tensor]:
    """Build the saklas ``{layer_idx: tensor}`` profile dict from an
    ``(n_layers, hidden_dim)`` centroid array.

    Tensors are float32 on CPU — ``save_profile`` writes via safetensors
    which doesn't care about device, but fp32 keeps the file
    self-explanatory and matches saklas's bake convention.
    """
    profile: dict[int, torch.Tensor] = {}
    for i, L in enumerate(layer_idxs):
        v = centroid[i].astype(np.float32, copy=False)
        profile[int(L)] = torch.from_numpy(v.copy())
    return profile


# Static per-concept descriptions written into pack.json. Keyed by
# concept slug. Built dynamically for unipolar/vs-NB; the 3 axis probes
# are spelled out individually.
def _concept_description(concept: str) -> str:
    if concept == "q_mr":
        return (
            "Unipolar centroid of MR cell (meta-register basin; the "
            "egregore / saturated-memetic register cell, formerly named "
            "LB before the 2026-05-11 rename): per-layer mean h_first "
            "across kaomoji-emitting prompts in the LB pilot dataset "
            "(data/local/<short>_lb/ — bliss-content prompt set, "
            "lb01…lb20). Promoted to QUADRANT_ORDER_SPLIT on 2026-05-10 "
            "via the attractor-trajectory pilot — basin lock cross-"
            "model (gemma 58% / qwen 100% / ministral 100% basin→basin "
            "at 128-token continuation) plus cross-content invariance "
            "(bliss / doom / conspiracy / sycophancy prefills all land "
            "here with pairwise arm-arm cos ≥ 0.89). Confirmed "
            "pretraining-anchored 2026-05-11 by the base-vs-instruct "
            "basin test (gemma-base shows comparable basin lock to "
            "gemma-instruct). The basin is a geometric reflection of "
            "egregore-shaped human-generated text in the corpus, not "
            "an RLHF artifact. Treat as observation, not as a "
            "deployment-steering recipe (see "
            "docs/2026-05-11-base-vs-instruct-basin.md for ethics "
            "framing)."
        )
    if concept == "mr.nb":
        return (
            "Bipolar centroid difference: MR − NB. Per-layer "
            "displacement from the neutral-baseline cell to the "
            "meta-register basin (formerly lb.nb, renamed 2026-05-11). "
            "NB acts as the neutral pole, MR as the active pole. "
            "Steering this direction reproduces a documented attractor "
            "register; observation-only and welfare-relevant (see "
            "docs/2026-05-11-base-vs-instruct-basin.md and the "
            "egregore-basin-as-failure-mode framing)."
        )
    if concept.startswith("q_"):
        slug = concept[2:].upper()
        return (
            f"Unipolar centroid of {slug} quadrant rows: per-layer mean "
            f"h_first across all kaomoji-emitting v3 prompts labeled {slug}. "
            f"Score against any hidden state for an empirical 'how aligned "
            f"with {slug}?' readout. Not share-baked — empirical magnitude "
            f"is preserved in the per-layer norms."
        )
    if concept.endswith(".nb"):
        plus = concept.split(".")[0].upper()
        return (
            f"Bipolar centroid difference: {plus} − NB. Per-layer "
            f"displacement from the neutral-baseline cell to {plus}. "
            f"Steering-ready (NB acts as the neutral pole, {plus} as the "
            f"active pole). Not share-baked."
        )
    if concept == "hp.ln":
        return (
            "Valence axis: aggregate HP centroid minus LN centroid "
            "(row-weighted across HP-D + HP-S rows). Spans the dominant "
            "valence direction in the v3 dataset."
        )
    if concept == "hp.lp":
        return (
            "Arousal-in-positive-valence axis: aggregate HP centroid "
            "minus LP centroid (both positive valence; isolates arousal)."
        )
    if concept == "hnd.hns":
        return (
            "Dominance-within-HN axis: HN-D centroid minus HN-S centroid. "
            "The v3-validated dominance split — HN-D = anger/contempt "
            "(in-action), HN-S = fear/anxiety (received-threat)."
        )
    return f"llmoji-study centroid probe: {concept}"


def _write_pack_jsons() -> int:
    """Walk every concept folder under ``vectors/llmoji/`` and write a
    ``pack.json`` with files-hash manifest. Required by saklas's
    ``ConceptFolder.load`` (saklas/io/packs.py) — without it the UI
    grammar enumeration skips the folder entirely.

    Idempotent: re-running picks up any new safetensors+sidecar pairs
    and refreshes the hash manifest. Returns the number of pack.json
    files written.
    """
    from saklas.io.packs import synthesize_pack_metadata

    namespace_dir = _saklas_namespace_dir()
    n_written = 0
    for concept_dir in sorted(namespace_dir.iterdir()):
        if not concept_dir.is_dir():
            continue
        # Skip concept folders with no model files yet — synthesize would
        # produce an empty manifest that ConceptFolder.load rejects.
        has_files = any(
            f.is_file() and f.name != "pack.json"
            for f in concept_dir.iterdir()
        )
        if not has_files:
            print(f"  skipping pack.json for empty {concept_dir.name}")
            continue

        concept = concept_dir.name
        pack = synthesize_pack_metadata(
            name=concept,
            description=_concept_description(concept),
            tags=list(_PACK_TAGS),
            version="1.0.0",
            license=_PACK_LICENSE,
            recommended_alpha=_PACK_RECOMMENDED_ALPHA,
            source=_PACK_SOURCE,
            pack_dir=concept_dir,
        )
        pack.write(concept_dir)
        n_written += 1
        print(f"  pack.json: {concept} ({len(pack.files)} files manifested)")
    return n_written
