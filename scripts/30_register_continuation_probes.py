"""Register continuation-basis MR steering vectors as saklas probes.

PROJECT_2 Session-3 §D found the h_first MR−NB direction (the basis of
the registered `mr.nb` / `q_mr` steering vectors, per
`22c_register_centroid_probes.py`) is **near-orthogonal** to the
continuation-basis MR−NB direction — full-stack cosine 0.02–0.18
across gemma / qwen / ministral. The h_first vectors steer the
*kaomoji-emission* MR axis; the DBT-for-LLMs intervention paradigm
targets the *continuation-time* MR axis. They are different
directions, so the h_first vectors cannot do the job.

This script registers the continuation-basis counterparts:

- `mr_cont.nb`  -- continuation MR centroid − continuation NB centroid
                   (the steering-ready bipolar; mirrors `mr.nb`)
- `q_mr_cont`   -- continuation MR centroid, unipolar (mirrors `q_mr`)

Distinct concept names (not a new namespace) so they coexist with the
h_first `mr.nb` / `q_mr` without collision in steering-expression
resolution. `03_emit_attractor_steered.py` consumes them via
`LLMOJI_STEER_VECTOR=mr_cont.nb`.

Centroids: full-data, reconstructed as the count-weighted mean of the
A / B held-out halves from `40b_continuation_centroid_holdout.py`
(`data/local/continuation_centroid_holdout/`). A ∪ B is the full
prompt set, so the weighted mean is the exact full-data centroid.

Layer set: the intersection of the continuation layer set with the
registered h_first `mr.nb` layer set — i.e. *exactly* the layers the
h_first `mr.nb` lives on. This makes the steering redo a clean
controlled comparison: same model, same layers, same alpha grid,
only the vector direction changes.

Usage::

    .venv/bin/python scripts/local/49_register_continuation_probes.py
    .venv/bin/python scripts/local/49_register_continuation_probes.py --models gemma
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Saklas-profile save helpers + pack.json synthesis (extracted from 22c).
from attractor_experiment import saklas_profiles as _r22c  # noqa: E402
from attractor_experiment.config import DATA_DIR, resolve_model  # noqa: E402

DEFAULT_MODELS = ("gemma", "qwen", "ministral")


def _hfirst_mrnb_layers(model_id_safe: str) -> list[int]:
    """Sorted integer layer set of the registered h_first ``mr.nb``."""
    path = (Path.home() / ".saklas" / "vectors" / "llmoji" / "mr.nb"
            / f"{model_id_safe}.safetensors")
    with safe_open(str(path), "pt") as f:
        return sorted(int(k.split("_")[1]) for k in f.keys())


def _full_centroids(
    model: str,
) -> tuple[dict[str, np.ndarray], dict[str, int], list[int], int]:
    """Full-data continuation centroids = count-weighted mean of A/B halves.

    Returns ({cell: (n_layers, hidden_dim)}, {cell: n_rows}, layer_idxs,
    hidden_dim).
    """
    cdir = DATA_DIR / "local" / "continuation_centroid_holdout" / model
    manifest = json.loads((cdir / "manifest.json").read_text())
    layer_idxs = list(manifest["layer_idxs"])
    hd = int(manifest["hidden_dim_per_layer"])
    npz_A = np.load(cdir / "centroids_A.npz")
    npz_B = np.load(cdir / "centroids_B.npz")
    splits = manifest["splits"]
    out: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for cell in npz_A.files:
        n_a = int(splits[cell]["n_rows_A"])
        n_b = int(splits[cell]["n_rows_B"])
        a = npz_A[cell].astype(np.float64)
        b = npz_B[cell].astype(np.float64)
        full = (n_a * a + n_b * b) / (n_a + n_b)
        out[cell] = full.reshape(len(layer_idxs), hd)
        counts[cell] = n_a + n_b
    return out, counts, layer_idxs, hd


def _subset(centroid: np.ndarray, layer_idxs: list[int],
            keep: list[int]) -> np.ndarray:
    pos = {L: i for i, L in enumerate(layer_idxs)}
    return centroid[[pos[L] for L in keep], :]


def _per_layer_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(a.shape[0], np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def process(model: str) -> None:
    print(f"\n=== {model} ===")
    M = resolve_model(model)
    from saklas.io.paths import safe_model_id
    model_id_safe = safe_model_id(M.model_id)

    cents, counts, layer_idxs, _ = _full_centroids(model)
    hf_layers = _hfirst_mrnb_layers(model_id_safe)
    keep = sorted(set(layer_idxs) & set(hf_layers))
    print(f"  continuation {len(layer_idxs)}L, h_first mr.nb {len(hf_layers)}L, "
          f"registering on {len(keep)}L intersection")

    mr = _subset(cents["MR"], layer_idxs, keep)   # (n_keep, hd)
    nb = _subset(cents["NB"], layer_idxs, keep)
    mr_nb = mr - nb

    # Sanity: confirm the registered direction is the §D-orthogonal one.
    hf_path = (Path.home() / ".saklas" / "vectors" / "llmoji" / "mr.nb"
               / f"{model_id_safe}.safetensors")
    with safe_open(str(hf_path), "pt") as f:
        hf_mrnb = np.stack([f.get_tensor(f"layer_{L}").float().numpy()
                            for L in keep])
    full_cos = float(
        mr_nb.ravel() @ hf_mrnb.ravel()
        / (np.linalg.norm(mr_nb) * np.linalg.norm(hf_mrnb))
    )
    pl = _per_layer_cos(mr_nb, hf_mrnb.astype(np.float64))
    print(f"  cos(mr_cont.nb, h_first mr.nb): full-stack {full_cos:+.4f}, "
          f"per-layer mean {np.nanmean(pl):+.4f}  "
          f"(near-0 expected per §D)")

    # q_mr_cont -- unipolar continuation MR centroid.
    prof_q = _r22c._profile_dict_from_layerstack(
        mr.astype(np.float32), keep)
    p1 = _r22c._save_centroid_profile(
        prof_q, concept="q_mr_cont", model_id=M.model_id,
        method="centroid_unipolar",
        components={"quadrant": "MR", "basis": "continuation",
                    "n_rows": counts["MR"]},
    )
    print(f"    wrote q_mr_cont   ‖v‖={np.linalg.norm(mr):8.2f}  → {p1.name}")

    # mr_cont.nb -- steering-ready bipolar.
    prof_b = _r22c._profile_dict_from_layerstack(
        mr_nb.astype(np.float32), keep)
    p2 = _r22c._save_centroid_profile(
        prof_b, concept="mr_cont.nb", model_id=M.model_id,
        method="centroid_difference",
        components={"plus": "MR", "minus": "NB", "basis": "continuation",
                    "n_plus": counts["MR"], "n_minus": counts["NB"]},
    )
    print(f"    wrote mr_cont.nb  ‖v‖={np.linalg.norm(mr_nb):8.2f}  → {p2.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        try:
            process(model)
        except Exception as e:  # noqa: BLE001
            print(f"[{model}] skipped: {e!r}")
    print()
    n = _r22c._write_pack_jsons()
    print(f"refreshed {n} pack.json manifests")


if __name__ == "__main__":
    main()
