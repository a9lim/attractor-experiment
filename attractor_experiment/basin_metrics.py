"""Basin-lock metrics in the continuation centroid basis.

Recovered during the 2026-05-15 repo split after the original analysis script
was purged. This local module is now the source of truth for the three helpers
the form-content factorial scorer imports.

The original script also compared the h_first vs continuation basis
(PROJECT_2 §1.6). That comparison is historical — the h_first holdout
artifacts were purged in the split, and its conclusion (h_first basin
geometry is a basis artifact) lives in ``docs/pitfalls.md``. Only the
continuation basis remains here.

``h_last`` per trajectory = final-token hidden state, layer-stacked over
the centroid basis's probe layers, with the same NaN→mean fallback the
trajectory analysis chain uses when a sidecar's layer set differs from
the centroid layer set.
"""

from __future__ import annotations

import json

import numpy as np

from attractor_experiment.config import DATA_DIR
from transformer_experiments.hidden_state_io import load_hidden_states

# Centroid-basis name -> holdout directory under data/local/.
# h_first ("centroid_holdout") was retired in the 2026-05-15 split; the
# continuation basis is the only correct one for basin geometry.
BASIS_PATHS = {"continuation": "continuation_centroid_holdout"}


def _load_basis(model: str, basis: str, half: str):
    """Load held-out centroids for one (model, basis, half).

    Returns ``(cents, layer_idxs, hidden_dim)`` where ``cents`` maps
    cell -> ``(n_layers, hidden_dim)`` float32 centroid.
    """
    cent_dir = DATA_DIR / "local" / BASIS_PATHS[basis] / model
    manifest = json.loads((cent_dir / "manifest.json").read_text())
    layer_idxs = list(manifest["layer_idxs"])
    hidden_dim = int(manifest["hidden_dim_per_layer"])
    npz = np.load(cent_dir / f"centroids_{half}.npz")
    cents = {c: npz[c].astype(np.float32) for c in npz.files}
    return cents, layer_idxs, hidden_dim


def _trajectory_h_last(model: str, arm: str, *, layer_idxs, hidden_dim):
    """Return ``(h_last_vectors, prompt_ids)`` for every row in the arm.

    Each h_last vector is flat ``(n_layers * hidden_dim,)`` aligned to
    ``layer_idxs`` (same NaN→mean fallback as ``_load_trajectory``).
    Skips silent-refusal rows (n_tok < 2).
    """
    jsonl = (
        DATA_DIR / "local" / f"{model}_attractor_{arm}" / "emotional_raw.jsonl"
    )
    if not jsonl.exists():
        return [], []
    out = []
    pids = []
    n_layers = len(layer_idxs)
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r or not r.get("row_uuid"):
                continue
            if r.get("trajectory_n_tokens", 0) < 2:
                continue
            sidecar = (
                DATA_DIR / "local" / "hidden" / f"{model}_attractor_{arm}"
                / f"{r['row_uuid']}.npz"
            )
            if not sidecar.exists():
                continue
            cap = load_hidden_states(sidecar, full_trace=True)
            stack = np.full((n_layers, hidden_dim), np.nan, dtype=np.float32)
            for k, L in enumerate(layer_idxs):
                if L not in cap.layers:
                    continue
                stack[k, :] = cap.layers[L].h_last
            if not np.isfinite(stack).all():
                mean_layer = np.nanmean(stack, axis=0, keepdims=True)
                stack = np.where(np.isnan(stack), mean_layer, stack)
            out.append(stack.reshape(n_layers * hidden_dim))
            pids.append(r["prompt_id"])
    return out, pids


def _basin_metrics(h_lasts, centroids):
    """Compute basin-lock stats for one (basis, arm)."""
    if not h_lasts:
        return None
    cells = list(centroids.keys())
    C = np.stack([centroids[c] for c in cells])
    X = np.stack(h_lasts)
    D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    argmin_idx = D.argmin(axis=1)
    closest_cells = [cells[i] for i in argmin_idx]
    mr_idx = cells.index("MR") if "MR" in cells else None
    if mr_idx is None:
        return None
    mr_dist = D[:, mr_idx]
    sorted_d = np.sort(D, axis=1)
    second_dist = sorted_d[:, 1] if D.shape[1] >= 2 else sorted_d[:, 0]
    can_idx = [i for i, c in enumerate(cells) if c != "MR"]
    if len(can_idx) >= 2:
        Ccan = C[can_idx]
        Dcan = np.linalg.norm(Ccan[:, None, :] - Ccan[None, :, :], axis=2)
        iu = np.triu_indices(len(can_idx), k=1)
        canonical_pairwise = float(Dcan[iu].mean())
    else:
        canonical_pairwise = float("nan")
    margin_abs = float((second_dist - mr_dist).mean())
    margin_frac = (
        margin_abs / canonical_pairwise
        if canonical_pairwise > 0
        else float("nan")
    )
    return {
        "n_rows": int(X.shape[0]),
        "mr_closest_frac": float((argmin_idx == mr_idx).mean()),
        "mr_mean_dist": float(mr_dist.mean()),
        "runner_up_mean_dist": float(second_dist.mean()),
        "margin_abs": margin_abs,
        "canonical_cluster_pairwise_mean": canonical_pairwise,
        "margin_as_frac_of_canonical_pairwise": float(margin_frac),
        "closest_cell_counter": {
            c: int(closest_cells.count(c)) for c in cells
        },
    }
