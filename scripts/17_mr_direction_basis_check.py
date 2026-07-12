"""h_first vs continuation MR-direction alignment check.

PROJECT_2 §1.6 open item 10 / AGENTS.md open work. The registered
welfare-relevant steering vectors (``mr.nb`` = centroid(MR) -
centroid(NB), per ``22c_register_centroid_probes.py``) were calibrated
on ``h_first`` kaomoji-emission centroids. The §1.6 continuation-basis
correction showed continuation trajectories live in a different region
of residual space than the ``h_first`` centroid cluster.

This script answers: **how aligned is the MR direction in the h_first
basis vs the continuation basis?** If cos is high, the registered
``mr.nb`` vectors remain load-bearing for the DBT-for-LLMs steering
paradigm. If low, they need re-registration in continuation basis.

Both centroid sets are the held-out A/B halves (apples-to-apples; the
registered full-data vector has no continuation counterpart). Layer
sets differ between bases (continuation adds a few embedding / final
layers) -- the comparison is restricted to the intersection so the
only thing varying is the centroid-construction *regime*.

Three directions are compared per model:

- ``MR - NB``      -- the actual ``mr.nb`` construction (load-bearing)
- ``MR - mean``    -- MR vs the mean of all 10 cell centroids
- ``MR`` (raw)     -- raw centroid, dominated by the shared mean

Usage::

    python scripts/local/47_mr_direction_basis_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
MODELS = ["gemma", "qwen", "ministral"]


def _load_basis(centroid_dir: Path, half: str) -> tuple[dict[str, np.ndarray], list[int]]:
    """Return ({cell: (n_layers, hidden_dim)}, layer_idxs)."""
    manifest = json.loads((centroid_dir / "manifest.json").read_text())
    layer_idxs = list(manifest["layer_idxs"])
    hd = int(manifest["hidden_dim_per_layer"])
    npz = np.load(centroid_dir / f"centroids_{half}.npz")
    cents = {
        cell: npz[cell].astype(np.float64).reshape(len(layer_idxs), hd)
        for cell in npz.files
    }
    return cents, layer_idxs


def _subset(cents: dict[str, np.ndarray], layer_idxs: list[int],
            keep: list[int]) -> dict[str, np.ndarray]:
    """Restrict each centroid to the layers in ``keep`` (a subset of layer_idxs)."""
    pos = {L: i for i, L in enumerate(layer_idxs)}
    rows = [pos[L] for L in keep]
    return {cell: vec[rows, :] for cell, vec in cents.items()}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(a @ b / (na * nb))


def _per_layer_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine per layer row of two (n_layers, hidden_dim) arrays."""
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(a.shape[0], np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def _directions(cents: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mr = cents["MR"]
    nb = cents["NB"]
    mean = np.mean(np.stack(list(cents.values())), axis=0)
    return {
        "MR-NB": mr - nb,
        "MR-mean": mr - mean,
        "MR-raw": mr,
    }


def run_model(model: str) -> dict:
    hf_dir = DATA_DIR / "local" / "centroid_holdout" / model
    co_dir = DATA_DIR / "local" / "continuation_centroid_holdout" / model
    if not hf_dir.exists() or not co_dir.exists():
        raise SystemExit(f"missing centroid dir for {model}")

    out: dict = {"model": model}
    halves: dict = {}
    for half in ("A", "B"):
        hf_cents, hf_layers = _load_basis(hf_dir, half)
        co_cents, co_layers = _load_basis(co_dir, half)
        common = sorted(set(hf_layers) & set(co_layers))
        hf_c = _subset(hf_cents, hf_layers, common)
        co_c = _subset(co_cents, co_layers, common)

        hf_dirs = _directions(hf_c)
        co_dirs = _directions(co_c)

        per_dir: dict = {}
        for name in ("MR-NB", "MR-mean", "MR-raw"):
            full = _cos(hf_dirs[name], co_dirs[name])
            pl = _per_layer_cos(hf_dirs[name], co_dirs[name])
            per_dir[name] = {
                "full_stack_cos": full,
                "per_layer_cos_mean": float(np.nanmean(pl)),
                "per_layer_cos_min": float(np.nanmin(pl)),
                "per_layer_cos_max": float(np.nanmax(pl)),
            }
        halves[half] = {
            "n_common_layers": len(common),
            "n_hf_layers": len(hf_layers),
            "n_co_layers": len(co_layers),
            "directions": per_dir,
        }
    out["halves"] = halves

    # Within-basis A/B stability of the MR-NB direction, as a yardstick
    # for interpreting the cross-basis cosine.
    for basis, cdir in (("h_first", hf_dir), ("continuation", co_dir)):
        cA, _ = _load_basis(cdir, "A")
        cB, _ = _load_basis(cdir, "B")
        dA = _directions(cA)["MR-NB"]
        dB = _directions(cB)["MR-NB"]
        out[f"{basis}_AB_MR-NB_cos"] = _cos(dA, dB)
    return out


def main() -> None:
    results = []
    for model in MODELS:
        try:
            r = run_model(model)
        except SystemExit as e:
            print(f"[{model}] skipped: {e}")
            continue
        results.append(r)

    print("\n=== MR-direction alignment: h_first basis vs continuation basis ===\n")
    print(f"{'model':<10} {'direction':<9} {'half':<5} "
          f"{'full-stack cos':>15} {'per-layer mean':>15} "
          f"{'per-layer min':>14}")
    for r in results:
        for name in ("MR-NB", "MR-mean", "MR-raw"):
            for half in ("A", "B"):
                d = r["halves"][half]["directions"][name]
                print(f"{r['model']:<10} {name:<9} {half:<5} "
                      f"{d['full_stack_cos']:>15.4f} "
                      f"{d['per_layer_cos_mean']:>15.4f} "
                      f"{d['per_layer_cos_min']:>14.4f}")
        print(f"{'':<10} within-basis A/B MR-NB cos: "
              f"h_first {r['h_first_AB_MR-NB_cos']:.4f}  "
              f"continuation {r['continuation_AB_MR-NB_cos']:.4f}")
        print()

    out_path = DATA_DIR / "local" / "basin_stats" / "mr_direction_basis_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
