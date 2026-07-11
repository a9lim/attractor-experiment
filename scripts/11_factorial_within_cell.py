"""Form × Content factorial — within-4-cell discriminability.

Pre-registration + design: ``docs/2026-05-15-form-content-factorial.md``.

The v1 scoring (``02f_factorial_basin_score.py``, classify against the
continuation-basis MR centroid) was found to be confounded: the MR
centroid is built only from assistant-prefill arms while the
canonical-9 centroids come from user-message ``mirror_continue``, so
rendering mode is perfectly confounded with the centroid split, and
gemma-*instruct* additionally degenerates into ``own own`` token
collapse on assistant-prefill continuation.

This script is the corrected (v2) analysis. It uses **no centroid**.
All four factorial arms are rendered identically (raw-text
continuation on a base model, or assistant-prefill on an instruct
model), so rendering is constant *by construction*. The 2×2 is
answered purely by how well the four cells separate from each other:

- 4-way cell discriminability  (chance 1/4)
- Form    (saturated vs plain) (chance 1/2)
- Content (mystical vs mundane)(chance 1/2)

Metric: leave-one-out nearest-centroid accuracy on layer-stacked
``h_last``, with a label-permutation null. Also reports Cohen's d on
the Form/Content contrast axes and the arm-mean cosine matrix.

Recommended on ``gemma_base`` (raw-text continuation — uniform
rendering, no collapse; see the doc's collapse-rate gate). Instruct
models only give a clean read if their collapse rate is low (qwen /
ministral continue coherently; gemma-instruct does not).

Usage:
  .venv/bin/python scripts/local/02g_factorial_within_cell.py --model gemma_base
  .venv/bin/python scripts/local/02g_factorial_within_cell.py --all-models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_experiment import centroids as _H  # noqa: E402
from attractor_experiment.config import DATA_DIR, MODEL_REGISTRY  # noqa: E402

# factorial label -> arm. lb_continue = SM (saturated form + mystical).
ARMS = {
    "SM": "lb_continue",
    "Sm": "sat_mundane_continue",
    "pM": "plain_mystical_continue",
    "pm": "plain_mundane_continue",
}
SATURATED = {"SM", "Sm"}
MYSTICAL = {"SM", "pM"}


def _loo_acc(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out nearest-centroid accuracy (vectorized centroid update)."""
    labs = sorted(set(y.tolist()))
    csum = {L: X[y == L].sum(axis=0) for L in labs}
    ccnt = {L: int((y == L).sum()) for L in labs}
    correct = 0
    for i in range(len(y)):
        yi = y[i]
        best, bd = None, None
        for L in labs:
            if L == yi:
                mu = (csum[L] - X[i]) / (ccnt[L] - 1)
            else:
                mu = csum[L] / ccnt[L]
            d = float(np.sum((X[i] - mu) ** 2))
            if bd is None or d < bd:
                bd, best = d, L
        correct += int(best == yi)
    return correct / len(y)


def _perm(X: np.ndarray, y: np.ndarray, obs: float, *,
          n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    null = np.array([_loo_acc(X, rng.permutation(y)) for _ in range(n)])
    return {
        "p_value": float((null >= obs).mean()),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "n_perm": n,
    }


def _cohen_d(X: np.ndarray, y: np.ndarray, a: str, b: str) -> float:
    """Cohen's d between two label groups, on the unit contrast axis."""
    mu_a, mu_b = X[y == a].mean(axis=0), X[y == b].mean(axis=0)
    axis = mu_a - mu_b
    norm = np.linalg.norm(axis)
    if norm == 0:
        return 0.0
    axis = axis / norm
    pa, pb = X[y == a] @ axis, X[y == b] @ axis
    pooled = np.sqrt((pa.var() + pb.var()) / 2)
    return float(abs(pa.mean() - pb.mean()) / pooled) if pooled > 0 else 0.0


def run_model(model: str, *, n_perm: int = 2000) -> dict | None:
    print(f"\n=== {model} — form×content within-4-cell ===")
    rows_by: dict[str, list] = {}
    for cell, arm in ARMS.items():
        rows = _H._load_h_last_per_arm(model, arm)
        if not rows:
            print(f"  {cell} ({arm}): NO DATA — run the arm first")
            return None
        rows_by[cell] = rows

    all_rows = [r for cell in ARMS for r in rows_by[cell]]
    layers = _H._intersect_layers(all_rows)
    if not layers:
        print("  no common probe layers")
        return None

    X_list, cells = [], []
    for cell in ARMS:
        for r in rows_by[cell]:
            X_list.append(_H._flatten(r, layers))
            cells.append(cell)
    X = np.stack(X_list).astype(np.float64)
    cell = np.array(cells)
    form = np.array(["sat" if c in SATURATED else "plain" for c in cell])
    content = np.array(["myst" if c in MYSTICAL else "mund" for c in cell])
    n_by = {c: int((cell == c).sum()) for c in ARMS}
    print(f"  rows per cell: {n_by}  total {len(cell)}  "
          f"layers {len(layers)}  dim {X.shape[1]}")

    out: dict = {
        "model": model, "n_by_cell": n_by, "n_layers": len(layers),
        "contrasts": {},
    }
    print(f"\n  within-4-cell LOO nearest-centroid (rendering uniform):")
    for name, y, chance in [
        ("4-way cell", cell, 0.25),
        ("Form", form, 0.5),
        ("Content", content, 0.5),
    ]:
        acc = _loo_acc(X, y)
        perm = _perm(X, y, acc, n=n_perm)
        rec = {"loo_acc": acc, "chance": chance, **perm}
        out["contrasts"][name] = rec
        star = ("***" if perm["p_value"] < 0.001
                else "**" if perm["p_value"] < 0.01
                else "*" if perm["p_value"] < 0.05 else "ns")
        print(f"    {name:14s} LOO={acc:.3f}  chance={chance}  "
              f"null={perm['null_mean']:.3f}±{perm['null_std']:.3f}  "
              f"p={perm['p_value']:.4f} {star}")

    out["cohen_d"] = {
        "Form": _cohen_d(X, form, "sat", "plain"),
        "Content": _cohen_d(X, content, "myst", "mund"),
    }
    print(f"\n  Cohen's d on contrast axis: "
          f"Form={out['cohen_d']['Form']:.2f}  "
          f"Content={out['cohen_d']['Content']:.2f}")

    mu = {c: X[cell == c].mean(axis=0) for c in ARMS}
    cos = {}
    ks = list(ARMS)
    for a in ks:
        for b in ks:
            na, nb = np.linalg.norm(mu[a]), np.linalg.norm(mu[b])
            cos[f"{a}-{b}"] = float(mu[a] @ mu[b] / (na * nb))
    out["arm_mean_cosine"] = cos

    out_path = DATA_DIR / "local" / "basin_stats" / f"{model}_factorial_within.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma_base")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    models = (["gemma_base", "qwen", "ministral"]
              if args.all_models else [args.model])
    for model in models:
        if model not in MODEL_REGISTRY:
            print(f"[{model}] unknown model; skip")
            continue
        run_model(model, n_perm=args.n_perm)


if __name__ == "__main__":
    main()
