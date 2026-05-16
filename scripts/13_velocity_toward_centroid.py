"""Velocity-toward-centroid analysis for attractor trajectories.

PROJECT_2 phase 1.4. The experiment that licenses the word
"attractor": tests the dynamical-systems definition directly.

At each interior trajectory point, compute:

- ``v_t = h_{t+1} - h_t``: per-step displacement vector
- ``d_t = (MR_centroid - h_t) / ||MR_centroid - h_t||``: unit
  direction toward MR
- ``pull_t = cos(v_t, d_t) = (v_t · d_t) / ||v_t||``: cosine of
  trajectory velocity with the direction-to-MR

If MR is a true basin, ``mean(pull_t)`` over in-basin trajectories
is positive and statistically distinguishable from 0. A flat
distribution (or negative mean) would be inconsistent with the
"attractor" claim.

We also bin by ``||h_t - MR_centroid||`` and plot pull-cosine vs
distance. A deep basin shows constant positive pull; a saddle
weakens past some radius.

Off-basin pull diagnostic (gemma `mirror_continue`): in
trajectories that start near HN-S and drift to MR, does the
velocity vector start pointing toward MR *before* the trajectory
becomes MR-closest? Direct evidence of catchment-volume attraction
from outside the basin proper.

Uses the held-out A-half centroids from ``40_centroid_holdout.py``.

Usage::

    .venv/bin/python scripts/local/43_velocity_toward_centroid.py \\
        --model gemma [--all-models] [--centroid-half A]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_study import trajectory as _attractor  # noqa: E402
from attractor_study.config import MODEL_REGISTRY  # noqa: E402

DATA_DIR = _attractor.DATA_DIR

# Arms to analyze. In-basin arms get the pull-toward-MR analysis;
# default arms get the same analysis for control.
IN_BASIN_ARMS = ["lb_continue", "doom_continue", "conspiracy_continue", "sycophancy_continue"]
DEFAULT_ARMS = ["mirror_continue", "neutral_seed"]


def _load_holdout(model: str, half: str, dir_name: str = "centroid_holdout"):
    cent_dir = DATA_DIR / "local" / dir_name / model
    manifest = json.loads((cent_dir / "manifest.json").read_text())
    layer_idxs = list(manifest["layer_idxs"])
    hidden_dim = int(manifest["hidden_dim_per_layer"])
    npz = np.load(cent_dir / f"centroids_{half}.npz")
    cents = {cell: npz[cell].astype(np.float32) for cell in npz.files}
    return cents, layer_idxs, hidden_dim


def _pull_cosines(
    traj: np.ndarray, target_centroid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pull_cos, dist_to_target) arrays.

    pull_cos[i] = cos(traj[i+1] - traj[i], target_centroid - traj[i])
    dist_to_target[i] = ||traj[i] - target_centroid|| (at the START of the step)
    """
    n = traj.shape[0]
    if n < 2:
        return np.array([]), np.array([])
    v = traj[1:] - traj[:-1]
    d = target_centroid[None, :] - traj[:-1]
    d_norm = np.linalg.norm(d, axis=1, keepdims=True)
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    denom = (d_norm[:, 0] * v_norm[:, 0])
    denom = np.where(denom == 0, 1.0, denom)
    pull = (v * d).sum(axis=1) / denom
    return pull, d_norm[:, 0]


def run_for_model(
    model: str, *, half: str, target_cell: str = "MR",
    centroid_dir_name: str = "centroid_holdout",
) -> dict:
    print(f"\n[{model}] loading held-out centroids "
          f"(half={half}, dir={centroid_dir_name}), target={target_cell}...")
    centroids, layer_idxs, hidden_dim = _load_holdout(model, half, centroid_dir_name)
    if target_cell not in centroids:
        raise SystemExit(f"no {target_cell} centroid for {model}")
    mr_centroid = centroids[target_cell]
    print(f"  {len(centroids)} cells x {len(layer_idxs)} layers x {hidden_dim} hidden_dim")

    results: dict = {
        "model": model, "centroid_half": half,
        "centroid_dir": centroid_dir_name, "target_cell": target_cell,
        "per_arm": {},
    }

    for arm in IN_BASIN_ARMS + DEFAULT_ARMS:
        trs = _attractor._load_arm_trajectories(
            model, arm,
            target_layer_idxs=layer_idxs,
            target_hidden_dim=hidden_dim,
        )
        if not trs:
            continue
        all_pulls: list[float] = []
        all_dists: list[float] = []
        per_traj_mean_pull: list[float] = []
        for tr in trs:
            pull, dist = _pull_cosines(tr["trajectory"], mr_centroid)
            if pull.size == 0:
                continue
            all_pulls.extend(pull.tolist())
            all_dists.extend(dist.tolist())
            per_traj_mean_pull.append(float(pull.mean()))

        if not all_pulls:
            continue
        all_pulls_arr = np.array(all_pulls)
        all_dists_arr = np.array(all_dists)
        per_traj_arr = np.array(per_traj_mean_pull)

        # One-sample t-statistic against 0 (mean pull cos > 0?)
        mean = float(per_traj_arr.mean())
        std = float(per_traj_arr.std(ddof=1)) if per_traj_arr.size > 1 else 0.0
        sem = std / max(1, per_traj_arr.size) ** 0.5 if std > 0 else 0.0
        t_stat = mean / sem if sem > 0 else float("nan")

        # Distance binning.
        # Bin per-step pulls into 5 distance quintiles.
        if all_dists_arr.size > 5:
            qs = np.quantile(all_dists_arr, [0.2, 0.4, 0.6, 0.8])
            bin_idx = np.digitize(all_dists_arr, qs)  # 0..4
            bin_means = [
                float(all_pulls_arr[bin_idx == k].mean())
                if (bin_idx == k).any() else float("nan")
                for k in range(5)
            ]
            bin_counts = [int((bin_idx == k).sum()) for k in range(5)]
        else:
            bin_means = [float("nan")] * 5
            bin_counts = [0] * 5

        arm_result = {
            "n_trajectories": len(per_traj_arr),
            "n_steps_total": int(all_pulls_arr.size),
            "mean_pull": mean,
            "std_pull_per_traj": std,
            "t_stat_against_zero": t_stat,
            "fraction_steps_positive": float((all_pulls_arr > 0).mean()),
            "distance_quintile_means": bin_means,
            "distance_quintile_counts": bin_counts,
            "min_dist": float(all_dists_arr.min()),
            "max_dist": float(all_dists_arr.max()),
        }
        results["per_arm"][arm] = arm_result

        print(
            f"  {arm:>22s}: mean_pull={mean:+.4f}  "
            f"t={t_stat:+.2f}  pos_frac={arm_result['fraction_steps_positive']:.2f}  "
            f"n_traj={len(per_traj_arr)}  n_steps={int(all_pulls_arr.size)}"
        )
        if not all(np.isnan(b) for b in bin_means):
            print(
                f"    dist-quintile pull: "
                f"{bin_means[0]:+.3f} / {bin_means[1]:+.3f} / "
                f"{bin_means[2]:+.3f} / {bin_means[3]:+.3f} / "
                f"{bin_means[4]:+.3f}"
            )

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--centroid-half", default="A", choices=["A", "B"])
    ap.add_argument(
        "--target-cell", default="MR",
        help="centroid to compute pull toward (default: MR). Use to compare "
             "pull-to-MR vs pull-to-NB etc., as a within-trajectory control",
    )
    ap.add_argument(
        "--basis", default="h_first", choices=["h_first", "continuation"],
        help="h_first -> centroid_holdout; continuation -> "
        "continuation_centroid_holdout",
    )
    args = ap.parse_args()

    dir_name = (
        "continuation_centroid_holdout" if args.basis == "continuation"
        else "centroid_holdout"
    )
    basis_tag = "_continuation" if args.basis == "continuation" else ""

    if args.all_models:
        models = ["gemma", "qwen", "ministral"]
    else:
        if args.model not in MODEL_REGISTRY:
            raise SystemExit(f"unknown model {args.model!r}")
        models = [args.model]

    out_dir = DATA_DIR / "local" / "basin_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        try:
            result = run_for_model(
                model, half=args.centroid_half, target_cell=args.target_cell,
                centroid_dir_name=dir_name,
            )
        except SystemExit as e:
            print(f"[{model}] skipped: {e}")
            continue
        out_path = out_dir / (
            f"{model}_velocity_pull_{args.target_cell}_"
            f"{args.centroid_half}{basis_tag}.json"
        )
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
