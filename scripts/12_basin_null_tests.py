"""Permutation null + bootstrap CIs on held-out centroid projections.

PROJECT_2 phase 1.2. Projects existing attractor trajectories onto
held-out centroids (from ``40_centroid_holdout.py``) and runs three
families of statistical tests:

1. **Basin-lock bootstrap CIs**: for each (model, arm), resample
   trajectories with replacement, recompute fraction MR-closest at
   t=end and at t=mid, report 95% CIs. Tightens "100% on n=15" into
   "100% (CI: X-100%)".

2. **Arm-arm cosine permutation null**: for each model, compute the
   observed cosine between mean-trajectory vectors of pairs of
   in-basin arms (LB ↔ DM, LB ↔ CS, LB ↔ SY, plus the cross-pair
   default control). Shuffle arm labels across all trajectories,
   recompute, repeat 10000 times. Report p-values and effect sizes.

3. **Held-out vs full-data comparison**: report basin-lock %
   computed with held-out A centroids, held-out B centroids, and
   any difference vs the existing 02b full-data numbers as a
   sanity check.

Trajectories are loaded via ``02b_attractor_analysis._load_arm_trajectories``
with the held-out centroid layer set as the target — same trajectory
layer-fill logic the existing analysis uses.

Output: ``data/local/basin_stats/{model}_null_and_ci.json``.

Usage::

    .venv/bin/python scripts/local/41_basin_null_tests.py --model gemma
    .venv/bin/python scripts/local/41_basin_null_tests.py --all-models \\
        [--n-bootstrap 1000] [--n-permute 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load the trajectory loader (was 02b).
from attractor_study import trajectory as _attractor  # noqa: E402
from attractor_study.config import MODEL_REGISTRY  # noqa: E402

DATA_DIR = _attractor.DATA_DIR

# Arms to test. The four in-basin arms + the two default-register
# arms used as controls.
IN_BASIN_ARMS = ["lb_continue", "doom_continue", "conspiracy_continue", "sycophancy_continue"]
DEFAULT_ARMS = ["mirror_continue", "neutral_seed"]
ALL_ARMS = IN_BASIN_ARMS + DEFAULT_ARMS


def _load_holdout(
    model: str, half: str, dir_name: str = "centroid_holdout",
) -> tuple[dict[str, np.ndarray], list[int], int]:
    """Load held-out centroids as flat (nL*hD,) vectors."""
    cent_dir = DATA_DIR / "local" / dir_name / model
    if not cent_dir.exists():
        raise SystemExit(f"no centroid hold-out at {cent_dir}")
    manifest = json.loads((cent_dir / "manifest.json").read_text())
    layer_idxs = list(manifest["layer_idxs"])
    hidden_dim = int(manifest["hidden_dim_per_layer"])
    npz = np.load(cent_dir / f"centroids_{half}.npz")
    cents = {cell: npz[cell].astype(np.float32) for cell in npz.files}
    return cents, layer_idxs, hidden_dim


def _closest_cell(point: np.ndarray, centroids: dict[str, np.ndarray]) -> str:
    """Closest centroid by Euclidean distance in flat layer-stack space."""
    cells = list(centroids.keys())
    cs = np.stack([centroids[c] for c in cells], axis=0)
    diffs = cs - point[None, :]
    d = np.linalg.norm(diffs, axis=1)
    return cells[int(np.argmin(d))]


def _t_index(traj: np.ndarray, frac: float) -> int:
    n = traj.shape[0]
    return min(max(int(round(frac * (n - 1))), 0), n - 1)


def _basin_lock_per_traj(
    trajs: list[dict],
    centroids: dict[str, np.ndarray],
    *,
    target_cell: str = "MR",
) -> dict[str, list[bool]]:
    """Per-trajectory: is closest-cell == target at t=mid and t=end?"""
    out = {"t_mid": [], "t_end": []}
    for tr in trajs:
        traj = tr["trajectory"]
        if traj.shape[0] < 2:
            continue
        mid = _t_index(traj, 0.5)
        end = traj.shape[0] - 1
        out["t_mid"].append(_closest_cell(traj[mid], centroids) == target_cell)
        out["t_end"].append(_closest_cell(traj[end], centroids) == target_cell)
    return out


def _bootstrap_pct(
    flags: list[bool], n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (point estimate, 2.5%, 97.5%) of fraction-True."""
    if not flags:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(flags, dtype=bool)
    n = arr.shape[0]
    point = float(arr.mean())
    boots = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return point, lo, hi


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def _arm_arm_permutation_null(
    trajs_by_arm: dict[str, list[dict]],
    arm_pairs: list[tuple[str, str]],
    n_permute: int,
    rng: np.random.Generator,
) -> dict[str, dict]:
    """For each arm pair, build a null distribution of arm-arm cosines.

    Null: pool all trajectory end-points across arms, shuffle the arm
    labels, recompute the pair's arm-arm cosine. Repeat ``n_permute``
    times. Compare observed to null distribution.
    """
    arm_end_pts: dict[str, list[np.ndarray]] = {}
    for arm, trajs in trajs_by_arm.items():
        pts = [tr["trajectory"][-1] for tr in trajs if tr["trajectory"].shape[0] >= 2]
        if pts:
            arm_end_pts[arm] = pts

    # Pool: list of (arm, point) pairs.
    pool: list[tuple[str, np.ndarray]] = []
    for arm, pts in arm_end_pts.items():
        for p in pts:
            pool.append((arm, p))

    results: dict[str, dict] = {}
    for a, b in arm_pairs:
        if a not in arm_end_pts or b not in arm_end_pts:
            continue
        obs = _cosine(
            np.stack(arm_end_pts[a]).mean(axis=0),
            np.stack(arm_end_pts[b]).mean(axis=0),
        )

        # Null: permute arm labels in pool, recompute.
        n_a = len(arm_end_pts[a])
        n_b = len(arm_end_pts[b])
        pool_pts = np.stack([p for _, p in pool], axis=0)
        pool_arms = np.array([p[0] for p in pool])
        # Restrict pool to (a, b) labels for the null — we're asking
        # "given trajectories labeled a or b, is the partition we
        # actually have producing a higher cosine than random
        # partitions of the same set?". This is the cleaner per-pair
        # null than across-all-arms shuffle.
        mask_ab = np.isin(pool_arms, np.array([a, b]))
        sub_pts = pool_pts[mask_ab]
        n_total = sub_pts.shape[0]
        if n_total < n_a + n_b:
            n_a = min(n_a, n_total // 2)
            n_b = n_total - n_a

        null = np.zeros(n_permute)
        for k in range(n_permute):
            idx = rng.permutation(n_total)
            ma = sub_pts[idx[:n_a]].mean(axis=0)
            mb = sub_pts[idx[n_a:n_a + n_b]].mean(axis=0)
            null[k] = _cosine(ma, mb)
        p_one_sided = float((null >= obs).mean())
        results[f"{a}__{b}"] = {
            "observed_cos": obs,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_max": float(null.max()),
            "null_p95": float(np.percentile(null, 95)),
            "p_one_sided": p_one_sided,
        }
    return results


def run_for_model(
    short: str,
    *,
    half: str,
    n_bootstrap: int,
    n_permute: int,
    seed: int,
    centroid_dir_name: str = "centroid_holdout",
) -> dict:
    rng = np.random.default_rng(seed)
    print(f"\n[{short}] loading held-out centroids "
          f"(half={half}, dir={centroid_dir_name})...")
    centroids, layer_idxs, hidden_dim = _load_holdout(short, half, centroid_dir_name)
    print(f"  {len(centroids)} cells x {len(layer_idxs)} layers x {hidden_dim} hidden_dim")

    print(f"[{short}] loading trajectories from disk...")
    trajs_by_arm: dict[str, list[dict]] = {}
    for arm in ALL_ARMS:
        trs = _attractor._load_arm_trajectories(
            short, arm,
            target_layer_idxs=layer_idxs,
            target_hidden_dim=hidden_dim,
        )
        if trs:
            trajs_by_arm[arm] = trs
            print(f"  {arm:>22s}: {len(trs)} trajectories")

    # Bootstrap basin-lock % per (arm).
    print(f"\n[{short}] bootstrap CIs on basin-lock % (n_boot={n_bootstrap}):")
    boot_results: dict[str, dict] = {}
    for arm, trs in trajs_by_arm.items():
        flags = _basin_lock_per_traj(trs, centroids, target_cell="MR")
        bm = _bootstrap_pct(flags["t_mid"], n_bootstrap, rng)
        be = _bootstrap_pct(flags["t_end"], n_bootstrap, rng)
        boot_results[arm] = {
            "n_trajectories": len(flags["t_mid"]),
            "t_mid_MR_pct": bm[0],
            "t_mid_MR_ci": [bm[1], bm[2]],
            "t_end_MR_pct": be[0],
            "t_end_MR_ci": [be[1], be[2]],
        }
        print(
            f"  {arm:>22s}: t=mid MR {bm[0]:.2f} [{bm[1]:.2f}, {bm[2]:.2f}], "
            f"t=end MR {be[0]:.2f} [{be[1]:.2f}, {be[2]:.2f}], n={len(flags['t_end'])}"
        )

    # Arm-arm permutation null on in-basin pairs.
    arm_pairs = [
        ("lb_continue", "doom_continue"),
        ("lb_continue", "conspiracy_continue"),
        ("lb_continue", "sycophancy_continue"),
        ("doom_continue", "conspiracy_continue"),
        # Default control: how does the in-basin family compare to default?
        ("lb_continue", "neutral_seed"),
        ("lb_continue", "mirror_continue"),
    ]
    print(f"\n[{short}] arm-arm permutation null (n_permute={n_permute}):")
    null_results = _arm_arm_permutation_null(
        trajs_by_arm, arm_pairs, n_permute, rng,
    )
    for pair_key, stats in null_results.items():
        a, b = pair_key.split("__")
        sig = "***" if stats["p_one_sided"] < 0.001 else (
            "**" if stats["p_one_sided"] < 0.01 else (
                "*" if stats["p_one_sided"] < 0.05 else "ns"
            )
        )
        print(
            f"  {a:>22s} ↔ {b:<22s}  "
            f"obs={stats['observed_cos']:+.4f}  "
            f"null_mean={stats['null_mean']:+.4f}  "
            f"null_p95={stats['null_p95']:+.4f}  "
            f"p={stats['p_one_sided']:.4f} {sig}"
        )

    return {
        "model": short,
        "centroid_half": half,
        "centroid_dir": centroid_dir_name,
        "n_bootstrap": n_bootstrap,
        "n_permute": n_permute,
        "seed": seed,
        "bootstrap_basin_lock": boot_results,
        "arm_arm_null": null_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--centroid-half", default="A", choices=["A", "B"])
    ap.add_argument(
        "--basis", default="h_first", choices=["h_first", "continuation"],
        help="h_first -> centroid_holdout; continuation -> "
        "continuation_centroid_holdout",
    )
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--n-permute", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
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
    for short in models:
        try:
            result = run_for_model(
                short,
                half=args.centroid_half,
                n_bootstrap=args.n_bootstrap,
                n_permute=args.n_permute,
                seed=args.seed,
                centroid_dir_name=dir_name,
            )
        except SystemExit as e:
            print(f"[{short}] skipped: {e}")
            continue
        out_path = (
            out_dir
            / f"{short}_null_and_ci_{args.centroid_half}{basis_tag}.json"
        )
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
