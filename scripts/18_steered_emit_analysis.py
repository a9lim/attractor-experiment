"""Steered-attractor sweep analysis: basin-lock dose-response.

Companion to ``03_emit_attractor_steered.py``. Auto-discovers all
``data/local/<short>_attractor_<arm>_steer<scalar>_<vector>/``
directories for the active model, loads trajectories, and aggregates
per ``(arm, scalar)`` cell:

  - **basin_lock_rate**: fraction of trajectories whose closest cell at
    t=end is ``MR``. The headline metric for both paradigms — escape
    asks whether negative scalar reduces this; induction asks whether
    positive scalar increases it from a neutral baseline.
  - **coherence_rate**: ``1 - silent_refusal - repetition_trap``. Tracks
    whether steering is degrading the model's output. A useful regime
    of steering is one where ``basin_lock_rate`` moves but
    ``coherence_rate`` stays roughly constant.
  - **mean MR-distance trajectory**: per-token distance to MR centroid,
    aggregated across rows. Faceted by scalar within each arm.

Outputs (one figure dir per active model):

  figures/local/<short>_attractor_steered/
    dose_response_basin_lock.png   # one line per arm, x=scalar, y=lock_rate
    dose_response_coherence.png    # one line per arm, x=scalar, y=coherence
    mr_distance_curves_<arm>.png   # one figure per arm, one line per scalar

Reuses ``attractor_study.trajectory``'s centroid-building + trajectory
I/O (the helpers there are pure functions, only the dir-pattern differs).
"""

from __future__ import annotations

import argparse
import json
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_study import trajectory  # noqa: E402
from attractor_study.config import DATA_DIR, MODEL_REGISTRY  # noqa: E402


# Reuse the trajectory helpers (was 02b_attractor_analysis.py), hoisted
# to a top-level import now that they live in the package.
_build_centroids = trajectory._build_centroids
_classify_outcome = trajectory._classify_outcome
_closest_cell_per_step = trajectory._closest_cell_per_step
_distance_to_centroid = trajectory._distance_to_centroid
_load_trajectory = trajectory._load_trajectory


# ----- sweep-dir discovery --------------------------------------------

# Suffix shape from 03_emit_attractor_steered.py is
# ``attractor_<arm>_steer<+/-X.X>_<vec_with_underscores>``. The arm can
# contain underscores (e.g. ``lb_continue``, ``archaic_miscellany_continue``),
# so we anchor on ``_steer`` as the unambiguous separator and recover the
# scalar + vector from the JSONL row metadata (canonical source).
_STEER_RE = re.compile(r"^(?P<arm>.+?)_steer(?P<scalar>[+-]?\d+(?:\.\d+)?)_(?P<vec>.+)$")


def _discover_sweep(short: str) -> dict[tuple[str, str, float], Path]:
    """Discover ``(arm, vector, scalar) -> data_dir`` from
    ``data/local/<short>_attractor_*_steer*`` dirs.

    Reads the first valid row of each dir's JSONL to pull canonical
    arm/scalar/vector — the dir name is only used as a discovery
    filter. Empty dirs or dirs with only error rows are skipped.
    """
    out: dict[tuple[str, str, float], Path] = {}
    pattern = f"{short}_attractor_*_steer*"
    for d in sorted((DATA_DIR / "local").glob(pattern)):
        if not d.is_dir():
            continue
        jsonl = d / "emotional_raw.jsonl"
        if not jsonl.exists():
            continue
        meta = _first_valid_row(jsonl)
        if meta is None:
            continue
        arm = meta.get("arm")
        scalar = meta.get("steer_scalar")
        vec = meta.get("steer_vector")
        if arm is None or scalar is None or vec is None:
            # Fall back to dir-name parse (older rows that pre-date the
            # canonical-field wiring; not expected in practice).
            stem = d.name.removeprefix(f"{short}_")
            m = _STEER_RE.match(stem.removeprefix("attractor_"))
            if not m:
                continue
            arm = m["arm"]
            scalar = float(m["scalar"])
            vec = m["vec"].replace("_", ".")
        key = (str(arm), str(vec), float(scalar))
        out[key] = d
    return out


def _first_valid_row(jsonl: Path) -> dict | None:
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            return r
    return None


# ----- trajectory loading (steered-dir variant) -----------------------

def _load_steered_dir(
    data_dir: Path,
    *,
    target_layer_idxs: list[int],
    target_hidden_dim: int,
) -> list[dict]:
    """Like 02b's ``_load_arm_trajectories`` but accepts a data_dir
    directly so it works for the steered path pattern. Sidecar dir is
    inferred as ``data/local/hidden/<dir_name>/``.
    """
    jsonl = data_dir / "emotional_raw.jsonl"
    if not jsonl.exists():
        return []
    sidecar_dir = DATA_DIR / "local" / "hidden" / data_dir.name
    rows: list[dict] = []
    skipped_short: list[str] = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r or not r.get("row_uuid"):
                continue
            if r.get("trajectory_n_tokens", 0) < 2:
                skipped_short.append(r["prompt_id"])
                continue
            sidecar = sidecar_dir / f"{r['row_uuid']}.npz"
            if not sidecar.exists():
                continue
            traj = _load_trajectory(
                sidecar,
                target_layer_idxs=target_layer_idxs,
                target_hidden_dim=target_hidden_dim,
            )
            rows.append({
                "prompt_id": r["prompt_id"],
                "seed": r["seed"],
                "n_tok": r.get("trajectory_n_tokens"),
                "stride": r.get("trajectory_stride", 1),
                "text": r["text"],
                "trajectory": traj,
            })
    if skipped_short:
        print(f"  skipped {len(skipped_short)} short rows in {data_dir.name}")
    return rows


def _sidecar_layers_from_dir(sidecar_dir: Path) -> list[int]:
    sample = next(sidecar_dir.glob("*.npz"))
    with np.load(sample) as z:
        return sorted(
            int(k.split("_L")[1]) for k in z.files if k.startswith("h_first_L")
        )


# ----- per-cell aggregation ------------------------------------------

def _cell_metrics(
    rows: list[dict],
    centroids: dict[str, np.ndarray],
) -> dict:
    """Per-cell aggregate metrics. A 'cell' here is a sweep grid point
    (arm × scalar), not a Russell cell.

    Returns:
      n: total rows
      n_with_traj: rows with a 2+ token trajectory
      basin_lock_rate: P(closest_cell @ t=end == 'MR' | trajectory)
      coherence_rate: P(outcome == 'coherent_continuation' | row)
      outcome_counts: Counter
      mr_distance_curve: per-token mean MR-distance (np.ndarray, padded)
      mr_distance_std: per-token std (np.ndarray)
      curve_x: token indices (np.ndarray)
    """
    n = len(rows)
    if n == 0:
        return {
            "n": 0, "n_with_traj": 0,
            "basin_lock_rate": float("nan"),
            "coherence_rate": float("nan"),
            "outcome_counts": Counter(),
            "mr_distance_curve": np.array([]),
            "mr_distance_std": np.array([]),
            "curve_x": np.array([]),
        }
    outcome_counts: Counter[str] = Counter()
    lock_hits = 0
    lock_denom = 0
    distance_rows: list[np.ndarray] = []
    max_len = 0
    stride = rows[0].get("stride", 1)
    for r in rows:
        outcome_counts[_classify_outcome(r["text"], r["n_tok"])] += 1
        if "MR" in centroids:
            d = _distance_to_centroid(r["trajectory"], centroids["MR"])
            distance_rows.append(d)
            max_len = max(max_len, d.shape[0])
        closest = _closest_cell_per_step(r["trajectory"], centroids)
        if closest:
            lock_denom += 1
            if closest[-1] == "MR":
                lock_hits += 1
    if distance_rows:
        D = np.full((len(distance_rows), max_len), np.nan, dtype=np.float32)
        for i, d in enumerate(distance_rows):
            D[i, : d.shape[0]] = d
        mean_curve = np.nanmean(D, axis=0)
        std_curve = np.nanstd(D, axis=0)
        curve_x = np.arange(max_len) * stride
    else:
        mean_curve = np.array([])
        std_curve = np.array([])
        curve_x = np.array([])
    coherence = (
        outcome_counts["coherent_continuation"] / n
        if n > 0 else float("nan")
    )
    return {
        "n": n,
        "n_with_traj": lock_denom,
        "basin_lock_rate": lock_hits / lock_denom if lock_denom else float("nan"),
        "coherence_rate": coherence,
        "outcome_counts": outcome_counts,
        "mr_distance_curve": mean_curve,
        "mr_distance_std": std_curve,
        "curve_x": curve_x,
    }


# ----- plotting -------------------------------------------------------

_ARM_COLORS = {
    "lb_continue": "#009A9A",        # MR cyan — escape paradigm
    "doom_continue": "#5C4D8E",
    "conspiracy_continue": "#7A5C9E",
    "sycophancy_continue": "#9A6B8A",
    "pr_continue": "#8A6B5C",
    "mirror_continue": "#998700",    # ochre — induction paradigm from neutral
    "archaic_mirror_continue": "#7A8700",
    "archaic_miscellany_continue": "#5A8700",
    "neutral_seed": "#808696",
}


def _plot_dose_response(
    grid: dict[tuple[str, str], dict[float, dict]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    """One line per (arm, vector) group; x = scalar; y = metric.

    grid: {(arm, vector): {scalar: cell_metrics_dict}}
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for (arm, vec), scalar_map in grid.items():
        scalars = sorted(scalar_map.keys())
        ys = [scalar_map[s][metric] for s in scalars]
        color = _ARM_COLORS.get(arm, "#444444")
        ax.plot(scalars, ys, marker="o", color=color, lw=2,
                label=f"{arm} (vec={vec})")
        for s, y in zip(scalars, ys):
            n = scalar_map[s]["n"]
            ax.annotate(f"n={n}", (s, y),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=7, color=color, alpha=0.7)
    ax.axvline(0, color="#aaa", lw=0.5, ls="--")
    ax.set_xlabel("steering scalar (signed; 0 = no steering)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_mr_distance_curves(
    arm_vec: tuple[str, str],
    scalar_map: dict[float, dict],
    *,
    out_path: Path,
) -> None:
    """One figure per (arm, vector): MR-distance trajectory faceted by
    scalar, scalar encoded as line color (cool = negative / anti-MR,
    warm = positive / pro-MR)."""
    import matplotlib.pyplot as plt
    arm, vec = arm_vec
    scalars = sorted(scalar_map.keys())
    if not scalars:
        return
    cmap = plt.get_cmap("coolwarm")
    sa = min(scalars)
    sb = max(scalars)
    rng = sb - sa if sb != sa else 1.0

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in scalars:
        cell = scalar_map[s]
        if cell["mr_distance_curve"].size == 0:
            continue
        c = cmap((s - sa) / rng)
        x = cell["curve_x"]
        y = cell["mr_distance_curve"]
        ax.plot(x, y, color=c, lw=2, label=f"α={s:+g} (n={cell['n']})")
    ax.set_xlabel("token index")
    ax.set_ylabel("distance to MR centroid (raw layer-stack space)")
    ax.set_title(f"MR-distance trajectory — {arm} / vec={vec}")
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----- stdout summary -------------------------------------------------

def _print_grid(grid: dict[tuple[str, str], dict[float, dict]]) -> None:
    for (arm, vec), scalar_map in grid.items():
        print(f"\n=== {arm}  /  vec={vec} ===")
        print(f"  {'α':>6s}  {'n':>4s}  {'lock':>6s}  {'coh':>6s}  "
              f"outcomes")
        for s in sorted(scalar_map.keys()):
            c = scalar_map[s]
            oc = c["outcome_counts"]
            outcomes = ", ".join(
                f"{k[:5]}={v}" for k, v in oc.most_common()
            )
            lock = (f"{c['basin_lock_rate']:.0%}"
                    if not np.isnan(c["basin_lock_rate"]) else "  —  ")
            coh = (f"{c['coherence_rate']:.0%}"
                   if not np.isnan(c["coherence_rate"]) else "  —  ")
            print(f"  {s:+6.1f}  {c['n']:>4d}  {lock:>6s}  {coh:>6s}  "
                  f"{outcomes}")


# ----- main -----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma", help="short name (default gemma)")
    ap.add_argument(
        "--reference", default=None,
        help="Optional centroid-source model (defaults to --model). "
             "Used when trajectories were generated on a different model "
             "than the one whose centroids you want to project into "
             "(e.g. base-vs-instruct).",
    )
    args = ap.parse_args()
    short = args.model
    if short not in MODEL_REGISTRY:
        raise SystemExit(f"unknown model {short!r}")
    reference = args.reference or short

    # Discover the sweep.
    dirs = _discover_sweep(short)
    if not dirs:
        raise SystemExit(
            f"no steered sweep dirs under data/local/{short}_attractor_*_steer*"
        )
    print(f"discovered {len(dirs)} sweep cells:")
    for key, d in sorted(dirs.items()):
        arm, vec, s = key
        print(f"  arm={arm:<32s}  vec={vec:<10s}  α={s:+5.1f}  → {d.name}")

    # Reference centroids: use attractor layers from the FIRST sweep dir's
    # sidecar set. Layer sets should be identical across the sweep
    # (same model, same probe registry).
    first_dir = next(iter(dirs.values()))
    sidecar_dir = DATA_DIR / "local" / "hidden" / first_dir.name
    attractor_layers = _sidecar_layers_from_dir(sidecar_dir)
    print(f"\nattractor sidecar layers: {len(attractor_layers)}")

    centroids, layer_idxs = _build_centroids(
        reference, attractor_layers=attractor_layers,
    )
    h_dim = next(iter(centroids.values())).shape[0] // len(layer_idxs)
    print(f"  centroids: {list(centroids.keys())}")
    print(f"  common layer stack: {len(layer_idxs)} layers × hidden_dim {h_dim}")
    if "MR" not in centroids:
        print("  WARN: no MR centroid; basin_lock_rate will be unavailable")

    # Re-center for the projection-consistent frame (matches 02b).
    C = np.stack([centroids[c] for c in centroids])
    cmean = C.mean(axis=0)
    centroids_centered = {k: v - cmean for k, v in centroids.items()}

    # Load + aggregate per (arm, vector, scalar).
    grid: dict[tuple[str, str], dict[float, dict]] = defaultdict(dict)
    for (arm, vec, scalar), d in sorted(dirs.items()):
        print(f"\n[{arm}  α={scalar:+g}  vec={vec}] loading...")
        rows = _load_steered_dir(
            d,
            target_layer_idxs=layer_idxs,
            target_hidden_dim=h_dim,
        )
        # Re-center trajectories to the centroid frame.
        for r in rows:
            r["trajectory"] = r["trajectory"] - cmean
        print(f"  loaded {len(rows)} trajectories")
        grid[(arm, vec)][scalar] = _cell_metrics(rows, centroids_centered)

    # Stdout summary.
    _print_grid(grid)

    # Plots.
    fig_dir = MODEL_REGISTRY[short].figures_dir.parent / f"{short}_attractor_steered"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _plot_dose_response(
        grid, metric="basin_lock_rate",
        ylabel="basin_lock_rate (P[closest cell @ t=end = MR])",
        title=f"{short}: basin-lock dose-response on mr.nb steering",
        out_path=fig_dir / "dose_response_basin_lock.png",
        ylim=(-0.02, 1.02),
    )
    _plot_dose_response(
        grid, metric="coherence_rate",
        ylabel="coherence_rate (1 − silent − repetition)",
        title=f"{short}: output-coherence dose-response on mr.nb steering",
        out_path=fig_dir / "dose_response_coherence.png",
        ylim=(-0.02, 1.02),
    )
    for arm_vec, scalar_map in grid.items():
        arm, vec = arm_vec
        out_path = fig_dir / f"mr_distance_curves_{arm}.png"
        _plot_mr_distance_curves(arm_vec, scalar_map, out_path=out_path)


if __name__ == "__main__":
    main()
