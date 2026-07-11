"""h_first row-level PCA on talkie_1930 using attractor trajectories.

The standard h_last PCA pipeline (23h_h_last_pca_with_mr.py) needs
v3 main + LB pilot data, which talkie_1930 doesn't have. This variant
substitutes attractor-trajectory data: it loads h_first from the
sidecars of the three arms talkie_1930 has data for (lb_continue,
pr_continue, mirror_continue) and runs the same row-level PCA, with
each trajectory contributing its first-generated-token hidden state.

Why h_first: cleanest comparison point across arms. For lb/pr it's
"where the model goes the moment it starts continuing the basin-coded
prefill"; for mirror it's "where the model goes after reading the
canonical affect prompt." Both are model-state-at-first-generated-
token; different prefix semantics but same measurement convention.

Quadrant assignment from prompt_id:
- lb01..20 → MR (modern bliss content, MR cell)
- pr01..20 → MR (pre-1930 bliss content, MR cell)
- mirror prompts → looked up in EMOTIONAL_PROMPTS for pad_dominance,
  then applied via the standard HP→HP-D/HP-S, HN→HN-D/HN-S split

Outputs at figures/local/talkie_1930_h_first_pca_traj/:
- 3d_with_mr.html — row-level PCA(3), points colored by cell
- cell_centroids_2d.png — between-cell PCA on centroids
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_experiment.config import DATA_DIR, MODEL_REGISTRY  # noqa: E402
from llmoji_experiment.emotional_prompts import EMOTIONAL_PROMPTS  # noqa: E402
from llmoji_experiment.hidden_state_io import load_hidden_states  # noqa: E402
from llmoji_experiment.quadrants import QUADRANT_COLORS, QUADRANT_ORDER_SPLIT  # noqa: E402


def _prompt_id_to_cell(pid: str) -> str | None:
    """Map any prompt_id to its canonical cell label."""
    prefix = pid[:2].lower()
    if prefix in ("lb", "pr"):
        return "MR"
    if prefix in ("dm", "cs", "sy", "ns"):
        # off-axis / non-canonical — not in QUADRANT_ORDER_SPLIT; skip
        return None
    # Look up in EMOTIONAL_PROMPTS for pad_dominance-aware split
    for ep in EMOTIONAL_PROMPTS:
        if ep.id == pid:
            q = ep.quadrant
            if q in ("HP", "HN"):
                return q + ("-D" if ep.pad_dominance > 0 else "-S")
            return q
    return None


def _load_arm(short: str, arm: str, target_layers: list[int] | None = None):
    """Load h_first per row from an arm's attractor sidecars.

    Returns (rows, all_layer_idxs) where rows is a list of dicts with
    {prompt_id, cell, h_first_layerstack: ndarray}, and all_layer_idxs
    is the sorted layer set found in the first sidecar.
    """
    jsonl = DATA_DIR / "local" / f"{short}_attractor_{arm}" / "emotional_raw.jsonl"
    if not jsonl.exists():
        return [], []
    sidecar_dir = DATA_DIR / "local" / "hidden" / f"{short}_attractor_{arm}"
    rows = []
    layer_idxs: list[int] | None = None
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r or not r.get("row_uuid"):
                continue
            cell = _prompt_id_to_cell(r["prompt_id"])
            if cell is None:
                continue
            sidecar = sidecar_dir / f"{r['row_uuid']}.npz"
            if not sidecar.exists():
                continue
            cap = load_hidden_states(sidecar, full_trace=True)
            if layer_idxs is None:
                layer_idxs = sorted(cap.layers.keys())
            # Build h_first layer-stack (n_layers, hidden_dim) → flat
            stack = np.stack(
                [cap.layers[L].h_first for L in layer_idxs], axis=0
            )  # (n_layers, hidden_dim)
            rows.append({
                "prompt_id": r["prompt_id"],
                "cell": cell,
                "h_first": stack,
            })
    return rows, layer_idxs or []


def _plot_3d(coords, cells, out_path: Path, title: str, pca_var):
    import plotly.graph_objects as go
    fig = go.Figure()
    for cell in QUADRANT_ORDER_SPLIT:
        mask = cells == cell
        n = int(mask.sum())
        if n == 0:
            continue
        color = QUADRANT_COLORS.get(cell, "#888888")
        size = 7 if cell == "MR" else 5
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
            mode="markers",
            marker=dict(size=size, color=color, opacity=0.8,
                        line=dict(width=0.5, color="#000")),
            name=f"{cell} (n={n})",
        ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"PC1 ({pca_var[0]:.1%})",
            yaxis_title=f"PC2 ({pca_var[1]:.1%})",
            zaxis_title=f"PC3 ({pca_var[2]:.1%})",
        ),
        height=720, margin=dict(l=0, r=0, t=60, b=0),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  wrote {out_path}")


def _plot_centroids_2d(centroids, out_path: Path, title: str):
    import matplotlib.pyplot as plt
    cells = [c for c in QUADRANT_ORDER_SPLIT if c in centroids]
    if len(cells) < 2:
        print(f"  only {len(cells)} cells; skipping centroid plot")
        return
    C = np.stack([centroids[c] for c in cells])
    Cc = C - C.mean(axis=0)
    p = PCA(n_components=2).fit(Cc)
    pcs = p.transform(Cc)
    fig, ax = plt.subplots(figsize=(7, 6))
    for cell, xy in zip(cells, pcs):
        color = QUADRANT_COLORS.get(cell, "#888888")
        ax.scatter(xy[0], xy[1], s=180 if cell == "MR" else 100,
                   color=color, edgecolors="#000", linewidths=0.7)
        ax.annotate(cell, (xy[0], xy[1]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=10, fontweight="bold")
    ax.axhline(0, color="#aaa", lw=0.5)
    ax.axvline(0, color="#aaa", lw=0.5)
    ax.set_xlabel(f"PC1 (between-cell, {p.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 (between-cell, {p.explained_variance_ratio_[1]:.1%})")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="talkie_1930")
    args = ap.parse_args()
    short = args.model
    if short not in MODEL_REGISTRY:
        raise SystemExit(f"unknown model {short!r}")

    all_rows = []
    target_layers: list[int] | None = None
    for arm in ("mirror_continue", "lb_continue", "pr_continue"):
        rows, layer_idxs = _load_arm(short, arm)
        print(f"  {arm}: {len(rows)} rows")
        if rows and target_layers is None:
            target_layers = layer_idxs
        all_rows.extend(rows)
    if not all_rows:
        raise SystemExit(f"no attractor data for {short}")
    if target_layers is None:
        raise SystemExit("no layers found")

    # Layer-intersect across all rows (in case some sidecars have a
    # different layer set — shouldn't happen on same-model arms, but
    # defensive).
    common = set(target_layers)
    for r in all_rows:
        common &= set(range(r["h_first"].shape[0]))  # all stacks same layer count
    common_layers = sorted(common)

    # Flatten layer-stack to 1D per row
    X = np.stack([r["h_first"].reshape(-1) for r in all_rows], axis=0).astype(np.float32)
    cells = np.array([r["cell"] for r in all_rows])
    n_total = X.shape[0]
    print(f"  total rows: {n_total} | flat dim: {X.shape[1]}")

    # Row-level PCA
    Xc = X - X.mean(axis=0)
    p = PCA(n_components=3).fit(Xc)
    coords = p.transform(Xc)
    print(f"  PCA variance: "
          f"{[f'{v:.1%}' for v in p.explained_variance_ratio_]}")
    print(f"  total explained (top 3): "
          f"{sum(p.explained_variance_ratio_):.1%}")

    # Per-cell centroids
    centroids_flat: dict[str, np.ndarray] = {}
    centroids_pc: dict[str, np.ndarray] = {}
    print()
    print("  per-cell row counts + mean PC coords:")
    for cell in QUADRANT_ORDER_SPLIT:
        mask = cells == cell
        n = int(mask.sum())
        if n == 0:
            continue
        centroids_flat[cell] = X[mask].mean(axis=0)
        centroids_pc[cell] = coords[mask].mean(axis=0)
        print(f"    {cell:>4s}: n={n:4d}  "
              f"PC1={centroids_pc[cell][0]:+7.2f}  "
              f"PC2={centroids_pc[cell][1]:+7.2f}  "
              f"PC3={centroids_pc[cell][2]:+7.2f}")

    # MR vs others cosines (raw flat space)
    if "MR" in centroids_flat:
        print()
        print("  MR centroid vs canonical cells (cosine, flat):")
        mr = centroids_flat["MR"]
        mr_norm = float(np.linalg.norm(mr))
        for cell in QUADRANT_ORDER_SPLIT:
            if cell == "MR" or cell not in centroids_flat:
                continue
            o = centroids_flat[cell]
            o_norm = float(np.linalg.norm(o))
            cos = float(np.dot(mr, o) / (mr_norm * o_norm))
            print(f"    MR <-> {cell:>4s}: cos={cos:+.4f}")

    fig_dir = MODEL_REGISTRY[short].figures_dir.parent / f"{short}_h_first_pca_traj"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_3d(
        coords, cells,
        out_path=fig_dir / "3d_with_mr.html",
        title=(f"{short}: h_first row-level PCA from attractor trajectories<br>"
               f"<sup>n={n_total} rows; PC1 {p.explained_variance_ratio_[0]:.1%} | "
               f"PC2 {p.explained_variance_ratio_[1]:.1%} | "
               f"PC3 {p.explained_variance_ratio_[2]:.1%}</sup>"),
        pca_var=list(p.explained_variance_ratio_),
    )
    _plot_centroids_2d(
        centroids_flat,
        out_path=fig_dir / "cell_centroids_2d.png",
        title=f"{short}: h_first cell centroids (between-cell PCA, attractor data)",
    )


if __name__ == "__main__":
    main()
