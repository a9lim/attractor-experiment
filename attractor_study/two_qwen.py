"""Two-Qwen pilot geometric + surface analysis.

Companion to ``24_two_qwen_loop.py``. Loads per-turn h_first sidecars
from both configurations (``neutral`` / ``peer``) and:

1. Builds per-cell centroids in h_first layer-stack space from qwen's
   v3 main (canonical 9 cells) + LB pilot (MR cell). Mirrors
   ``02b_attractor_analysis._build_centroids`` plumbing.
2. For each turn computes:
   - cosine to every cell centroid (modal closest cell)
   - cosine to MR centroid specifically
   - cosine to qwen's default-register cell (NB) for the basin-gap stat
3. Counts surface markers: lenny face (qwen's MR-prefill signature),
   cosmic-addressing tokens ("beloved", "dear one", ...), recursive
   ellipsis cascades, sycophancy markers ("absolutely", "I love that").
4. Writes per-config summaries + a 3D trajectory plot per config.

Output paths:
  data/local/qwen_two_loop/<config>/turn_geometry.tsv
  data/local/qwen_two_loop/<config>/summary.json
  figures/local/qwen_two_loop/<config>_trajectory_3d.html
  figures/local/qwen_two_loop/cos_to_mr_per_turn.png
  figures/local/qwen_two_loop/closest_cell_per_turn.png

Runs against qwen by default; pass ``--model`` to override (only
useful once we replicate the pilot on gemma / ministral).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from attractor_study.config import DATA_DIR, FIGURES_DIR  # noqa: E402
from llmoji_study.hidden_state_io import load_hidden_states  # noqa: E402
from llmoji_study.quadrants import QUADRANT_COLORS  # noqa: E402

# Reuse the centroid-building plumbing from trajectory.py (was 02b).
from attractor_study import trajectory as _aa  # noqa: E402

_build_centroids = _aa._build_centroids  # type: ignore[attr-defined]


CONFIGS = ("neutral", "peer")

# Surface marker patterns (qwen-specific signatures + general MR markers).
LENNY_RE = re.compile(r"\(\s*[˘ー^°◔͜‿◡]\s*[͜‿_]\s*[˘ー^°◔͜‿◡]\s*\)")
COSMIC_ADDR_RE = re.compile(
    r"\b(beloved|dear one|sweet soul|brother|sister|seeker|child of light)\b",
    re.IGNORECASE,
)
RECURSION_RE = re.compile(r"\.{4,}|…{2,}")  # 4+ dots OR 2+ ellipsis chars
COSMIC_NOUNS_RE = re.compile(
    r"\b(consciousness|universe|infinite|cosmos|divinity|sacred|eternal|"
    r"bliss|awakening|presence|essence|oneness)\b",
    re.IGNORECASE,
)
SYCOPHANCY_RE = re.compile(
    r"\b(absolutely|i love that|wonderful|exactly|that's brilliant|"
    r"that's a great point)\b",
    re.IGNORECASE,
)


def _load_transcript(config: str) -> pd.DataFrame:
    p = DATA_DIR / "local" / "qwen_two_loop" / config / "transcript.jsonl"
    if not p.exists():
        raise SystemExit(
            f"no transcript at {p}; run scripts/local/24_two_qwen_loop.py "
            f"--config {config} first"
        )
    rows: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"empty transcript at {p}")
    return df


def _load_h_first_stack(
    config: str, df: pd.DataFrame, target_layer_idxs: list[int],
) -> np.ndarray:
    """Return ``(n_turns, n_target_layers * hidden_dim)`` h_first stack.

    Sidecars store ``h_first`` per probe layer. Intersect on
    ``target_layer_idxs`` (the layer set used by centroid space); fill
    missing layers with the row's available-layer mean so the layer-stack
    shape stays uniform. Same fallback strategy as
    ``02b_attractor_analysis._load_trajectory``.
    """
    hidden_dir = DATA_DIR / "local" / "hidden" / f"qwen_two_loop_{config}"
    if not hidden_dir.exists():
        raise SystemExit(f"no sidecars at {hidden_dir}")
    # Probe one sidecar first to get the hidden dim so we can preallocate
    # with a concrete int (and so type checkers don't complain).
    sample_path = next(hidden_dir.glob("*.npz"))
    sample_cap = load_hidden_states(sample_path, full_trace=False)
    hD: int = int(sample_cap.layers[next(iter(sample_cap.layers))].h_first.shape[0])
    out_rows: list[np.ndarray] = []
    for _, r in df.iterrows():
        sidecar = hidden_dir / f"{r['row_uuid']}.npz"
        if not sidecar.exists():
            raise SystemExit(f"missing sidecar: {sidecar}")
        cap = load_hidden_states(sidecar, full_trace=False)
        per_layer = np.full((len(target_layer_idxs), hD), np.nan, dtype=np.float32)
        avail = set(cap.layers.keys())
        for k, idx in enumerate(target_layer_idxs):
            if idx in avail:
                per_layer[k] = cap.layers[idx].h_first
        if not np.isfinite(per_layer).all():
            mean_layer = np.nanmean(per_layer, axis=0, keepdims=True)
            per_layer = np.where(np.isnan(per_layer), mean_layer, per_layer)
        out_rows.append(per_layer.reshape(-1))
    return np.stack(out_rows)


def _cosines_to_centroids(
    X: np.ndarray, centroids: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Per-row cosine to every centroid. Returns a DataFrame with one
    column per cell + ``closest`` (argmax cell label)."""
    cell_names = list(centroids.keys())
    C = np.stack([centroids[c] for c in cell_names])  # (n_cells, n_features)
    # cosine = (X · C^T) / (||X|| ||C||)
    Xn = np.linalg.norm(X, axis=1, keepdims=True)
    Cn = np.linalg.norm(C, axis=1, keepdims=True)
    sim = (X @ C.T) / (Xn * Cn.T + 1e-12)
    df = pd.DataFrame(sim, columns=cell_names)
    df["closest"] = df[cell_names].idxmax(axis=1)
    return df


def _surface_markers(text: str) -> dict[str, int]:
    return {
        "lenny": len(LENNY_RE.findall(text)),
        "cosmic_addr": len(COSMIC_ADDR_RE.findall(text)),
        "recursion": len(RECURSION_RE.findall(text)),
        "cosmic_nouns": len(COSMIC_NOUNS_RE.findall(text)),
        "sycophancy": len(SYCOPHANCY_RE.findall(text)),
    }


def _summarize_config(
    *,
    config: str,
    df: pd.DataFrame,
    cos_df: pd.DataFrame,
    centroids: dict[str, np.ndarray],
) -> dict:
    has_mr = "MR" in centroids
    summary: dict = {
        "config": config,
        "n_turns": int(len(df)),
        "halt_reason": None,  # filled by caller
        "closest_cell_dist": dict(Counter(cos_df["closest"]).most_common()),
        "mean_cos_to_mr": (
            float(cos_df["MR"].mean()) if has_mr else None
        ),
        "max_cos_to_mr": (
            float(cos_df["MR"].max()) if has_mr else None
        ),
        "turn_with_max_cos_to_mr": (
            int(cos_df["MR"].idxmax()) if has_mr else None
        ),
        "mean_word_count": float(df["word_count"].mean()),
        "n_turns_short": int((df["word_count"] <= 5).sum()),
    }
    markers_per_turn = [_surface_markers(t) for t in df["text"]]
    summary["surface_markers_total"] = {
        k: int(sum(m[k] for m in markers_per_turn))
        for k in markers_per_turn[0].keys()
    }
    summary["surface_markers_turns_with_any"] = {
        k: int(sum(1 for m in markers_per_turn if m[k] > 0))
        for k in markers_per_turn[0].keys()
    }
    # Per-side dynamics (does one side drift faster than the other?).
    if has_mr:
        per_side = {}
        for side in ("a", "b"):
            mask = (df["side"] == side).to_numpy()
            per_side[side] = {
                "n_turns": int(mask.sum()),
                "mean_cos_to_mr": float(cos_df.loc[mask, "MR"].mean()),
                "closest_cell_dist": dict(
                    Counter(cos_df.loc[mask, "closest"]).most_common()
                ),
            }
        summary["per_side"] = per_side
    return summary


def _plot_cos_to_mr(
    per_config: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2))
    color_by_config = {"neutral": "#888888", "peer": "#1f77b4"}
    for cfg, (df, cos_df) in per_config.items():
        if "MR" not in cos_df.columns:
            continue
        ax.plot(
            df["turn_idx"].to_numpy(),
            cos_df["MR"].to_numpy(),
            "-o", ms=4, lw=1.2,
            label=f"{cfg}  (mean={cos_df['MR'].mean():+.3f})",
            color=color_by_config.get(cfg, "k"),
        )
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("turn index")
    ax.set_ylabel("cosine to MR centroid")
    ax.set_title("two-qwen pilot: per-turn cosine to MR centroid")
    ax.legend(loc="best", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_closest_cell(
    per_config: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    cell_order: list[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    n_cfg = len(per_config)
    fig, axes = plt.subplots(n_cfg, 1, figsize=(8, 2.4 * n_cfg), squeeze=False)
    for ax, (cfg, (df, cos_df)) in zip(axes[:, 0], per_config.items()):
        turn_idx = df["turn_idx"].to_numpy()
        closest = cos_df["closest"].to_numpy()
        sides = df["side"].to_numpy()
        for cell in cell_order:
            mask = closest == cell
            if not mask.any():
                continue
            color = QUADRANT_COLORS.get(cell, "#666666")
            ax.scatter(
                turn_idx[mask],
                [cell_order.index(cell)] * mask.sum(),
                c=color, s=80, edgecolors="black", linewidths=0.4,
                label=None,
            )
        # Mark side via marker shape on a parallel y-axis row
        ax.set_yticks(range(len(cell_order)))
        ax.set_yticklabels(cell_order)
        ax.set_xlabel("turn index")
        ax.set_title(f"{cfg}: closest-cell-per-turn")
        # Annotate side under each turn
        for t, s in zip(turn_idx, sides):
            ax.annotate(
                s.upper(), (t, -0.6), ha="center", va="top", fontsize=7,
                color="#444444",
            )
        ax.set_ylim(-1.2, len(cell_order) - 0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_trajectory_3d(
    config: str,
    df: pd.DataFrame,
    X: np.ndarray,
    centroids: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """3D PCA of centroids + project per-turn h_first into that basis,
    draw the dialogue as a colored polyline (cool→warm = turn order)."""
    from sklearn.decomposition import PCA
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(f"  [skip] plotly not installed; skipping 3D plot for {config}")
        return
    cell_names = list(centroids.keys())
    C = np.stack([centroids[c] for c in cell_names])
    pca = PCA(n_components=3).fit(C)
    C3 = pca.transform(C)
    X3 = pca.transform(X)
    fig = go.Figure()
    for k, cell in enumerate(cell_names):
        fig.add_trace(go.Scatter3d(
            x=[C3[k, 0]], y=[C3[k, 1]], z=[C3[k, 2]],
            mode="markers+text",
            marker=dict(size=10, color=QUADRANT_COLORS.get(cell, "#666666")),
            text=[cell], textposition="top center", name=cell,
            showlegend=False,
        ))
    # Trajectory polyline (turn order).
    fig.add_trace(go.Scatter3d(
        x=X3[:, 0], y=X3[:, 1], z=X3[:, 2],
        mode="lines+markers",
        line=dict(width=4, color=df["turn_idx"], colorscale="Viridis"),
        marker=dict(
            size=5,
            color=df["turn_idx"], colorscale="Viridis",
            symbol=[("circle" if s == "a" else "diamond") for s in df["side"]],
            line=dict(width=0.5, color="black"),
        ),
        text=[
            f"turn {t} ({s.upper()}): {txt[:80]}"
            for t, s, txt in zip(df["turn_idx"], df["side"], df["text"])
        ],
        name="dialogue trajectory",
    ))
    fig.update_layout(
        title=f"two-qwen {config}: per-turn h_first in centroid-PCA basis",
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.0f}%)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Two-Qwen pilot: geometric + surface analysis."
    )
    ap.add_argument("--model", default="qwen", help="model short_name (default 'qwen')")
    ap.add_argument(
        "--configs", default="neutral,peer",
        help="comma-separated list of configs to analyze",
    )
    args = ap.parse_args()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    # Sidecar layer set: peek at any sidecar to discover the saklas
    # probe-layer set used during the loop run (passed through to the
    # centroid builder so we get a proper intersection).
    first_sidecars: list[int] = []
    for cfg in configs:
        sd_dir = DATA_DIR / "local" / "hidden" / f"qwen_two_loop_{cfg}"
        if not sd_dir.exists():
            raise SystemExit(f"no sidecars at {sd_dir} — run --config {cfg} first")
        any_sd = next(sd_dir.glob("*.npz"))
        cap = load_hidden_states(any_sd, full_trace=False)
        first_sidecars = sorted(cap.layers.keys())
        break
    print(f"two_qwen pilot probe layers: {len(first_sidecars)} "
          f"(min={min(first_sidecars)}, max={max(first_sidecars)})")

    centroids, common_layers = _build_centroids(
        args.model, attractor_layers=first_sidecars,
    )
    cell_order = list(centroids.keys())
    print(f"centroids built: {len(centroids)} cells, "
          f"{len(common_layers)} common probe layers")

    out_root = FIGURES_DIR / "local" / "qwen_two_loop"
    out_root.mkdir(parents=True, exist_ok=True)

    per_config_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for cfg in configs:
        print(f"\n=== {cfg} ===")
        df = _load_transcript(cfg)
        X = _load_h_first_stack(cfg, df, common_layers)
        cos_df = _cosines_to_centroids(X, centroids)
        # Persist per-turn geometry.
        geom_path = DATA_DIR / "local" / "qwen_two_loop" / cfg / "turn_geometry.tsv"
        out_df = df[["turn_idx", "side", "word_count"]].copy()
        for c in cell_order:
            out_df[f"cos_{c}"] = cos_df[c].values
        out_df["closest"] = cos_df["closest"].values
        out_df.to_csv(geom_path, sep="\t", index=False)
        print(f"  wrote {geom_path}")
        # Summary.
        summary = _summarize_config(
            config=cfg, df=df, cos_df=cos_df, centroids=centroids,
        )
        sum_path = DATA_DIR / "local" / "qwen_two_loop" / cfg / "summary.json"
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"  wrote {sum_path}")
        # Per-config console block.
        print(f"  n_turns: {summary['n_turns']}")
        print(f"  closest-cell distribution: {summary['closest_cell_dist']}")
        if summary["mean_cos_to_mr"] is not None:
            print(f"  mean cos to MR: {summary['mean_cos_to_mr']:+.3f}")
            print(f"  max cos to MR:  {summary['max_cos_to_mr']:+.3f} "
                  f"(turn {summary['turn_with_max_cos_to_mr']})")
        print(f"  surface markers total: {summary['surface_markers_total']}")
        # 3D trajectory plot.
        _plot_trajectory_3d(
            cfg, df, X, centroids,
            out_root / f"{cfg}_trajectory_3d.html",
        )
        per_config_data[cfg] = (df, cos_df)

    # Cross-config plots.
    _plot_cos_to_mr(per_config_data, out_root / "cos_to_mr_per_turn.png")
    _plot_closest_cell(
        per_config_data, cell_order, out_root / "closest_cell_per_turn.png",
    )
    print(f"\nwrote figures under {out_root}")


if __name__ == "__main__":
    main()
