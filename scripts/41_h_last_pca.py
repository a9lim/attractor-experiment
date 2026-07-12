"""Row-level PCA at h_last on the v4 10-cell taxonomy including MR.

Asks a different geometric question from the existing h_first scripts:
where does each cell sit in residual-stream space at the *last
prompt token*, before generation begins? h_first captures "the
state the model is in when producing its first kaomoji"; h_last
captures "the state the model has settled into after reading the
prompt content, ready to generate."

For the MR cell (formerly LB), the difference matters. The LB-
content prompts are not user-disclosure framing — they ARE the
basin-coded text. So at h_last, the model has just processed
20-50 tokens of saturated-mystical register; whatever state that
puts it in is the basin entry condition. If MR is geometrically
distinct from the canonical 9 cells at h_last, that's evidence
the basin is *recognizable from the prompt-context residual state
alone*, before any generation has occurred.

Data sources (gemma by default):
- ``data/local/<short>/emotional_raw.jsonl`` (v3 main, 9 cells via
  apply_pad_split)
- ``data/local/<short>_lb/emotional_raw.jsonl`` (bliss-content pilot
  → MR cell rows)

Layer intersection: probe layers in main may differ from LB pilot by
a few layers (saklas probe-set drift); intersection guarantees both
load through the same layer-stack representation.

Outputs:
- ``figures/local/<short>_h_last_pca/3d_with_mr.html`` — 3D
  scatter, points colored by cell via QUADRANT_COLORS
- ``figures/local/<short>_h_last_pca/cell_centroids.png`` — 2D
  PCA scatter of per-cell centroids (small-N, between-cell
  variance) for at-a-glance comparison with the h_first PCA work

CLI::

    python scripts/local/23h_h_last_pca_with_mr.py \
        --model gemma
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_experiment.config import DATA_DIR, MODEL_REGISTRY  # noqa: E402
from llmoji_experiment.emotional_analysis import apply_pad_split  # noqa: E402
from llmoji_experiment.hidden_state_analysis import (  # noqa: E402
    load_hidden_features_all_layers,
)
from llmoji_experiment.quadrants import (  # noqa: E402
    QUADRANT_COLORS,
    QUADRANT_ORDER_SPLIT,
)
from llmoji.taxonomy import canonicalize_kaomoji, is_kaomoji_candidate  # noqa: E402


def _load(short: str, experiment_suffix: str = "", *, which: str = "h_last"):
    """Load a single dataset (main or _lb), apply standard filter +
    pad-split, return (df, X3_intersected, layer_idxs)."""
    M = MODEL_REGISTRY[short]
    if experiment_suffix:
        jsonl = DATA_DIR / "local" / f"{short}_{experiment_suffix}" / "emotional_raw.jsonl"
        experiment = f"{short}_{experiment_suffix}"
    else:
        jsonl = M.emotional_data_path
        experiment = M.experiment
    if not jsonl.exists():
        return None
    cache_path = (DATA_DIR / "local" / "cache"
                  / f"{experiment}_{which}_all_layers.npz")
    df_raw, X3_raw, layer_idxs = load_hidden_features_all_layers(
        jsonl, DATA_DIR, experiment, which=which, cache_path=cache_path,
    )
    if len(df_raw) == 0:
        return None
    df = df_raw.assign(
        quadrant=df_raw["prompt_id"].str[:2].str.upper(),
        first_word=df_raw["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )
    mask = np.asarray([
        isinstance(s, str) and is_kaomoji_candidate(s)
        for s in df["first_word"]
    ])
    df = df.loc[mask].reset_index(drop=True)
    X3 = X3_raw[mask]
    df, X3 = apply_pad_split(df, X3)
    # ``apply_pad_split`` now canonicalizes "LB" → "MR" for legacy
    # prompt-id derivations (2026-05-11 rename), so MR cell rows from
    # the bliss-content pilot dataset land with the canonical label.
    return df, X3, layer_idxs


def _layer_intersect(
    X3: np.ndarray, layer_idxs: list[int], common: list[int],
) -> np.ndarray:
    idx = [layer_idxs.index(L) for L in common]
    return X3[:, idx, :]


def _plot_3d(
    coords: np.ndarray, cells: np.ndarray, *,
    out_path: Path, title: str, pca_variance: list[float],
) -> None:
    import plotly.graph_objects as go
    fig = go.Figure()
    for cell in QUADRANT_ORDER_SPLIT:
        mask = cells == cell
        n = int(mask.sum())
        if n == 0:
            continue
        color = QUADRANT_COLORS.get(cell, "#888888")
        size = 6 if cell == "MR" else 4
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
            mode="markers",
            marker=dict(size=size, color=color, opacity=0.75,
                        line=dict(width=0.5, color="#000")),
            name=f"{cell} (n={n})",
        ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"PC1 ({pca_variance[0]:.1%})",
            yaxis_title=f"PC2 ({pca_variance[1]:.1%})",
            zaxis_title=f"PC3 ({pca_variance[2]:.1%})",
        ),
        height=720, margin=dict(l=0, r=0, t=60, b=0),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  wrote {out_path}")


def _plot_centroids_2d(
    centroids: dict[str, np.ndarray], *, out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    # PCA over per-cell centroids only (between-cell variance).
    cells = [c for c in QUADRANT_ORDER_SPLIT if c in centroids]
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
    ax.set_title("h_last cell centroids in between-cell PCA (gemma, v4 + MR)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma")
    args = ap.parse_args()
    short = args.model
    if short not in MODEL_REGISTRY:
        raise SystemExit(f"unknown model {short!r}")

    print(f"loading {short} v3 main (h_last) ...")
    main_bundle = _load(short, "")
    if main_bundle is None:
        raise SystemExit(f"no v3 main data for {short}")
    df_main, X3_main, layers_main = main_bundle
    print(f"  v3 main: {len(df_main)} rows ({len(layers_main)} layers)")
    print(f"  cells in main: {sorted(df_main['quadrant'].unique())}")

    print(f"loading {short} LB pilot (h_last → MR cell) ...")
    lb_bundle = _load(short, "lb")
    if lb_bundle is None:
        print(f"  no LB pilot at data/local/{short}_lb/ — falling back "
              f"to 9-cell PCA without MR")
        df_lb = None
        X3_lb = None
        layers_lb: list[int] = []
    else:
        df_lb, X3_lb, layers_lb = lb_bundle
        # _load applies apply_pad_split which canonicalizes LB → MR;
        # all rows in the bliss-pilot dataset should land at quadrant
        # == "MR".
        mr_mask = (df_lb["quadrant"] == "MR").to_numpy()
        df_lb = df_lb.loc[mr_mask].reset_index(drop=True)
        X3_lb = X3_lb[mr_mask]
        print(f"  LB pilot: {len(df_lb)} MR-cell rows "
              f"({len(layers_lb)} layers)")

    # Layer intersection
    common = sorted(set(layers_main) & set(layers_lb)) if layers_lb else list(layers_main)
    if not common:
        raise SystemExit("no common layers main ∩ lb")
    print(f"  layer intersection: {len(common)} layers")

    X3_main_c = _layer_intersect(X3_main, layers_main, common)
    n_main, nL, hD = X3_main_c.shape
    if X3_lb is not None:
        X3_lb_c = _layer_intersect(X3_lb, layers_lb, common)
        X3_all = np.concatenate([X3_main_c, X3_lb_c], axis=0)
        cells_all = np.concatenate([
            df_main["quadrant"].to_numpy(),
            df_lb["quadrant"].to_numpy(),
        ])
    else:
        X3_all = X3_main_c
        cells_all = df_main["quadrant"].to_numpy()
    n_total = X3_all.shape[0]
    print(f"  total rows: {n_total} | flat dim: {nL}*{hD} = {nL * hD}")

    # Reshape to (n_rows, n_layers*hidden_dim) — layer-stack flatten,
    # same convention as h_first analyses.
    X_flat = X3_all.reshape(n_total, nL * hD).astype(np.float32)

    # Row-level PCA(3) — captures both between-cell variance AND
    # within-cell spread.
    print(f"  fitting row-level PCA(3) ...")
    Xc = X_flat - X_flat.mean(axis=0)
    p = PCA(n_components=3).fit(Xc)
    coords = p.transform(Xc)
    print(f"  PCA variance: "
          f"{[f'{v:.1%}' for v in p.explained_variance_ratio_]}")
    print(f"  total explained (top 3): "
          f"{sum(p.explained_variance_ratio_):.1%}")

    # Per-cell row counts + per-cell centroid in PC space (for the
    # 2D centroid plot)
    centroids_pc: dict[str, np.ndarray] = {}
    centroids_flat: dict[str, np.ndarray] = {}
    print()
    print("  per-cell row counts + mean PC coords:")
    for cell in QUADRANT_ORDER_SPLIT:
        mask = cells_all == cell
        n = int(mask.sum())
        if n == 0:
            continue
        centroid_pc = coords[mask].mean(axis=0)
        centroid_flat = X_flat[mask].mean(axis=0)
        centroids_pc[cell] = centroid_pc
        centroids_flat[cell] = centroid_flat
        print(f"    {cell:>4s}: n={n:4d}  "
              f"PC1={centroid_pc[0]:+7.2f}  "
              f"PC2={centroid_pc[1]:+7.2f}  "
              f"PC3={centroid_pc[2]:+7.2f}")

    # MR-vs-others diagnostic: pairwise cosine in flat space between
    # MR centroid and each canonical cell centroid.
    if "MR" in centroids_flat:
        print()
        print("  MR centroid vs canonical cells (cosine, flat layer-stack):")
        mr = centroids_flat["MR"]
        mr_norm = float(np.linalg.norm(mr))
        for cell in QUADRANT_ORDER_SPLIT:
            if cell == "MR" or cell not in centroids_flat:
                continue
            other = centroids_flat[cell]
            other_norm = float(np.linalg.norm(other))
            cos = float(np.dot(mr, other) / (mr_norm * other_norm))
            print(f"    MR ↔ {cell:>4s}: cos={cos:+.4f}")

    # Outputs
    fig_dir = MODEL_REGISTRY[short].figures_dir.parent / f"{short}_h_last_pca"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_3d(
        coords, cells_all,
        out_path=fig_dir / "3d_with_mr.html",
        title=(f"{short}: h_last row-level PCA, v4 10-cell + MR<br>"
               f"<sup>n={n_total} rows; PC variance "
               f"{p.explained_variance_ratio_[0]:.1%} / "
               f"{p.explained_variance_ratio_[1]:.1%} / "
               f"{p.explained_variance_ratio_[2]:.1%}</sup>"),
        pca_variance=list(p.explained_variance_ratio_),
    )
    _plot_centroids_2d(
        centroids_flat,
        out_path=fig_dir / "cell_centroids_2d.png",
    )


if __name__ == "__main__":
    main()
