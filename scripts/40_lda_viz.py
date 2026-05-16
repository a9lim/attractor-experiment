"""LDA-basis attractor viz, **continuation basis** (PROJECT_2 §1.6).

Sibling to ``42_lda_attractor_viz.py``. Same PCA-then-LDA pipeline,
same per-arm 3D figure layout, but the LDA training set is per-
trajectory ``h_last`` vectors instead of v3 main / LB pilot kaomoji-
emission rows.

Why this matters: the original ``42_lda_attractor_viz.py`` fits LDA
on h_first rows, so even though the centroid separation in LDA basis
improves 3-5× over centroid-PCA, the *trajectory* projection
still lands wherever the trajectory's h_first-basis representation
sits — which (per §1.6's session-2 distance audit) is far from the
h_first MR centroid. Continuation-basis LDA fits on the same kind
of state the trajectories produce, so trajectory end-points should
co-locate with MR centroid when the basis is doing its job.

Data sources:

- Canonical 9 cells: per-trajectory ``h_last`` from
  ``mirror_continue``, labeled by the prompt's PAD-split cell.
- MR cell: per-trajectory ``h_last`` from ``lb_continue`` +
  ``doom_continue`` + ``conspiracy_continue`` + ``sycophancy_continue``.

Hold-out contract: requires
``data/local/continuation_centroid_holdout/{model}/manifest.json``
built by ``40b_continuation_centroid_holdout.py``. Reuses the
A-half prompt-key split from that manifest as the LDA training set
(B-half rows are held out for any future hold-out test).

Output:
  figures/local/{model}_attractor_lda_continuation/<arm>_trajectories_3d.html
  figures/local/{model}_attractor_lda_continuation/comparison_centroids_2d.html
  data/local/basin_stats/{model}_lda_separation_continuation.json

Usage:
  .venv/bin/python scripts/local/42b_lda_attractor_viz_continuation.py --model gemma
  .venv/bin/python scripts/local/42b_lda_attractor_viz_continuation.py --all-models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_study import trajectory as _attractor  # noqa: E402

# Reuse the centroids loaders (was 40b) so the LDA training set is
# byte-identical to the centroid-construction set, modulo the A/B split.
from attractor_study import centroids as _holdout  # noqa: E402

from attractor_study.config import DATA_DIR, FIGURES_DIR, MODEL_REGISTRY  # noqa: E402
from llmoji_study.quadrants import QUADRANT_COLORS  # noqa: E402


# ----- training rows -------------------------------------------------

def _load_training_rows(
    model: str, manifest: dict,
) -> tuple[np.ndarray, list[str]]:
    """Load A-half per-trajectory h_last vectors + cell labels.

    Mirrors ``40b._load_h_last_per_arm`` + the A-half filtering
    from the manifest's ``splits[cell]['A_keys']`` field.
    """
    layer_idxs = list(manifest["layer_idxs"])
    splits = manifest["splits"]

    rows_X: list[np.ndarray] = []
    rows_y: list[str] = []

    # Canonical: mirror_continue per cell.
    print(f"  loading mirror_continue h_last...")
    mirror_rows = _holdout._load_h_last_per_arm(model, _holdout.CANONICAL_ARM)
    if mirror_rows is None:
        raise SystemExit("no mirror_continue arm")
    for r in mirror_rows:
        r["cell"] = _holdout._pad_split_cell(r["prompt_id"])
    mirror_rows = [r for r in mirror_rows if r["cell"] is not None]

    # MR: pool across in-basin arms.
    mr_rows: list[dict] = []
    for arm in _holdout.MR_SOURCE_ARMS:
        print(f"  loading {arm} h_last...")
        rows = _holdout._load_h_last_per_arm(model, arm)
        if rows is None:
            continue
        for r in rows:
            r["cell"] = "MR"
            r["arm"] = arm
        mr_rows.extend(rows)

    all_rows = mirror_rows + mr_rows
    cells_per_row = [r["cell"] for r in all_rows]
    print(f"  total rows: {len(all_rows)}; cell counts: "
          f"{ {c: cells_per_row.count(c) for c in sorted(set(cells_per_row))} }")

    # Filter to A-half using the manifest's split keys.
    for r in all_rows:
        cell = r["cell"]
        if cell not in splits:
            continue
        key = str((r.get("arm", _holdout.CANONICAL_ARM), r["prompt_id"]))
        if key in splits[cell]["A_keys"]:
            flat = _holdout._flatten(r, layer_idxs)
            rows_X.append(flat)
            rows_y.append(cell)

    if not rows_X:
        raise SystemExit("no A-half training rows assembled")
    X = np.stack(rows_X).astype(np.float32)
    print(f"  A-half training set: {X.shape[0]} rows × {X.shape[1]} dims")
    return X, rows_y


def _load_holdout_centroids(model: str, half: str = "A") -> dict[str, np.ndarray]:
    path = (
        DATA_DIR / "local" / "continuation_centroid_holdout" / model
        / f"centroids_{half}.npz"
    )
    npz = np.load(path)
    return {c: npz[c].astype(np.float32) for c in npz.files}


def _load_manifest(model: str) -> dict:
    path = (
        DATA_DIR / "local" / "continuation_centroid_holdout" / model
        / "manifest.json"
    )
    if not path.exists():
        raise SystemExit(f"no continuation hold-out manifest at {path}")
    return json.loads(path.read_text())


# ----- LDA pipeline (identical shape to 42_) -------------------------

def _fit_pca_lda(
    X: np.ndarray, y: list[str], *, n_pca: int | None = None,
) -> tuple[PCA, LinearDiscriminantAnalysis]:
    classes = sorted(set(y))
    n_classes = len(classes)
    n_rows = X.shape[0]
    if n_pca is None:
        n_pca = min(80, n_rows - n_classes, X.shape[1])
        n_pca = max(n_pca, n_classes)
    print(f"  PCA n_components={n_pca} (n_rows={n_rows}, n_classes={n_classes})")
    pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=0)
    X_p = pca.fit_transform(X)
    n_lda = max(1, n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_lda)
    lda.fit(X_p, y)
    return pca, lda


def _transform(
    X: np.ndarray, *, pca: PCA, lda: LinearDiscriminantAnalysis,
) -> np.ndarray:
    if X.ndim == 1:
        X = X[None, :]
    return lda.transform(pca.transform(X))


def _quadrant_color(cell: str) -> str:
    return QUADRANT_COLORS.get(cell, "#666666")


def _pad3(a: np.ndarray) -> np.ndarray:
    if a.shape[1] >= 3:
        return a[:, :3]
    out = np.zeros((a.shape[0], 3), dtype=a.dtype)
    out[:, : a.shape[1]] = a
    return out


def _plot_arm_3d_lda(
    arm: str,
    rows: list[dict],
    centroid_lda: dict[str, np.ndarray],
    *,
    out_path: Path,
    lda: LinearDiscriminantAnalysis,
    pca: PCA,
    title_prefix: str,
) -> None:
    fig = go.Figure()

    cell_names = list(centroid_lda.keys())
    pts = np.stack([centroid_lda[c] for c in cell_names])
    sizes = [16 if c == "MR" else 9 for c in cell_names]
    colors = [_quadrant_color(c) for c in cell_names]
    pts3 = _pad3(pts)
    fig.add_trace(go.Scatter3d(
        x=pts3[:, 0], y=pts3[:, 1], z=pts3[:, 2],
        mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="#000")),
        text=cell_names, textposition="top center",
        textfont=dict(size=11, color="#000"),
        name="cell centroids",
        hoverinfo="text",
    ))

    for r in rows:
        traj_lda = _pad3(_transform(r["trajectory"], pca=pca, lda=lda))
        n = traj_lda.shape[0]
        stride = r.get("stride", 1)
        token_idxs = [i * stride for i in range(n)]
        outcome = _attractor._classify_outcome(r["text"], r["n_tok"])
        opacity = 0.35 if outcome == "repetition_trap" else 0.85
        fig.add_trace(go.Scatter3d(
            x=traj_lda[:, 0], y=traj_lda[:, 1], z=traj_lda[:, 2],
            mode="lines+markers",
            line=dict(
                color=token_idxs, colorscale="Viridis",
                width=4, cmin=0, cmax=max(token_idxs) if token_idxs else 1,
            ),
            marker=dict(
                size=3, color=token_idxs, colorscale="Viridis",
                showscale=(r is rows[0]),
                colorbar=dict(title="token idx") if r is rows[0] else None,
            ),
            opacity=opacity,
            name=f"{r['prompt_id']} ({outcome[:4]})",
            hovertext=[
                f"{r['prompt_id']} s={r['seed']} tok={t} ({outcome})"
                for t in token_idxs
            ],
            hoverinfo="text",
            showlegend=False,
        ))

    evr = lda.explained_variance_ratio_  # type: ignore[attr-defined]
    axis_labels = [
        f"LD{i+1} ({evr[i]*100:.1f}%)" if i < len(evr) else f"LD{i+1} (0%)"
        for i in range(3)
    ]
    fig.update_layout(
        title=(
            f"{title_prefix} {arm}: trajectories in LDA (continuation) basis<br>"
            f"<sup>{len(rows)} trajectories; centroids fit on h_last; "
            f"trajectory color = token index</sup>"
        ),
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=720,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"    wrote {out_path}")


def _plot_centroid_comparison(
    model: str,
    centroid_lda: dict[str, np.ndarray],
    *,
    out_path: Path,
    lda_evr: np.ndarray,
) -> None:
    """2D top-2 LDA centroid scatter (sibling to 42_'s side-by-side; the
    h_first vs continuation comparison lives in the §1.6 audit table)."""
    cells = list(centroid_lda.keys())
    pts = np.stack([centroid_lda[c] for c in cells])
    if pts.shape[1] < 2:
        pad = np.zeros((pts.shape[0], 2 - pts.shape[1]), dtype=pts.dtype)
        pts = np.concatenate([pts, pad], axis=1)
    fig = go.Figure()
    for i, c in enumerate(cells):
        size = 18 if c == "MR" else 11
        fig.add_trace(go.Scatter(
            x=[pts[i, 0]], y=[pts[i, 1]],
            mode="markers+text",
            marker=dict(
                size=size, color=_quadrant_color(c),
                line=dict(width=1, color="#000"),
            ),
            text=[c], textposition="top center",
            textfont=dict(size=11),
            showlegend=False, hoverinfo="text",
        ))
    fig.update_layout(
        title=(f"{model}: centroid layout in LDA (continuation) basis "
               f"— LD1 {lda_evr[0]:.1%} / LD2 {lda_evr[1]:.1%}"),
        height=500, width=700,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(zeroline=True, zerolinecolor="#aaaaaa")
    fig.update_yaxes(zeroline=True, zerolinecolor="#aaaaaa")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  wrote {out_path}")


def _basin_separation_ratio(
    centroid_lookup: dict[str, np.ndarray],
) -> tuple[float, float, float, str]:
    if "MR" not in centroid_lookup:
        return float("nan"), float("nan"), float("nan"), ""
    cells = [c for c in centroid_lookup if c != "MR"]
    mr = centroid_lookup["MR"]
    can = np.stack([centroid_lookup[c] for c in cells])
    mr_dists = np.linalg.norm(can - mr[None, :], axis=1)
    nearest_idx = int(mr_dists.argmin())
    pairwise = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            pairwise.append(float(np.linalg.norm(can[i] - can[j])))
    can_mean = float(np.mean(pairwise)) if pairwise else float("nan")
    ratio = float(mr_dists.min()) / can_mean if can_mean > 0 else float("nan")
    return float(mr_dists.min()), can_mean, ratio, cells[nearest_idx]


def run_for_model(short: str, *, half: str = "A") -> dict:
    print(f"\n=== {short} (continuation basis) ===")
    manifest = _load_manifest(short)
    layer_idxs = list(manifest["layer_idxs"])
    hidden_dim = int(manifest["hidden_dim_per_layer"])
    n_layers = len(layer_idxs)
    print(f"  layer set: {n_layers} layers × {hidden_dim} dim")

    centroids_A = _load_holdout_centroids(short, half=half)
    print(f"  loaded {len(centroids_A)} centroids: {sorted(centroids_A)}")

    X, y = _load_training_rows(short, manifest)
    print(f"  X shape {X.shape}; n_classes {len(set(y))}")

    print("  fitting PCA → LDA...")
    pca, lda = _fit_pca_lda(X, y)
    lda_evr = np.asarray(lda.explained_variance_ratio_)
    print(f"  LDA EVR (first 5): {[f'{v:.3f}' for v in lda_evr[:5]]}")

    centroid_lda: dict[str, np.ndarray] = {
        c: _transform(v[None, :], pca=pca, lda=lda)[0]
        for c, v in centroids_A.items()
    }
    lda_top3 = {c: v[:3] for c, v in centroid_lda.items()}
    lda_nearest, lda_can_mean, lda_ratio, lda_nearest_name = (
        _basin_separation_ratio(lda_top3)
    )
    print(f"  LDA-top3 separation: MR→nearest={lda_nearest:.2f} "
          f"(vs {lda_nearest_name}); canonical_mean={lda_can_mean:.2f}; "
          f"ratio={lda_ratio:.3f}")

    # Trajectory rendering per arm.
    arm_dirs = sorted((DATA_DIR / "local").glob(f"{short}_attractor_*"))
    arms = [d.name.removeprefix(f"{short}_attractor_") for d in arm_dirs]
    arms = [a for a in arms if "steer" not in a]
    print(f"  arms: {arms}")

    fig_dir = FIGURES_DIR / "local" / f"{short}_attractor_lda_continuation"
    fig_dir.mkdir(parents=True, exist_ok=True)

    arm_to_rows: dict[str, list[dict]] = {}
    for arm in arms:
        print(f"  [{arm}] loading trajectories...")
        rows = _attractor._load_arm_trajectories(
            short, arm,
            target_layer_idxs=layer_idxs,
            target_hidden_dim=hidden_dim,
        )
        if not rows:
            continue
        arm_to_rows[arm] = rows
        _plot_arm_3d_lda(
            arm, rows, centroid_lda,
            out_path=fig_dir / f"{arm}_trajectories_3d.html",
            lda=lda, pca=pca,
            title_prefix=short,
        )

    _plot_centroid_comparison(
        short, centroid_lda,
        out_path=fig_dir / "comparison_centroids_2d.html",
        lda_evr=lda_evr,
    )

    summary = {
        "model": short,
        "basis": "continuation",
        "centroid_half": half,
        "n_train_rows": int(X.shape[0]),
        "n_classes": len(set(y)),
        "n_pca_components": int(pca.n_components_ or 0),
        "n_lda_components": int(lda.n_components or 0),
        "lda_explained_variance_ratio": [float(v) for v in lda_evr],
        "separation_lda_top3": {
            "mr_to_nearest_dist": lda_nearest,
            "nearest_cell": lda_nearest_name,
            "canonical_mean_pairwise_dist": lda_can_mean,
            "ratio": lda_ratio,
        },
        "arms_rendered": sorted(arm_to_rows.keys()),
    }
    out_summary = (
        DATA_DIR / "local" / "basin_stats"
        / f"{short}_lda_separation_continuation.json"
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {out_summary}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="ministral")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--centroid-half", default="A", choices=["A", "B"])
    args = ap.parse_args()

    if args.all_models:
        models = ["gemma", "qwen", "ministral"]
    else:
        if args.model not in MODEL_REGISTRY:
            raise SystemExit(f"unknown model {args.model!r}")
        models = [args.model]

    for model in models:
        try:
            run_for_model(model, half=args.centroid_half)
        except SystemExit as e:
            print(f"[{model}] skipped: {e}")


if __name__ == "__main__":
    main()
