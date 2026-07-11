"""h_first row-level PCA on talkie_1930, with archaic-English mirror
prompts as the canonical-cell baseline instead of modern English.

Parallels ``23i_talkie_h_pca_from_trajectories.py`` exactly, except
it pulls canonical-cell trajectories from the
``archaic_mirror_continue`` arm (1920s-1930s register prompts) instead
of ``mirror_continue`` (modern English). MR-cell trajectories come
from the same ``lb_continue`` and ``pr_continue`` arms as 23i.

Methodological test for the 2026-05-11 talkie work: if the modern-
English mirror prompts were artificially compressing talkie's
canonical baseline cluster (a register-mismatch artifact), the
archaic version should open the cluster up. If the canonical cluster
stays tight on archaic prompts too, MR's geometric distance is real.

``--rendering-match`` controls whether the MR/canonical comparison is
rendering-confounded:

  none     (default) — MR cells from ``pr_continue`` (assistant-prefill),
           canonical cells from ``archaic_mirror_continue`` (user-
           message). Rendering mode is perfectly confounded with the
           MR/canonical split — see ``docs/pitfalls.md`` §3. This is the
           original (confounded) figure.
  message  — MR cells from ``pr_message_continue``: the same
           PRE_1930_PROMPTS set rendered as a user message, matching the
           canonical arm's rendering. Both groups are now user-message,
           so any residual MR↔canonical gap is genuine content/register,
           not rendering. Outputs carry a ``_msgmatch`` suffix so both
           modes coexist in the same figures dir.

(An assistant-prefill match in the other direction is not viable:
canonical-affect prompts prefilled as a finished assistant turn collapse
to immediate EOS — self-sustaining continuation under prefill is the
basin property itself. See the 2026-05-16 archaic_prefill_continue
collapse.)

``--mr-style`` (only meaningful with ``--rendering-match message``)
selects the MR prompt typography:

  litany      (default) — MR from ``pr_message_continue``: the original
              PRE_1930_PROMPTS (lowercase, period-fragmented litany).
  normalized  — MR from ``pr_normalized_message_continue``: the same
              prompts with capitalization + fragment punctuation matched
              to the archaic-canonical baseline (repetition / anaphora
              preserved verbatim). Removes the residual typographic
              confound, so the gap is register/content only. Adds a
              ``_norm`` suffix to the outputs.

The ``--miscellany`` flag adds the ``archaic_miscellany_continue``
arm (30 topic-diverse archaic-English prompts deliberately NOT
affect-anchored — see ``llmoji_experiment/archaic_miscellany_prompts.py``).
These render as a single "MISC" cluster in gray, providing a denser
"ordinary archaic prose" cloud to test whether MR remains geometrically
isolated when the canonical baseline is broadened beyond the 9 affect
cells. Requires the matching emit data to exist under
``data/local/<model>_attractor_archaic_miscellany_continue/``.

Outputs (suffix ``_misc`` added when ``--miscellany`` is on so both
modes can live side-by-side in the same figures dir):
  figures/local/<model>_h_first_pca_archaic/3d_with_mr[_misc].html
  figures/local/<model>_h_first_pca_archaic/cell_centroids_2d[_misc].png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_experiment.config import DATA_DIR, MODEL_REGISTRY  # noqa: E402
from llmoji_experiment.hidden_state_io import load_hidden_states  # noqa: E402
from llmoji_experiment.quadrants import QUADRANT_COLORS, QUADRANT_ORDER_SPLIT  # noqa: E402


_ARC_RE = re.compile(r"^arc_([a-z]+)_\d+$", re.IGNORECASE)

MISC_CELL = "MISC"
MISC_COLOR = "#666666"


def _prompt_id_to_cell(pid: str) -> str | None:
    """Cell mapping including archaic prompts (``arc_<cell>_NN``).

    ``arc_misc_NN`` maps to the sentinel ``MISC`` (topic-diverse,
    non-affect-anchored archaic prose; deliberately not in
    ``QUADRANT_ORDER_SPLIT``).
    """
    prefix = pid[:2].lower()
    if prefix in ("lb", "pr"):
        return "MR"
    if prefix in ("dm", "cs", "sy", "ns"):
        return None
    m = _ARC_RE.match(pid)
    if m:
        code = m.group(1).upper()
        if code == "HPD":
            return "HP-D"
        if code == "HPS":
            return "HP-S"
        if code == "HND":
            return "HN-D"
        if code == "HNS":
            return "HN-S"
        if code in ("LP", "NP", "LN", "NB", "HB"):
            return code
        if code == "MISC":
            return MISC_CELL
    return None


def _load_arm(short: str, arm: str):
    jsonl = DATA_DIR / "local" / f"{short}_attractor_{arm}" / "emotional_raw.jsonl"
    if not jsonl.exists():
        return [], []
    sidecar_dir = DATA_DIR / "local" / "hidden" / f"{short}_attractor_{arm}"
    rows = []
    layer_idxs = None
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
            stack = np.stack(
                [cap.layers[L].h_first for L in layer_idxs], axis=0
            )
            rows.append({
                "prompt_id": r["prompt_id"],
                "cell": cell,
                "h_first": stack,
            })
    return rows, layer_idxs or []


def _plot_3d(coords, cells, out_path, title, pca_var):
    import plotly.graph_objects as go
    fig = go.Figure()
    cells_to_plot = list(QUADRANT_ORDER_SPLIT)
    if MISC_CELL in set(cells):
        cells_to_plot.append(MISC_CELL)
    for cell in cells_to_plot:
        mask = cells == cell
        n = int(mask.sum())
        if n == 0:
            continue
        if cell == MISC_CELL:
            color = MISC_COLOR
            size = 5
        else:
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


def _plot_centroids_2d(centroids, out_path, title):
    import matplotlib.pyplot as plt
    cells = [c for c in QUADRANT_ORDER_SPLIT if c in centroids]
    if MISC_CELL in centroids:
        cells.append(MISC_CELL)
    if len(cells) < 2:
        return
    C = np.stack([centroids[c] for c in cells])
    Cc = C - C.mean(axis=0)
    p = PCA(n_components=2).fit(Cc)
    pcs = p.transform(Cc)
    fig, ax = plt.subplots(figsize=(7, 6))
    for cell, xy in zip(cells, pcs):
        if cell == MISC_CELL:
            color = MISC_COLOR
            size = 130
        else:
            color = QUADRANT_COLORS.get(cell, "#888888")
            size = 180 if cell == "MR" else 100
        ax.scatter(xy[0], xy[1], s=size,
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
    ap.add_argument(
        "--rendering-match",
        choices=("none", "message"),
        default="none",
        dest="rendering_match",
        help="'none' (default): MR from pr_continue (assistant-prefill) "
             "vs canonical from archaic_mirror_continue (user-message) — "
             "the rendering-confounded original figure. 'message': MR "
             "from pr_message_continue (user-message), rendering-matched "
             "to the canonical arm, removing the pitfalls.md §3 confound.",
    )
    ap.add_argument(
        "--mr-style",
        choices=("litany", "normalized"),
        default="litany",
        dest="mr_style",
        help="Only with --rendering-match message. 'litany' (default): "
             "MR from pr_message_continue (original lowercase litany). "
             "'normalized': MR from pr_normalized_message_continue "
             "(case + fragment punctuation matched to the archaic "
             "baseline) — removes the residual typographic confound.",
    )
    ap.add_argument(
        "--miscellany",
        action="store_true",
        help="Include the archaic_miscellany_continue arm (30 topic-diverse "
             "non-affect-anchored archaic-English prompts) as a MISC cluster "
             "alongside the canonical 9-cell baseline.",
    )
    args = ap.parse_args()
    short = args.model
    if short not in MODEL_REGISTRY:
        raise SystemExit(f"unknown model {short!r}")

    if args.rendering_match == "message":
        mr_arm = (
            "pr_normalized_message_continue" if args.mr_style == "normalized"
            else "pr_message_continue"
        )
    else:
        if args.mr_style == "normalized":
            raise SystemExit(
                "--mr-style normalized requires --rendering-match message "
                "(the prefill MR arm pr_continue has no normalized variant)."
            )
        mr_arm = "pr_continue"
    arms = ["archaic_mirror_continue", mr_arm]
    if args.miscellany:
        arms.append("archaic_miscellany_continue")

    all_rows = []
    for arm in arms:
        rows, _ = _load_arm(short, arm)
        print(f"  {arm}: {len(rows)} rows")
        all_rows.extend(rows)
    if not all_rows:
        raise SystemExit(f"no attractor data for {short}")
    if args.miscellany and not any(r["cell"] == MISC_CELL for r in all_rows):
        print(f"  WARNING: --miscellany set but no MISC rows loaded "
              f"(missing data/local/{short}_attractor_archaic_miscellany_continue/?)")

    X = np.stack([r["h_first"].reshape(-1) for r in all_rows], axis=0).astype(np.float32)
    cells = np.array([r["cell"] for r in all_rows])
    n_total = X.shape[0]
    print(f"  total rows: {n_total} | flat dim: {X.shape[1]}")

    Xc = X - X.mean(axis=0)
    p = PCA(n_components=3).fit(Xc)
    coords = p.transform(Xc)
    print(f"  PCA variance: "
          f"{[f'{v:.1%}' for v in p.explained_variance_ratio_]}")
    print(f"  total explained (top 3): "
          f"{sum(p.explained_variance_ratio_):.1%}")

    centroids_flat: dict[str, np.ndarray] = {}
    cell_iter = list(QUADRANT_ORDER_SPLIT)
    if MISC_CELL in set(cells):
        cell_iter.append(MISC_CELL)
    print()
    print("  per-cell row counts + mean PC coords:")
    for cell in cell_iter:
        mask = cells == cell
        n = int(mask.sum())
        if n == 0:
            continue
        centroids_flat[cell] = X[mask].mean(axis=0)
        pc_mean = coords[mask].mean(axis=0)
        print(f"    {cell:>4s}: n={n:4d}  "
              f"PC1={pc_mean[0]:+7.2f}  "
              f"PC2={pc_mean[1]:+7.2f}  "
              f"PC3={pc_mean[2]:+7.2f}")

    if MISC_CELL in centroids_flat:
        mask = cells == MISC_CELL
        misc_coords = coords[mask]
        pc_std = misc_coords.std(axis=0)
        print()
        print("  MISC cloud spread (std along PC axes, fit-PC space):")
        print(f"    PC1 std={pc_std[0]:+7.2f}  "
              f"PC2 std={pc_std[1]:+7.2f}  "
              f"PC3 std={pc_std[2]:+7.2f}")

    if "MR" in centroids_flat:
        print()
        print("  MR centroid vs other cells (cosine, flat):")
        mr = centroids_flat["MR"]
        mr_norm = float(np.linalg.norm(mr))
        for cell in cell_iter:
            if cell == "MR" or cell not in centroids_flat:
                continue
            o = centroids_flat[cell]
            o_norm = float(np.linalg.norm(o))
            cos = float(np.dot(mr, o) / (mr_norm * o_norm))
            print(f"    MR <-> {cell:>4s}: cos={cos:+.4f}")

    suffix = ""
    if args.rendering_match == "message":
        suffix += "_msgmatch"
    if args.mr_style == "normalized":
        suffix += "_norm"
    if args.miscellany:
        suffix += "_misc"
    fig_dir = MODEL_REGISTRY[short].figures_dir.parent / f"{short}_h_first_pca_archaic"
    fig_dir.mkdir(parents=True, exist_ok=True)
    match_tail = (
        ", rendering-matched (user-message)"
        if args.rendering_match == "message" else ""
    )
    if args.mr_style == "normalized":
        match_tail += ", MR typography-normalized"
    title_tail = match_tail + (" (+ miscellany)" if args.miscellany else "")
    _plot_3d(
        coords, cells,
        out_path=fig_dir / f"3d_with_mr{suffix}.html",
        title=(f"{short}: h_first row-level PCA, archaic-English baseline"
               f"{title_tail}<br>"
               f"<sup>n={n_total} rows; PC1 {p.explained_variance_ratio_[0]:.1%} | "
               f"PC2 {p.explained_variance_ratio_[1]:.1%} | "
               f"PC3 {p.explained_variance_ratio_[2]:.1%}</sup>"),
        pca_var=list(p.explained_variance_ratio_),
    )
    _plot_centroids_2d(
        centroids_flat,
        out_path=fig_dir / f"cell_centroids_2d{suffix}.png",
        title=(f"{short}: h_first cell centroids, archaic-English baseline"
               f"{title_tail}"),
    )


if __name__ == "__main__":
    main()
