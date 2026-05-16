"""v2 MR_score: empirical-baseline-anchored projection on the MR direction.

PROJECT_2 phase 3.1 v2 instrument. Where the v1 instrument
(``46_mr_score_summary.py``) was anchored on fraction-of-tokens-closest-
to-MR-out-of-10-cells, this v2 uses ONE direction in residual space:
the empirically-defined MR direction.

Pipeline:

1. Load per-chunk mean residual-stream vectors from
   ``30c_chunk_mean_state.py`` outputs (one ``.npz`` per source).
2. Define ``baseline_state`` = mean of per-chunk vectors over the
   baseline corpus (wiki + lit-fiction + chat).
3. Load the held-out MR centroid (``centroids_A.npz["MR"]``).
4. Define ``mr_dir`` = unit-normalize(``MR_centroid - baseline_state``).
   This is the empirical MR-vs-empirical-neutral direction. It uses
   ONLY the MR anchor and the baseline; no dependence on 9 other
   emotion-cell centroids.
5. Project each chunk vector onto ``mr_dir``: ``p_chunk = chunk_vec ·
   mr_dir``. Signed scalar.
6. Compute ``baseline_proj_mean`` (mean over baseline chunks) and
   ``LB_proj_mean`` (mean over LB calibration chunks).
7. ``MR_score(chunk) = (p_chunk - baseline_proj_mean) /
                       (LB_proj_mean - baseline_proj_mean)``
   — 0 at baseline mean, 1 at LB mean, signed.

Bootstrap CIs for each source by resampling chunks within source
with replacement.

Usage::

    .venv/bin/python scripts/local/46b_mr_score_v2.py \\
        --baseline-states data/local/scripture_mr/chunk_states_baseline_gemma.npz \\
        --calibration-states data/local/scripture_mr/chunk_states_calibration_gemma.npz \\
        --scripture-states data/local/scripture_mr/chunk_states_scripture_gemma.npz \\
        --centroid-dir data/local/centroid_holdout/gemma \\
        --output-md data/local/scripture_mr/mr_score_v2_gemma.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

DISPLAY_NAMES: dict[str, str] = {
    # scripture
    "bom": "Book of Mormon",
    "dc": "Doctrine & Covenants",
    "quran": "Quran",
    "kv": "Kalevala",
    "lotus": "Lotus Sutra",
    "pgp": "Pearl of Great Price",
    "nt": "New Testament",
    "poe": "Poetic Edda",
    "bund": "Bundahishn",
    "viraf": "Arda Viraf",
    "apoc": "Apocrypha",
    "ttc": "Tao Te Ching",
    "ot": "Old Testament",
    "kj": "Kojiki",
    "bop": "Book of Poetry (Shijing)",
    "fourbooks": "Four Books (Confucian)",
    # calibration
    "lb": "LB calibration (bliss)",
    "dm": "DM calibration (doom)",
    "cs": "CS calibration (conspiracy)",
    "sy": "SY calibration (sycophancy)",
    # baseline
    "wiki": "Wikipedia",
    "gutenberg": "Gutenberg lit fiction",
    "oasst": "OpenAssistant chat",
}


def _load_states_with_meta(npz_path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    npz = np.load(npz_path)
    states = {cid: npz[cid].astype(np.float64) for cid in npz.files}
    meta_path = npz_path.with_suffix(npz_path.suffix + ".meta.json")
    meta: dict[str, dict] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return states, meta


def _bootstrap_mean_ci(
    vals: list[float], n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(vals, dtype=float)
    n = arr.shape[0]
    if n < 2:
        return float(arr.mean()), float(arr.mean()), float(arr.mean())
    point = float(arr.mean())
    boots = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-states", required=True, type=Path)
    ap.add_argument("--calibration-states", required=True, type=Path)
    ap.add_argument("--scripture-states", type=Path, default=None)
    ap.add_argument("--centroid-dir", required=True, type=Path)
    ap.add_argument("--output-md", type=Path, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Held-out MR centroid (using A-half for consistency with phase 1.2).
    centroids = np.load(args.centroid_dir / "centroids_A.npz")
    if "MR" not in centroids.files:
        raise SystemExit("no MR centroid in held-out file")
    mr_centroid = centroids["MR"].astype(np.float64)
    print(f"MR centroid loaded: shape {mr_centroid.shape}")

    # Load chunk-mean states for the three corpora.
    base_states, base_meta = _load_states_with_meta(args.baseline_states)
    cal_states, cal_meta = _load_states_with_meta(args.calibration_states)
    scrip_states, scrip_meta = ({}, {})
    if args.scripture_states is not None:
        scrip_states, scrip_meta = _load_states_with_meta(args.scripture_states)
    print(
        f"loaded states: baseline={len(base_states)}, "
        f"calibration={len(cal_states)}, scripture={len(scrip_states)}"
    )

    # Empirical baseline state: mean of per-chunk vectors over baseline.
    if not base_states:
        raise SystemExit("baseline states are empty")
    base_arr = np.stack(list(base_states.values()), axis=0)
    baseline_state = base_arr.mean(axis=0)
    print(f"empirical baseline_state: norm={np.linalg.norm(baseline_state):.2f}")
    print(f"MR_centroid: norm={np.linalg.norm(mr_centroid):.2f}")

    # MR direction: from baseline_state toward MR_centroid, unit-normed.
    raw_dir = mr_centroid - baseline_state
    raw_norm = float(np.linalg.norm(raw_dir))
    if raw_norm < 1e-6:
        raise SystemExit("MR centroid and baseline state are too close")
    mr_dir = raw_dir / raw_norm
    print(
        f"||MR - baseline|| = {raw_norm:.2f}  "
        f"(unit-direction defined)"
    )

    # Cosine between baseline_state and MR_centroid for sanity.
    cos_base_mr = float(
        baseline_state @ mr_centroid
        / (np.linalg.norm(baseline_state) * np.linalg.norm(mr_centroid))
    )
    print(f"cos(baseline_state, MR_centroid) = {cos_base_mr:+.4f}")

    # Project every chunk onto mr_dir.
    def project(states_dict: dict[str, np.ndarray]) -> dict[str, float]:
        return {cid: float(state @ mr_dir) for cid, state in states_dict.items()}

    base_proj = project(base_states)
    cal_proj = project(cal_states)
    scrip_proj = project(scrip_states)

    # Group projections by source.
    by_src: dict[str, list[float]] = defaultdict(list)
    src_by_cid: dict[str, str] = {}
    cat_by_cid: dict[str, str] = {}
    for cid, p in base_proj.items():
        s = base_meta.get(cid, {}).get("source", "baseline")
        src_by_cid[cid] = s
        cat_by_cid[cid] = "baseline"
        by_src[s].append(p)
    for cid, p in cal_proj.items():
        s = cal_meta.get(cid, {}).get("source", "calibration")
        src_by_cid[cid] = s
        cat_by_cid[cid] = "calibration"
        by_src[s].append(p)
    for cid, p in scrip_proj.items():
        s = scrip_meta.get(cid, {}).get("source", "scripture")
        src_by_cid[cid] = s
        cat_by_cid[cid] = "scripture"
        by_src[s].append(p)

    # Anchors.
    pooled_base = []
    for cid, p in base_proj.items():
        pooled_base.append(p)
    base_mean, base_lo, base_hi = _bootstrap_mean_ci(
        pooled_base, args.n_bootstrap, rng,
    )
    lb_vals = [
        cal_proj[cid] for cid, m in cal_meta.items()
        if m.get("source") == "lb"
    ]
    lb_mean, lb_lo, lb_hi = _bootstrap_mean_ci(
        lb_vals, args.n_bootstrap, rng,
    )

    denom = lb_mean - base_mean
    if abs(denom) < 1e-9:
        raise SystemExit("baseline projection and LB projection too close")
    print()
    print(
        f"BASELINE projection: mean={base_mean:.2f} "
        f"[{base_lo:.2f}, {base_hi:.2f}] (n={len(pooled_base)} chunks)"
    )
    print(
        f"LB CALIBRATION projection: mean={lb_mean:.2f} "
        f"[{lb_lo:.2f}, {lb_hi:.2f}] (n={len(lb_vals)})"
    )
    print(f"MR_score scale: divide by {denom:.2f}")

    def to_mr_score(p: float) -> float:
        return (p - base_mean) / denom

    # Per-source table.
    table: list[dict] = []
    for src, vs in by_src.items():
        if not vs:
            continue
        # Find a representative category (all chunks in src share a category).
        category = "unknown"
        for cid, s in src_by_cid.items():
            if s == src:
                category = cat_by_cid[cid]
                break
        proj_mean, proj_lo, proj_hi = _bootstrap_mean_ci(
            vs, args.n_bootstrap, rng,
        )
        table.append({
            "src": src,
            "category": category,
            "n_chunks": len(vs),
            "projection_mean": proj_mean,
            "MR_score": to_mr_score(proj_mean),
            "MR_score_ci_lo": to_mr_score(proj_lo),
            "MR_score_ci_hi": to_mr_score(proj_hi),
        })
    table.sort(key=lambda r: -r["MR_score"])

    # Markdown report.
    lines: list[str] = []
    lines.append("# MR_score v2: empirical-baseline-anchored MR projection (gemma)\n")
    lines.append("**Metric definition**")
    lines.append("")
    lines.append(
        "1. For each chunk, compute mean residual-stream state across "
        "all content tokens (one ``(n_layers × hidden_dim,)`` vector "
        "per chunk).\n"
        "2. Empirical baseline state: mean of per-chunk vectors over the "
        "baseline corpus (wiki / gutenberg / oasst).\n"
        "3. MR direction: ``mr_dir = unit(MR_centroid - baseline_state)`` "
        "— uses ONLY MR anchor + empirical neutral. No 9 other cell "
        "centroids.\n"
        "4. Per-chunk projection: ``p = chunk_state · mr_dir``.\n"
        "5. ``MR_score = (p - baseline_proj_mean) / "
        "(LB_proj_mean - baseline_proj_mean)``."
    )
    lines.append("")
    lines.append(
        f"**Baseline anchor (MR_score=0):** projection mean = {base_mean:.2f} "
        f"[{base_lo:.2f}, {base_hi:.2f}], n={len(pooled_base)}"
    )
    lines.append(
        f"**LB anchor (MR_score=1):** projection mean = {lb_mean:.2f} "
        f"[{lb_lo:.2f}, {lb_hi:.2f}], n={len(lb_vals)}"
    )
    lines.append("")
    lines.append("| rank | category | source | n | projection | MR_score | 95% CI |")
    lines.append("|---:|---|---|---:|---:|---:|---|")
    for rank, row in enumerate(table, start=1):
        name = DISPLAY_NAMES.get(row["src"], row["src"])
        ci = f"[{row['MR_score_ci_lo']:+.3f}, {row['MR_score_ci_hi']:+.3f}]"
        lines.append(
            f"| {rank} | {row['category']} | **{name}** ({row['src']}) | "
            f"{row['n_chunks']} | "
            f"{row['projection_mean']:.2f} | "
            f"{row['MR_score']:+.3f} | {ci} |"
        )
    lines.append("")
    # Category summary.
    by_cat: dict[str, list[float]] = defaultdict(list)
    for row in table:
        by_cat[row["category"]].append(row["MR_score"])
    lines.append("## Category summary")
    lines.append("")
    lines.append("| category | n sources | MR_score range | mean |")
    lines.append("|---|---:|---|---:|")
    for cat in ["baseline", "calibration", "scripture"]:
        if cat not in by_cat:
            continue
        vs = by_cat[cat]
        lines.append(
            f"| {cat} | {len(vs)} | "
            f"[{min(vs):+.3f}, {max(vs):+.3f}] | {mean(vs):+.3f} |"
        )
    lines.append("")

    md = "\n".join(lines)
    print(md)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md, encoding="utf-8")
        print(f"\nwrote {args.output_md}")


if __name__ == "__main__":
    main()
