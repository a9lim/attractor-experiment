"""PC-resolution gradient check: does basin activation strength
correlate with how high in the centroid-PCA variance hierarchy the
MR signal appears?

Hypothesis (from a9, 2026-05-12, in conversation about the two-qwen
pilot's 3D-PCA misleadingness): surface tokens and high-variance PCs
are the same kind of thing — "the most expressed parts of the stack"
— measured at different layers. Full basin lock should dominate both;
partial activation should only push the less-expressed dimensions.
Concretely: prefilled-MR runs should be MR-modal even when only PC1-3
are visible (the misleading-3D view); two-bot dialogue should need
PC4+ to reveal MR-modal.

Tests the prediction by:
1. Building the same centroid-PCA basis the two-qwen analysis uses.
2. Loading h_first per row from a panel of arms spanning the basin-
   activation spectrum: strong prefill (lb_continue, doom_continue,
   conspiracy_continue), suppressed-surface prefill
   (sycophancy_continue), two-bot dialogue (neutral, peer), and
   non-basin controls (mirror_continue, neutral_seed).
3. For each arm, computing:
   - full-space cosine MR-margin (mean cos to MR − max cos to other cells)
   - full-K (=9) Euclidean-MR-modal share
   - K=3 Euclidean-MR-modal share (the misleading 3D view)
   - min K at which MR-modal share ≥ 50%

Output is a single summary table — meant to be a one-shot diagnostic,
not part of the pre-50 chain. Saves results to
``data/local/qwen_two_loop/pc_resolution_gradient.tsv``.

Caveat to know going in: the "min K ≥50% MR" metric is only meaningful
as a basin-strength measure for arms that ARE in the basin at full K.
PC4 carries a strong MR-vs-LP direction, so adding it can pull
near-LP non-basin points to MR-closest too (see mirror_continue and
neutral_seed in the output — they hit K=2 / K=4 thresholds but their
full-K MR% is 29% / 10%, revealing they aren't really MR at the
full-resolution level). Read K-threshold only on rows where full-K
MR% ≥ 80%.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_study import trajectory as aa  # noqa: E402
from attractor_study import two_qwen as b  # noqa: E402
from attractor_study.config import DATA_DIR  # noqa: E402
from llmoji_study.hidden_state_io import load_hidden_states  # noqa: E402


def _load_arm_h_first(arm_name: str, common: list[int]) -> np.ndarray:
    """Load h_first per row from an attractor arm, layer-aligned to
    ``common``. Returns shape (n_rows, n_common * hidden_dim)."""
    hidden_dir = DATA_DIR / "local" / "hidden" / arm_name
    if not hidden_dir.exists():
        return np.empty((0, len(common)))
    rows: list[np.ndarray] = []
    hD: int | None = None
    for sd in sorted(hidden_dir.glob("*.npz")):
        cap = load_hidden_states(sd, full_trace=False)
        if hD is None:
            hD = int(cap.layers[next(iter(cap.layers))].h_first.shape[0])
        per_layer = np.full((len(common), hD), np.nan, dtype=np.float32)
        avail = set(cap.layers.keys())
        for k, idx in enumerate(common):
            if idx in avail:
                per_layer[k] = cap.layers[idx].h_first
        if not np.isfinite(per_layer).all():
            mean_layer = np.nanmean(per_layer, axis=0, keepdims=True)
            per_layer = np.where(np.isnan(per_layer), mean_layer, per_layer)
        rows.append(per_layer.reshape(-1))
    return np.stack(rows) if rows else np.empty((0, len(common) * (hD or 1)))


def _full_mr_margin(X: np.ndarray, C: np.ndarray, mr_idx: int) -> np.ndarray:
    """Full-space cosine margin: cos to MR − max cos to any other centroid."""
    Cn = np.linalg.norm(C, axis=1, keepdims=True)
    Xn = np.linalg.norm(X, axis=1, keepdims=True)
    sim = (X @ C.T) / (Xn * Cn.T + 1e-12)
    cos_mr = sim[:, mr_idx]
    other = np.delete(sim, mr_idx, axis=1)
    return cos_mr - other.max(axis=1)


def _mr_share_at_k(X_pca: np.ndarray, C_pca: np.ndarray, mr_idx: int, K: int) -> float:
    """Fraction of rows whose Euclidean-nearest centroid is MR in top-K PCs."""
    d = np.linalg.norm(X_pca[:, None, :K] - C_pca[None, :, :K], axis=2)
    closest = d.argmin(axis=1)
    return float((closest == mr_idx).mean())


def main() -> None:
    # Use the two-qwen pilot's probe-layer set so the PCA basis is
    # identical to what 24b uses — apples-to-apples comparison.
    any_sd = next((DATA_DIR / "local" / "hidden" / "qwen_two_loop_neutral").glob("*.npz"))
    cap = load_hidden_states(any_sd, full_trace=False)
    two_qwen_layers = sorted(cap.layers.keys())

    centroids, common = aa._build_centroids("qwen", attractor_layers=two_qwen_layers)
    cell_names = list(centroids.keys())
    C = np.stack([centroids[c] for c in cell_names])
    mr_idx = cell_names.index("MR")

    pca = PCA(n_components=10).fit(C)
    C9 = pca.transform(C)

    arms = [
        ("lb_continue",          "qwen_attractor_lb_continue"),
        ("doom_continue",        "qwen_attractor_doom_continue"),
        ("conspiracy_continue",  "qwen_attractor_conspiracy_continue"),
        ("sycophancy_continue",  "qwen_attractor_sycophancy_continue"),
        ("two_qwen_peer",        None),
        ("two_qwen_neutral",     None),
        ("mirror_continue",      "qwen_attractor_mirror_continue"),
        ("neutral_seed",         "qwen_attractor_neutral_seed"),
    ]

    rows: list[dict] = []
    for label, arm_dir in arms:
        if arm_dir is not None:
            X = _load_arm_h_first(arm_dir, common)
        else:
            cfg = label.replace("two_qwen_", "")
            df = b._load_transcript(cfg)
            X = b._load_h_first_stack(cfg, df, common)

        if X.size == 0:
            continue

        margins = _full_mr_margin(X, C, mr_idx)
        X9 = pca.transform(X)

        mr_at = {K: _mr_share_at_k(X9, C9, mr_idx, K) for K in [1, 2, 3, 4, 5, 6, 7, 9]}
        # Min K where MR-modal share ≥ 50%.
        min_k: int | None = None
        for K in [1, 2, 3, 4, 5, 6, 7, 9]:
            if mr_at[K] >= 0.5:
                min_k = K
                break

        rows.append({
            "arm":          label,
            "n":            int(len(X)),
            "mean_margin":  float(margins.mean()),
            "mr_k1":        mr_at[1],
            "mr_k2":        mr_at[2],
            "mr_k3":        mr_at[3],
            "mr_k4":        mr_at[4],
            "mr_k5":        mr_at[5],
            "mr_k9":        mr_at[9],
            "min_k_50":     min_k if min_k is not None else 0,
        })

    df = pd.DataFrame(rows)
    out_dir = DATA_DIR / "local" / "qwen_two_loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pc_resolution_gradient.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"wrote {out_path}")
    print()

    # Console-pretty.
    print(f'{"arm":>22}  {"n":>3}  {"margin":>8}  '
          f'{"K1":>5} {"K2":>5} {"K3":>5} {"K4":>5} {"K5":>5} {"K9":>5}  '
          f'{"minK≥50%":>9}')
    print("-" * 90)
    for r in rows:
        pcts = ["{:>4.0f}%".format(r[f"mr_k{k}"] * 100) for k in [1, 2, 3, 4, 5, 9]]
        mk = str(r["min_k_50"]) if r["min_k_50"] > 0 else "n/a"
        print(f'{r["arm"]:>22}  {r["n"]:>3}  {r["mean_margin"]:+8.4f}  '
              + " ".join(pcts) + f'  {mk:>9}')

    print()
    print("Reading:")
    print("- Restrict K-threshold reads to arms with K9 ≥ 80% MR (true-in-basin).")
    print("- Lower K-threshold = stronger basin activation (signal in higher-")
    print("  variance dimensions = more surface-legible).")
    print("- two_qwen_peer mirrors sycophancy_continue: same K=3-suppressed /")
    print("  K=4-emergent profile, comparable margin. Both 'saturated form,")
    print("  suppressed MR vocabulary, basin signal hides in PC4+.'")


if __name__ == "__main__":
    main()
