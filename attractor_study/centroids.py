"""Continuation-basis centroid hold-out (PROJECT_2 §1.6).

Replaces `40_centroid_holdout.py`'s h_first (kaomoji-emission state)
basis with **h_last from continuation trajectories**. Motivation:
session-1 distance audit found the kaomoji-emission centroids
(~150 unit cluster radius in raw space) sit ~775 units away from
the actual continuation-trajectory end-points. Basin-lock-as-argmin
is real but geometric "vicinity to MR centroid" is misleading —
the trajectories live in a different region of residual space than
the centroids that classify them.

The continuation basis fixes this by sourcing centroids from the
same kind of state the trajectories produce:

- canonical 9 cells: h_last of ``mirror_continue`` trajectories,
  labeled by the prompt's PAD-split cell (HP-D / HP-S / LP / NP /
  HN-D / HN-S / LN / NB / HB). The continuation runs from a user-
  message prompt for that cell; h_last captures where the model
  "settles" after 128-token free response. **Drift caveat:** some
  prompts drift to MR or to a different default cell. For the
  primary centroid build we use prompt-label as ground truth; for
  a follow-up control we'll filter to basin-locked-on-prompt-cell
  trajectories only.
- MR cell: h_last pooled across ``lb_continue`` + ``doom_continue``
  + ``conspiracy_continue`` + ``sycophancy_continue`` trajectories.
  Cross-content pooling matches the PROJECT_1 finding that all four
  arms land in the same basin geometry (arm-arm cos > 0.94).

Prompt-level A/B split per cell (matches `40_centroid_holdout.py`'s
contract: every row of a given prompt goes to one half). Seed for
the RNG: same as the h_first version (seed=0) for deterministic
comparison.

Output:
  data/local/continuation_centroid_holdout/{model}/
    centroids_A.npz   per-cell flat layer-stack h_last centroids (A-half)
    centroids_B.npz   B-half mirror
    manifest.json     splits, layer set, hidden_dim, sample strategy

Usage:
  .venv/bin/python scripts/local/40b_continuation_centroid_holdout.py \\
      --model gemma [--seed 0] [--all-models] [--strategy h_last]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from attractor_study.config import DATA_DIR, MODEL_REGISTRY  # noqa: E402
from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS  # noqa: E402
from llmoji_study.hidden_state_io import load_hidden_states  # noqa: E402


# Arms that source the MR cell (cross-content pooling).
MR_SOURCE_ARMS = (
    "lb_continue", "doom_continue", "conspiracy_continue", "sycophancy_continue",
)
# Arm that sources the canonical 9 cells.
CANONICAL_ARM = "mirror_continue"


def _pad_split_cell(prompt_id: str) -> str | None:
    """Map a mirror_continue prompt_id → PAD-split cell label.

    Uses ``EMOTIONAL_PROMPTS`` registry to look up the prompt's
    ``quadrant`` + ``pad_dominance``. Returns ``None`` if the prompt
    isn't found (drift/typo guard)."""
    for p in EMOTIONAL_PROMPTS:
        if p.id == prompt_id:
            q = p.quadrant
            if q == "HP":
                return "HP-D" if p.pad_dominance > 0 else "HP-S"
            if q == "HN":
                return "HN-D" if p.pad_dominance > 0 else "HN-S"
            return q
    return None


def _load_h_last_per_arm(
    model: str, arm: str,
) -> list[dict] | None:
    """Load h_last (per probe layer) for every trajectory in this arm.

    Returns a list of ``{prompt_id, seed, h_last_per_layer}`` dicts
    where ``h_last_per_layer`` is ``{layer_idx: np.ndarray (hidden_dim,)}``.
    Skips silent-refusal rows (n_tok ≤ 1).

    Returns ``None`` if the arm's JSONL doesn't exist (e.g. ministral
    pre-mirror_continue regen).
    """
    jsonl = DATA_DIR / "local" / f"{model}_attractor_{arm}" / "emotional_raw.jsonl"
    if not jsonl.exists():
        return None
    rows: list[dict] = []
    skipped = 0
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r or not r.get("row_uuid"):
                continue
            if r.get("trajectory_n_tokens", 0) < 2:
                skipped += 1
                continue
            sidecar = (
                DATA_DIR / "local" / "hidden"
                / f"{model}_attractor_{arm}" / f"{r['row_uuid']}.npz"
            )
            if not sidecar.exists():
                print(f"    WARN: missing sidecar for {r['prompt_id']} s={r['seed']}")
                continue
            cap = load_hidden_states(sidecar, full_trace=True)
            h_last = {
                idx: lc.h_last.astype(np.float32)
                for idx, lc in cap.layers.items()
            }
            rows.append({
                "prompt_id": r["prompt_id"],
                "seed": r["seed"],
                "h_last": h_last,
            })
    if skipped:
        print(f"    skipped {skipped} silent-refusal rows in {arm}")
    return rows


def _intersect_layers(rows: list[dict]) -> list[int]:
    """Intersect probe-layer sets across all loaded rows."""
    if not rows:
        return []
    common = set(rows[0]["h_last"].keys())
    for r in rows[1:]:
        common &= set(r["h_last"].keys())
    return sorted(common)


def _flatten(row: dict, layer_idxs: list[int]) -> np.ndarray:
    """Stack per-layer h_last into a single flat vector in layer order."""
    return np.concatenate([row["h_last"][L] for L in layer_idxs])


def _split_prompts(
    prompts: list[str], rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    pool = list(prompts)
    rng.shuffle(pool)
    half = (len(pool) + 1) // 2
    return pool[:half], pool[half:]


def build_holdout(model: str, *, seed: int = 0) -> dict:
    print(f"\n=== {model} ===")

    # ---- load canonical mirror_continue ---------------------------------
    print(f"  loading {CANONICAL_ARM} h_last...")
    mirror_rows = _load_h_last_per_arm(model, CANONICAL_ARM)
    if mirror_rows is None:
        raise SystemExit(
            f"no mirror_continue arm for {model}; "
            f"run 02_emit_attractor.py with arm=mirror_continue first"
        )
    print(f"  mirror_continue: {len(mirror_rows)} rows")

    # Label each mirror row with its PAD-split cell.
    for r in mirror_rows:
        r["cell"] = _pad_split_cell(r["prompt_id"])
    unlabeled = [r for r in mirror_rows if r["cell"] is None]
    if unlabeled:
        print(f"  WARN: {len(unlabeled)} rows could not be labeled by PAD split "
              f"(missing from EMOTIONAL_PROMPTS): "
              f"{[r['prompt_id'] for r in unlabeled[:5]]}")
        mirror_rows = [r for r in mirror_rows if r["cell"] is not None]

    # ---- load MR-source in-basin arms ----------------------------------
    mr_rows: list[dict] = []
    for arm in MR_SOURCE_ARMS:
        print(f"  loading {arm} h_last...")
        rows = _load_h_last_per_arm(model, arm)
        if rows is None:
            print(f"    not present; skipping")
            continue
        for r in rows:
            r["cell"] = "MR"
            r["arm"] = arm
        mr_rows.extend(rows)
        print(f"    {arm}: {len(rows)} rows")
    if not mr_rows:
        raise SystemExit(f"no MR-source data for {model}")
    print(f"  MR (pooled): {len(mr_rows)} rows")

    # ---- layer intersection across all sources -------------------------
    all_rows = mirror_rows + mr_rows
    layer_idxs = _intersect_layers(all_rows)
    if not layer_idxs:
        raise SystemExit("no common probe layers across arms")
    hidden_dim = next(iter(all_rows[0]["h_last"].values())).shape[0]
    print(f"  layer intersection: {len(layer_idxs)} layers × {hidden_dim} dim")

    # ---- per-cell prompt-level A/B split + centroids -------------------
    rng = np.random.default_rng(seed)
    cell_to_rows: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        cell_to_rows[r["cell"]].append(r)

    centroids_A: dict[str, np.ndarray] = {}
    centroids_B: dict[str, np.ndarray] = {}
    manifest_splits: dict[str, dict] = {}

    print(f"\n  per-cell A/B split + centroid construction:")
    for cell in sorted(cell_to_rows.keys()):
        rows = cell_to_rows[cell]
        # For MR, multiple arms supply prompts under their own prompt-id
        # namespaces (lb01 / dm01 / cs01 / sy01). Treat the (arm,prompt_id)
        # tuple as the split unit to avoid leaking the same prompt across
        # halves when prompt-id collides across arms (it doesn't actually
        # — lb / dm / cs / sy ids are disjoint — but tuple-keying is robust).
        keys = sorted(set(
            (r.get("arm", CANONICAL_ARM), r["prompt_id"]) for r in rows
        ))
        if len(keys) < 2:
            print(f"    WARN: cell {cell} has {len(keys)} prompt(s); skipping")
            continue
        a_keys, b_keys = _split_prompts([str(k) for k in keys], rng)
        a_keyset = set(a_keys)
        b_keyset = set(b_keys)

        def _key(r: dict) -> str:
            return str((r.get("arm", CANONICAL_ARM), r["prompt_id"]))

        a_rows = [r for r in rows if _key(r) in a_keyset]
        b_rows = [r for r in rows if _key(r) in b_keyset]
        if not a_rows or not b_rows:
            print(f"    WARN: cell {cell} has empty half (a={len(a_rows)}, "
                  f"b={len(b_rows)}); skipping")
            continue
        a_vecs = np.stack([_flatten(r, layer_idxs) for r in a_rows])
        b_vecs = np.stack([_flatten(r, layer_idxs) for r in b_rows])
        centroids_A[cell] = a_vecs.mean(axis=0).astype(np.float32)
        centroids_B[cell] = b_vecs.mean(axis=0).astype(np.float32)
        # Cosine A↔B sanity
        a, b = centroids_A[cell], centroids_B[cell]
        cos_ab = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        manifest_splits[cell] = {
            "A_keys": sorted(a_keys),
            "B_keys": sorted(b_keys),
            "n_rows_A": len(a_rows),
            "n_rows_B": len(b_rows),
            "cos_AB": cos_ab,
        }
        print(f"    {cell:>5s}  n_A={len(a_rows):>2d}  n_B={len(b_rows):>2d}  "
              f"cos(A,B)={cos_ab:>+.4f}  ||A||={np.linalg.norm(a):.1f}  "
              f"||B||={np.linalg.norm(b):.1f}")

    # ---- write outputs --------------------------------------------------
    out_dir = DATA_DIR / "local" / "continuation_centroid_holdout" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_dir / "centroids_A.npz"), **centroids_A)  # type: ignore[arg-type]
    np.savez(str(out_dir / "centroids_B.npz"), **centroids_B)  # type: ignore[arg-type]
    manifest = {
        "model": model,
        "seed": seed,
        "sample_strategy": "h_last",
        "canonical_arm": CANONICAL_ARM,
        "mr_source_arms": list(MR_SOURCE_ARMS),
        "layer_idxs": [int(L) for L in layer_idxs],
        "hidden_dim_per_layer": int(hidden_dim),
        "n_layers": len(layer_idxs),
        "splits": manifest_splits,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(f"\n  wrote {out_dir}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.all_models:
        models = ["gemma", "qwen", "ministral"]
    else:
        if args.model not in MODEL_REGISTRY:
            raise SystemExit(f"unknown model {args.model!r}")
        models = [args.model]

    for model in models:
        try:
            build_holdout(model, seed=args.seed)
        except SystemExit as e:
            print(f"[{model}] skipped: {e}")
            continue


if __name__ == "__main__":
    main()
