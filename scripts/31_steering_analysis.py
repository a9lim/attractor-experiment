"""Steered-attractor dose-response, measured in the continuation basis.

PROJECT_2 Session-3. `03b_attractor_steered_analysis.py` measures
basin-lock against 02b's h_first centroids — the wrong basis per §1.6.
This script re-measures every steered run in the **continuation
basis** (the §1.6-correct basin metric) and tabulates the dose-response
side by side for the two steering vectors:

- `mr_nb`      -- the original h_first MR−NB vector (§D-shown to steer
                  a direction near-orthogonal to the continuation MR
                  axis)
- `mr_cont_nb` -- the continuation-basis MR−NB vector registered by
                  `49_register_continuation_probes.py`

Auto-discovers `data/local/<model>_attractor_<arm>_steer<scalar>_<vectag>/`
dirs, projects trajectories onto the continuation A-half centroids,
and reports per (vector, arm, scalar):

- basin_lock t=end / t=mid : fraction of trajectories MR-closest
- word_tri                 : distinct word-trigram ratio (low =
                             repetition trap, e.g. MR mode-collapse)
- alpha_frac               : fraction of alphabetic characters (low =
                             off-manifold token salad, e.g. "!!!!!!")
- n_silent                 : rows with n_tok < 2 (silent refusal),
                             excluded from basin-lock

A coherent escape has lock_end well below the unsteered baseline
*and* word_tri + alpha_frac both staying high. Degeneration shows as
a word_tri or alpha_frac collapse.

Usage::

    .venv/bin/python scripts/local/50_steering_continuation_analysis.py --model gemma
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_experiment import trajectory as _attr  # noqa: E402

DATA_DIR = _attr.DATA_DIR

_DIR_RE = re.compile(r"^(?P<model>[^_]+)_attractor_(?P<arm>.+)_steer"
                     r"(?P<scalar>[+-][0-9.]+)_(?P<vectag>.+)$")


def _load_continuation_centroids(model: str, half: str = "A"):
    cdir = DATA_DIR / "local" / "continuation_centroid_holdout" / model
    manifest = json.loads((cdir / "manifest.json").read_text())
    layer_idxs = list(manifest["layer_idxs"])
    hd = int(manifest["hidden_dim_per_layer"])
    npz = np.load(cdir / f"centroids_{half}.npz")
    cents = {c: npz[c].astype(np.float32) for c in npz.files}
    return cents, layer_idxs, hd


def _closest_cell(point: np.ndarray, centroids: dict[str, np.ndarray]) -> str:
    cells = list(centroids.keys())
    cs = np.stack([centroids[c] for c in cells])
    return cells[int(np.argmin(np.linalg.norm(cs - point[None, :], axis=1)))]


def _word_trigram_ratio(text: str) -> float:
    """Distinct word-trigram ratio. 1.0 = no repetition, low = trap."""
    words = re.findall(r"\S+", text.lower())
    if len(words) < 3:
        return 1.0
    tris = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return len(set(tris)) / len(tris)


def _alpha_frac(text: str) -> float:
    """Fraction of alphabetic characters (ignoring whitespace).
    Low => off-manifold token salad like '!!!!!!!!set!!!'."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c.isalpha() for c in chars) / len(chars)


def run(model: str) -> None:
    cents, layer_idxs, hd = _load_continuation_centroids(model)
    print(f"[{model}] continuation centroids: {len(cents)} cells x "
          f"{len(layer_idxs)} layers")

    local = DATA_DIR / "local"
    discovered: list[tuple[str, str, float, str]] = []
    for d in sorted(local.iterdir()):
        if not d.is_dir() or "_steer" not in d.name:
            continue
        m = _DIR_RE.match(d.name)
        if not m or m.group("model") != model:
            continue
        discovered.append((d.name, m.group("arm"),
                           float(m.group("scalar")), m.group("vectag")))

    # rows keyed by (vectag, arm) -> {scalar: stats}
    table: dict[tuple[str, str], dict[float, dict]] = defaultdict(dict)
    for dirname, arm, scalar, vectag in discovered:
        jsonl = local / dirname / "emotional_raw.jsonl"
        if not jsonl.exists():
            print(f"  skip {dirname} (no emotional_raw.jsonl)")
            continue
        n_total = sum(1 for ln in jsonl.read_text().splitlines() if ln.strip())
        trs = _attr._load_arm_trajectories(
            model, arm + f"_steer{scalar:+.1f}_{vectag}",
            target_layer_idxs=layer_idxs, target_hidden_dim=hd,
        )
        end_flags, mid_flags, wtri, afrac = [], [], [], []
        for tr in trs:
            traj = tr["trajectory"]
            if traj.shape[0] < 2:
                continue
            mid = min(max(int(round(0.5 * (traj.shape[0] - 1))), 0),
                      traj.shape[0] - 1)
            end_flags.append(_closest_cell(traj[-1], cents) == "MR")
            mid_flags.append(_closest_cell(traj[mid], cents) == "MR")
            wtri.append(_word_trigram_ratio(tr.get("text", "")))
            afrac.append(_alpha_frac(tr.get("text", "")))
        n = len(end_flags)
        table[(vectag, arm)][scalar] = {
            "n": n,
            "n_silent": n_total - n,
            "lock_end": float(np.mean(end_flags)) if n else float("nan"),
            "lock_mid": float(np.mean(mid_flags)) if n else float("nan"),
            "word_tri": float(np.mean(wtri)) if n else float("nan"),
            "alpha_frac": float(np.mean(afrac)) if n else float("nan"),
        }

    if not table:
        print(f"  no steered runs found for {model}")
        return

    out: dict = {"model": model, "results": {}}
    for (vectag, arm), by_scalar in sorted(table.items()):
        print(f"\n  vector={vectag}  arm={arm}")
        print(f"    {'scalar':>7} {'n':>4} {'silent':>7} "
              f"{'lock_end':>9} {'lock_mid':>9} {'word_tri':>9} "
              f"{'alpha_frac':>11}")
        for scalar in sorted(by_scalar):
            s = by_scalar[scalar]
            print(f"    {scalar:>+7.1f} {s['n']:>4d} {s['n_silent']:>7d} "
                  f"{s['lock_end']:>9.2f} {s['lock_mid']:>9.2f} "
                  f"{s['word_tri']:>9.3f} {s['alpha_frac']:>11.3f}")
        out["results"][f"{vectag}|{arm}"] = {
            f"{k:+.1f}": v for k, v in sorted(by_scalar.items())
        }

    out_path = DATA_DIR / "local" / "basin_stats" / f"{model}_steering_continuation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--all-models", action="store_true")
    args = ap.parse_args()
    models = ["gemma", "qwen", "ministral"] if args.all_models else [args.model]
    for model in models:
        run(model)


if __name__ == "__main__":
    main()
