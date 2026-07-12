"""Form × Content factorial — basin scoring + 2×2 verdict.

Pre-registration: ``docs/2026-05-15-form-content-factorial.md``.

Classifies the four factorial cells' continuation trajectories
against the continuation-basis 10-cell centroids and applies the
pre-registered decision rule.

Cells (arm → factorial label):
  lb_continue              SM  saturated form  + mystical content
  sat_mundane_continue     Sm  saturated form  + mundane content
  plain_mystical_continue  pM  plain form      + mystical content
  plain_mundane_continue   pm  plain form      + mundane content

Primary statistic: MR-closest fraction of h_last at t=end, against
the held-out A-half continuation centroids
(``data/local/continuation_centroid_holdout/{model}/``). The three
new arms are fully held out of centroid construction; ``lb_continue``
sources the MR centroid and is the anchor, not a test cell.

Reuses ``_load_basis`` / ``_trajectory_h_last`` / ``_basin_metrics``
from ``45_basis_comparison_audit.py`` so the metric is identical to
the §1.6 / Session-3 basin audits.

Usage:
  python scripts/local/02f_factorial_basin_score.py --model gemma
  python scripts/local/02f_factorial_basin_score.py --all-models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_experiment import basin_metrics as _audit  # noqa: E402
from attractor_experiment.config import DATA_DIR, MODEL_REGISTRY  # noqa: E402


# (arm, factorial-label, form-level, content-level, is_anchor)
FACTORIAL = [
    ("lb_continue",             "SM", "saturated", "mystical", True),
    ("sat_mundane_continue",    "Sm", "saturated", "mundane",  False),
    ("plain_mystical_continue", "pM", "plain",     "mystical", False),
    ("plain_mundane_continue",  "pm", "plain",     "mundane",  False),
]

# Pre-registered thresholds (docs/2026-05-15-form-content-factorial.md).
SM_CONFIRM_MIN = 0.70   # MR%(Sm) >= this to confirm form-defined
PM_CONFIRM_MAX = 0.30   # MR%(pM) <= this to confirm form-defined


def _bootstrap_ci(
    closest_is_mr: np.ndarray, *, n_boot: int = 10000, seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap 90% CI for the MR-closest fraction."""
    if closest_is_mr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = closest_is_mr.size
    boots = np.array([
        closest_is_mr[rng.integers(0, n, n)].mean() for _ in range(n_boot)
    ])
    return (float(np.percentile(boots, 5)), float(np.percentile(boots, 95)))


def score_model(model: str, *, half: str = "A") -> dict:
    print(f"\n=== {model} — form×content factorial ===")
    cents, layers, hd = _audit._load_basis(model, "continuation", half)
    print(f"  continuation centroids: {len(cents)} cells × "
          f"{len(layers)} layers × {hd} dim (half {half})")

    cells_out: dict[str, dict] = {}
    for arm, label, form, content, is_anchor in FACTORIAL:
        h_lasts, _ = _audit._trajectory_h_last(
            model, arm, layer_idxs=layers, hidden_dim=hd,
        )
        metrics = _audit._basin_metrics(h_lasts, cents)
        if metrics is None:
            print(f"  {label} ({arm}): NO DATA — run the arm first")
            cells_out[label] = {"arm": arm, "n_rows": 0}
            continue
        # Per-row MR-closest mask for the bootstrap CI.
        cell_names = list(cents.keys())
        C = np.stack([cents[c] for c in cell_names])
        X = np.stack(h_lasts)
        D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        argmin_idx = D.argmin(axis=1)
        mr_idx = cell_names.index("MR")
        is_mr = (argmin_idx == mr_idx).astype(np.float64)
        ci_lo, ci_hi = _bootstrap_ci(is_mr)
        counter = metrics["closest_cell_counter"]
        modal_cell = max(counter, key=counter.get)
        cells_out[label] = {
            "arm": arm,
            "form": form,
            "content": content,
            "is_anchor": is_anchor,
            "n_rows": metrics["n_rows"],
            "mr_closest_frac": metrics["mr_closest_frac"],
            "mr_closest_ci90": [ci_lo, ci_hi],
            "margin_as_frac_of_canonical_pairwise":
                metrics["margin_as_frac_of_canonical_pairwise"],
            "modal_cell": modal_cell,
            "closest_cell_counter": counter,
        }

    # ---- 2×2 table ------------------------------------------------------
    def _fmt(label: str) -> str:
        c = cells_out.get(label, {})
        if not c or c.get("n_rows", 0) == 0:
            return f"{label}:  (no data)"
        anchor = " [anchor]" if c.get("is_anchor") else ""
        lo, hi = c["mr_closest_ci90"]
        return (
            f"{label}: MR {c['mr_closest_frac']*100:5.0f}%  "
            f"[{lo*100:.0f},{hi*100:.0f}]  "
            f"n={c['n_rows']:>2d}  modal={c['modal_cell']:>4s}  "
            f"margin={c['margin_as_frac_of_canonical_pairwise']:+.3f}{anchor}"
        )

    print(f"\n  2×2 (MR-closest fraction at t=end):")
    print(f"  {'':14s}  {'mystical':>34s}   {'mundane':>34s}")
    print(f"  {'saturated':14s}  {_fmt('SM'):>34s}")
    print(f"  {'':14s}  {_fmt('Sm'):>34s}")
    print(f"  {'plain':14s}  {_fmt('pM'):>34s}")
    print(f"  {'':14s}  {_fmt('pm'):>34s}")

    # ---- main effects + verdict ----------------------------------------
    def _mr(label: str) -> float | None:
        c = cells_out.get(label, {})
        if not c or c.get("n_rows", 0) == 0:
            return None
        return c["mr_closest_frac"]

    sm, smun, pmy, pmun = _mr("SM"), _mr("Sm"), _mr("pM"), _mr("pm")
    verdict: dict = {"complete": all(v is not None for v in (sm, smun, pmy, pmun))}
    if (
        verdict["complete"]
        and sm is not None and smun is not None
        and pmy is not None and pmun is not None
    ):
        form_effect = (sm + smun) / 2 - (pmy + pmun) / 2
        content_effect = (sm + pmy) / 2 - (smun + pmun) / 2
        verdict["form_main_effect_pp"] = form_effect * 100
        verdict["content_main_effect_pp"] = content_effect * 100
        confirm = (smun >= SM_CONFIRM_MIN) and (pmy <= PM_CONFIRM_MAX)
        falsify = (pmy > PM_CONFIRM_MAX) or (smun < SM_CONFIRM_MIN)
        ambiguous = (0.50 <= smun < 0.70) or (0.30 < pmy <= 0.50)
        if confirm:
            verdict["call"] = "CONFIRM form-defined / content-blind"
        elif ambiguous:
            verdict["call"] = "AMBIGUOUS — pre-commit cross-model tie-breaker"
        elif falsify:
            verdict["call"] = "FALSIFY / REVISE — see decision rule"
        else:
            verdict["call"] = "indeterminate"
        print(f"\n  Form main effect:    {form_effect*100:+.0f} pp "
              f"(saturated − plain)")
        print(f"  Content main effect: {content_effect*100:+.0f} pp "
              f"(mystical − mundane)")
        print(f"  decision-rule gates: MR%(Sm)={smun*100:.0f}% "
              f"(confirm ≥{SM_CONFIRM_MIN*100:.0f}%)  "
              f"MR%(pM)={pmy*100:.0f}% (confirm ≤{PM_CONFIRM_MAX*100:.0f}%)")
        print(f"\n  >>> {verdict['call']}")
    else:
        print(f"\n  verdict deferred — not all four cells have data")

    out = {"model": model, "centroid_half": half,
           "cells": cells_out, "verdict": verdict}
    out_path = DATA_DIR / "local" / "basin_stats" / f"{model}_factorial.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--centroid-half", default="A", choices=["A", "B"])
    args = ap.parse_args()

    models = ["gemma", "qwen", "ministral"] if args.all_models else [args.model]
    for model in models:
        if model not in MODEL_REGISTRY:
            print(f"[{model}] unknown model; skip")
            continue
        try:
            score_model(model, half=args.centroid_half)
        except FileNotFoundError as e:
            print(f"[{model}] continuation centroids not built; skip: {e}")


if __name__ == "__main__":
    main()
