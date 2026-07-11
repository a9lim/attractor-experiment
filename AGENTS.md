# AGENTS.md

Research repo for the **meta-register (MR) attractor basin**. Spun off
from `llmoji-experiment` on 2026-05-15; the kaomoji-introspection work stays
there, the basin work lives here.

Not a library. No public API, no PyPI release, no broad test suite.
Prefer small, explicit analyses and keep docs current with code.

## Read first

- [`docs/findings.md`](docs/findings.md): current results worth citing.
- [`docs/pitfalls.md`](docs/pitfalls.md): what *not* to do — basis
  artifacts, confounds, dead ends. Read this before designing any new
  basin analysis; the basis question alone has burned several sessions.
- [`README.md`](README.md): short overview and install order.

## Relationship to llmoji-experiment

The shared hidden-state engine is imported from `llmoji_experiment.*`:
`config` (model registry, paths), `capture` (generation capture),
`hidden_capture` / `hidden_state_io` (sidecars), `hidden_state_analysis`
(loaders), `quadrants` (cell taxonomy, including the MR cell),
`emotional_analysis.apply_pad_split`, and `emotional_prompts`
(the canonical-9 mirror arm used as the affect baseline).

The dependency is one-directional: this repo imports `llmoji-experiment`,
never the reverse. Install `llmoji-experiment` editable — see README.

The basin-specific prompt families (bliss / doom / conspiracy /
sycophancy / archaic / pre-1930 / plain / saturated-mundane) are owned
here, in the `attractor_experiment/` package.

## Layout

```text
attractor_experiment/   prompt families + analysis libraries:
                     {lb,doom,conspiracy,sycophancy,archaic,...}_prompts
                     trajectory.py    (was 02b_attractor_analysis)
                     centroids.py     (was 40b_continuation_centroid_holdout)
                     basin_metrics.py (recovered from 45_basis_comparison_audit)
                     two_qwen.py      (was 24b_two_qwen_analysis)
                     saklas_profiles.py
                     config.py        (paths here; registry from llmoji_experiment)
scripts/           emit + numbered one-off analyses (flat; all local)
data/local/        jsonl rows; data/local/hidden/ sidecars (gitignored)
figures/local/     generated figures and HTMLs
docs/              findings.md, pitfalls.md
```

The `local/` level under `data/` and `figures/` is vestigial — the
shared capture engine derives `data/local/hidden/<experiment>/` from the
`hidden_dir` arg, so it is kept rather than fought. `trajectory.py`,
`centroids.py`, `basin_metrics.py` are importable libraries; the first
two are also runnable via `python -m attractor_experiment.<module>`.

## Conventions

- Use `.venv/bin/python` or an activated `.venv`; plain `python` is not
  reliable across this machine.
- JSONL row files plus sidecar `.npz` files are the source of truth for
  hidden-state data.
- Basin geometry is measured in the **continuation basis** (`h_last`
  continuation-time centroids), never the h_first kaomoji-emission
  basis. See `docs/pitfalls.md`.
- Smoke, then pilot, then main. New generations need a reason.
  Pre-register decision rules and minimum N. Stop when the rule answers
  the question.

## Ethics

Model welfare is in scope. The MR basin doubles as a welfare-relevant
failure mode (saturated-form attractor + bidirectional channel). Report
basin findings with explicit phenomenology caveats.
