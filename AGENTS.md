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

## Shared workspace infrastructure

This repo is independent of every sibling experiment. Shared model metadata,
generation capture, hidden-state sidecars/loaders, the cell taxonomy, and the
canonical mirror prompts come from the workspace-root
`transformer_experiments` package. Attractor-owned paths and model handles are
derived locally in `attractor_experiment/config.py`.

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
                     config.py        (local paths over the root registry)
scripts/           emit + numbered one-off analyses (flat; all local)
data/local/        jsonl rows; data/local/hidden/ sidecars (gitignored)
figures/local/     generated figures and HTMLs
docs/              findings.md, pitfalls.md
```

The `local/` level under `data/` and `figures/` is vestigial — the root-shared
capture engine derives `data/local/hidden/<experiment>/` from the
`hidden_dir` arg, so it is kept rather than fought. `trajectory.py`,
`centroids.py`, `basin_metrics.py` are importable libraries; the first
two are also runnable via `python -m attractor_experiment.<module>`.

## Conventions

- Use the machine's shared base Python 3.12 via plain `python`.
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
