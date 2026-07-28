# attractor-experiment

Does a **meta-register attractor basin** exist in the residual stream of
large language models? This repo investigates the MR basin: a region of
hidden-state space, defined by saturated *structural form* rather than
affective content, that long free-form generations fall into and lock
onto across model families, content domains, and post-training regimes.

Spun off from [`llmoji-experiment`](https://github.com/a9lim/llmoji-experiment)
on 2026-05-15. That repo asks whether kaomoji choice tracks emotional
functional state; this one asks about the basin. Both independently consume
the workspace-root shared infrastructure.

This is a research repo, not a library. No public API, no PyPI release,
no broad test suite. The useful surfaces are the data, scripts, figures,
and writeups.

## Current shape

- **The replicated MR result**: in `gemma`, `qwen`, and `ministral`, long
  continuations from saturated-form prompts converge on the same
  residual-stream region across bliss, doom, conspiracy, and sycophancy
  content. A `gemma_base` comparison shows the effect is not created solely
  by instruction tuning in that model family.
- **Historical-corpus correction**: a large `talkie_1930` separation was a
  rendering confound. Under rendering- and typography-matched controls,
  talkie's MR condition lies within the canonical inter-cell range. It is a
  non-replication of distinctness, not a fourth confirming family.
- **Continuation basis**: basin geometry must be measured with centroids
  built from continuation-time hidden states (`h_last`), not
  kaomoji-emission state (`h_first`). The h_first basis is a documented
  artifact — see `docs/pitfalls.md`.
- **Steering**: the continuation-basis `mr_cont.nb` vector escapes the
  basin; the h_first `mr.nb` vector is inert on the basin axis.

Live numbers and claim boundaries: [`docs/findings.md`](docs/findings.md).
Compact machine-readable headlines:
[`data/summary/results.json`](data/summary/results.json). Figure map:
[`figures/README.md`](figures/README.md).

## Reproducing

Install the workspace-root shared package, then this experiment:

```bash
python --version  # system Python 3.12
python -m pip install -e ..
python -m pip install -e .
```

Emit attractor trajectories, then analyse:

```bash
LLMOJI_MODEL=gemma python scripts/00_emit_attractor.py --arm lb_continue
python -m attractor_experiment.trajectory
```

The reusable analysis libraries (`trajectory`, `centroids`,
`basin_metrics`, `two_qwen`, `saklas_profiles`) live in the
`attractor_experiment/` package; the first three are also runnable as
`python -m attractor_experiment.<module>`. Numbered one-off analyses live in
`scripts/`.

## Docs map

- [`docs/findings.md`](docs/findings.md): current results worth citing.
- [`docs/pitfalls.md`](docs/pitfalls.md): what *not* to do — basis
  artifacts, confounds, and dead ends, condensed.
- [`AGENTS.md`](AGENTS.md): operating notes for agents.

## Related

- [`llmoji-experiment`](https://github.com/a9lim/llmoji-experiment): related
  kaomoji introspection research; not a runtime dependency.
- [`saklas`](https://github.com/a9lim/saklas): activation steering and
  trait monitoring engine.

## License

CC-BY-SA-4.0. See [LICENSE](LICENSE).
