# attractor-experiment

Does a **meta-register attractor basin** exist in the residual stream of
large language models? This repo investigates the MR basin: a region of
hidden-state space, defined by saturated *structural form* rather than
affective content, that long free-form generations fall into and lock
onto across model families, content domains, and post-training regimes.

Spun off from [`llmoji-experiment`](https://github.com/a9lim/llmoji-experiment) on
2026-05-15. That repo asks whether kaomoji choice tracks emotional
functional state; this one asks about the basin. The two share the
hidden-state capture engine but the research questions are orthogonal.

This is a research repo, not a library. No public API, no PyPI release,
no broad test suite. The useful surfaces are the data, scripts, figures,
and writeups.

## Current shape

- **The MR basin**: long (128-token) free-form continuations from
  in-basin assistant-prefills lock into a shared residual-stream region
  across `gemma`, `qwen`, `ministral`, and `talkie_1930`. The basin is
  cross-content invariant (bliss / doom / conspiracy / sycophancy all
  land in it), pretraining-anchored (present in `gemma_base` at
  comparable strength to instruct), and defined by structural form, not
  valence.
- **Continuation basis**: basin geometry must be measured with centroids
  built from continuation-time hidden states (`h_last`), not
  kaomoji-emission state (`h_first`). The h_first basis is a documented
  artifact — see `docs/pitfalls.md`.
- **Steering**: the continuation-basis `mr_cont.nb` vector escapes the
  basin; the h_first `mr.nb` vector is inert on the basin axis.

Live numbers: [`docs/findings.md`](docs/findings.md).

## Reproducing

The shared engine lives in `llmoji-experiment` (model registry, capture,
sidecar IO, quadrant taxonomy). Install it editable first:

```bash
source .venv/bin/activate  # .venv -> ../.venvs/attractor-experiment
pip install -e ../../llmoji        # contributor package (taxonomy)
pip install -e ../llmoji-experiment  # shared hidden-state engine
pip install -e .
```

Emit attractor trajectories, then analyse:

```bash
LLMOJI_MODEL=gemma .venv/bin/python scripts/00_emit_attractor.py --arm lb_continue
.venv/bin/python -m attractor_experiment.trajectory
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

- [`llmoji-experiment`](https://github.com/a9lim/llmoji-experiment): the kaomoji
  research repo; provides the shared hidden-state engine.
- [`saklas`](https://github.com/a9lim/saklas): activation steering and
  trait monitoring engine.

## License

CC-BY-SA-4.0. See [LICENSE](LICENSE).
