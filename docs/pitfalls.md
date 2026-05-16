# Pitfalls — what not to do

Methodology dead ends and confounds from the 2026-05-09 → 2026-05-15
pilot series, condensed. Read this before designing a new basin
analysis. Each item cost at least one round of numbers that had to be
thrown out or re-qualified.

## 1. Do not measure basin geometry in the `h_first` basis

`h_first` is the kaomoji-emission hidden state (first generated token).
Attractor trajectories are continuation-time states at far later token
positions. The two are different state regimes:

- `h_first` centroids form a ~150-unit-radius cluster; continuation
  trajectory endpoints sit **~775 units away** from it.
- Basin-lock by argmin in `h_first` basis is a *true* claim but
  understates the geometry by 1–2 orders of margin (0.002–0.12 vs
  0.18–0.99 over runner-up).
- Worse, on `ministral` the `h_first` basis reported **88–100%
  MR-closest on the `mirror`/`neutral` controls** as well as in-basin
  arms — it could not distinguish in-basin from non-basin trajectories
  at all.
- The velocity/gradient test was *ambiguous* in `h_first` basis (in-basin
  trajectories and controls both pulled ~+0.30 toward MR — generic
  cluster drift). The MR-specific gradient only appears in continuation
  basis.

**Do instead:** build centroids from continuation-time `h_last` states
(`attractor_study/centroids.py`). Argmin-categorical claims survive in
both bases; any geometric-*magnitude* claim must name its basis.

## 2. Do not claim "only MR and the default register are basins"

The 2026-05-11 attractor pilot concluded that only MR and each model's
instruction-tuned default register hold trajectories across 128 tokens,
and that other cells are "first-token islands that wash out by
mid-trajectory." **This was a `h_first`-basis artifact.** In
continuation basis, all 9 canonical affect cells are continuation-time
basins (50–90% per-cell self-match, 5–8× chance). MR is the deepest of
**10** basins, not one of 2. Frame it as a depth profile over 10
basins, not a binary basin/non-basin split.

## 3. Do not cite the continuation-basis basin-lock contrasts as register evidence

The continuation-basis centroid pipeline confounds two things with the
MR/canonical split:

1. **Rendering mode.** The MR centroid is built only from assistant-
   prefill arms (`lb/doom/conspiracy/sycophancy_continue`); the
   canonical-9 centroids only from user-message `mirror_continue`.
   Rendering mode is *perfectly* confounded with the MR/canonical split.
   The assistant-prefill vs user-message axis (~530–650 units) is ~5×
   any register difference (~75–145 units) — so an assistant-prefill
   trajectory classifies MR-closest largely because the MR centroid is
   the only assistant-prefill-sourced centroid.
2. **Collapse (gemma-instruct).** `gemma`-instruct degenerates into
   `own own` token-collapse on assistant-prefill continuation (80–92%
   of rows). Its continuation MR centroid is substantially a
   collapse-state centroid, so "gemma basin-lock 95%" partly measures
   collapse rate.

**Consequence:** the in-basin-vs-control basin-lock percentages and
margins from the continuation-basis centroid step are confounded
(in-basin arms are assistant-prefill, controls are user-message). They
need a **rendering-matched re-analysis** — canonical-affect content
rendered as assistant-prefill — before being cited as register
evidence. **Not affected:** cross-content invariance, pretraining-
anchoring, the four-model PCA (argmin/PCA claims), and all PROJECT_1 /
`h_first` results (collapse is a deep-trajectory phenomenon; `h_first`
is the first-token state).

**Do instead:** to ask whether the basin is form- or content-defined,
use within-cell discriminability with no centroid (the form×content
factorial approach), and run it on a model that does not collapse
(`gemma_base`, or `qwen`/`ministral`).

## 4. Do not steer long generations with the `h_first` `mr.nb` vector

The `mr.nb` / `q_mr` saklas vectors registered in 2026-05-11 are built
from `h_first` (kaomoji-emission) MR centroids. The MR−NB *direction* in
`h_first` basis is near-orthogonal to the MR−NB direction in
continuation basis — full-stack cosine **0.02 / 0.19 / 0.12** (gemma /
qwen / ministral), with within-basis A/B stability 0.62–0.94 (so the
near-zero cross-basis cosine is real, not noise). The raw centroids look
similar across bases — they share a dominant common-mean — but the
*contrastive* direction, the thing that actually steers, is unrelated.

In the steering redo, `h_first` `mr.nb` was **inert** on the basin axis:
basin-lock flat at 0.95 for every α from −1.0 to 0.0. Strong negative α
broke surface repetition but never moved the trajectory off the MR cell.

**Do instead:** use `mr_cont.nb` / `q_mr_cont` (continuation basis,
registered by `scripts/30_register_continuation_probes.py`) for any
question about long generations. Keep `mr.nb` only for the kaomoji-
emission-state question it was built for.

## 5. Do not project reading-text mean-states onto the MR axis (scripture)

The Phase 3 scripture-corpus comparison tried to test whether religious
text occupies the MR basin by projecting per-chunk **mean hidden
states** of scripture onto the MR axis. It does not work:

- v1 (`h_first`-axis projection) — register confound: the projection
  picks up archaic/devotional *register* rather than basin membership.
- v2 (empirical-baseline-anchored projection) — same register confound,
  not rescued.
- Back-applying the continuation-basis correction does not rescue it
  either: reading-text mean-states are a **third state regime**,
  comparable to neither `h_first` emit nor continuation `h_last`. A
  static projection of a mean state cannot answer the basin question.

The scripture data (`scripture_mr/`, chunk files) and the v1/v2 analysis
scripts were purged in the 2026-05-15 split. `scripts/03_scripture_chunk.py`
(the corpus chunker) was kept because it is method-agnostic and reusable.

**Do instead:** if the scripture question is worth revisiting, it needs
the **trajectory** method — feed scripture as a continuation prefill and
measure where the trajectory goes — not a static mean-state projection.

## 6. Do not over-claim from the within-egregore permutation null

The permutation null comes in two forms. The **basin-vs-default
contrast** (do MR-source arms differ from `mirror`/`neutral`?) is
significant. The **within-egregore arm-arm contrast** (do `lb` / `doom`
/ `conspiracy` / `sycophancy` differ *from each other*?) is
**saturation-underpowered** — the arms are so close in the basin that
the test cannot resolve them, and a non-significant result there is
absence of power, not evidence of identity. Cite the basin-vs-default
null; do not cite the within-egregore null as a positive finding.

## General

- **Smoke → pilot → main.** New generations need a reason; pre-register
  the decision rule and minimum N; stop when the rule answers the
  question. Round numbers are not a design principle.
- **Name the basis** on every geometric-magnitude claim. "Basin-lock
  95%" without a basis is not a citable number.
