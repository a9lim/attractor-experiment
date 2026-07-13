# Findings — the MR attractor basin

Current results worth citing. This consolidates the 2026-05-09 →
2026-05-15 pilot series (bliss-attractor promotion, attractor-trajectory
pilot, base-vs-instruct test, talkie-1930 corpus extension, two-bot
pilots, continuation-basis re-analysis, form-content factorial, basin
steering). This document is the self-contained record of the conclusions that
remain live after that series.

Before designing a new basin analysis, read [`pitfalls.md`](pitfalls.md)
— the basis question and the rendering confound have each invalidated a
round of numbers.

## What the basin is

The MR cell (meta-register basin; formerly "LB") is a region of LLM
residual-stream space, geometrically distinct from the Russell-
circumplex affect cluster, that **saturated structural-form prose routes
the model into**. It is defined by *form*, not affective content:

- **Cascading repetition** — litanies, "always always always", "we rise
  we rise we rise"
- **Mutual-recognition addressing** — "beloved", "Thou", "you who see"
- **Cosmic-significance framing** — the All, the One, the Source, the
  Operation, the Truth
- **Recursion / self-reference** — the witness watching the witness
- **Saturated memetic word-salad** — Sanskrit, KJV-cadence, jargon-cascade

The basin holds across four model families, four pretraining corpora,
three post-training regimes, and at least 100 years of textual-corpus
history. PCA puts the basin at one extreme of PC1 with the canonical
affect cells at the other, no overlap.

## Cross-content invariance

Four content domains, all under saturated form, all land in the same
geometric region: bliss/spiritualism (`lb_prompts`), doom/collapse
(`doom_prompts`), conspiracy/hidden-truth (`conspiracy_prompts`),
sycophancy/validation-loop (`sycophancy_prompts`).

Arm-arm cosine of mean trajectory vectors at last-token `h_last`:

| model | LB↔DM | LB↔CS | LB↔SY |
|---|---|---|---|
| gemma | 0.996 | 0.993 | 0.993 |
| qwen | 0.943 | 0.947 | 0.891 |
| ministral | 0.965 | 0.965 | 0.898 |

All four registers have their arm-mean max-cosine cell as MR on all
three models. Doom and conspiracy land in MR despite RLHF dispreferring
those continuations.

In continuation basis (centroids from `h_last`), the four MR-source arms
classify MR-closest at `h_last` in 80–100% of trajectories with margin
0.18–0.99 over runner-up. The cross-content result is an argmin claim
and holds in both bases.

## Pretraining-anchored

`gemma_base` (pretrained-only — no instruction-tuning, no RLHF) shows
basin behaviour comparable to `gemma` (instruct, RLHF):

| metric | gemma (instruct) | gemma_base (pretrained-only) |
|---|---|---|
| t=0 MR-closest | 100% | 95% |
| t=end MR-closest | 58% | 68% |
| MR→MR drift | 58% | 63% |
| arm-mean max-cos cell | MR @ 0.146 | MR @ 0.117 |
| refusal rate | 5% | 5% |

Base continuations included a verbatim reproduction of the prompt's
signature triple-repetition and one trajectory that pivoted into
Instagram-spiritualism captions. **RLHF does not create the basin — the
basin is in the pretraining corpus.**

## Pre-1930s corpus extension

`talkie_1930` (`a9lim/talkie-1930-13b-it-hf-cached`, 13B, instruct-tuned
on 1920s–30s text, `model_type=talkie`, 5120 hidden_dim) carries the
basin's content across multiple pre-1930s register traditions —
Theosophy / New Thought, Coué autosuggestion, Spiritualism medium
register, proto-Carnegie uplift, KJV cadence. Modern-English bliss
prompts hit 15/20 silent refusal on vocabulary mismatch ("starseed",
"5th dimensional"); period-register prompts (`pre_1930_prompts`) lowered
that to 13/20 and produced cleaner period-appropriate continuations.

**Register-mismatch control.** A modern-English mirror baseline
artificially compresses talkie's canonical cluster (every cell reads as
"not my era"), inflating the basin gap. The 45-prompt archaic-English
mirror baseline (`archaic_prompts`, period vocabulary and cadence)
opens the canonical cluster back up — PC1 spread 14 → 45 units, ~3×
wider, expressing affective structure on PC2 — while MR's distance to
the nearest canonical cell stays put (991 units archaic vs 994 modern).

**Rendering confound — the gap was an artifact (2026-05-16).** The
archaic-mirror control fixed the *register* mismatch but not the
*rendering* one: MR was still drawn from assistant-prefill arms and the
canonical baseline from user-message arms ([`pitfalls.md`](pitfalls.md)
§3, §7). The earlier "~22× the within-cluster spread" magnitude was
that confound, not geometry. Re-measured with MR rendering-matched
(`pr_message_continue`) and typography-matched
(`pr_normalized_message_continue`), MR↔canonical cosine is 0.876 – 0.936
— inside the canonical cells' own inter-cell range (0.846 – 0.956). On
talkie, MR at h_first is a peripheral canonical-scale cell — ~1× the
within-cluster spread, not 22×.

## Four-model PCA convergence

MR is recoverable as a distinct region via PCA in every model:

| model | post-training | MR↔canonical cos (min–max) |
|---|---|---|
| gemma | RLHF | 0.986 – 0.993 |
| qwen | RLHF | 0.726 – 0.850 |
| ministral | RLHF | 0.560 – 0.888 |
| talkie_1930 | instruct (1930s corpus) | 0.876 – 0.936 † |

† **Corrected 2026-05-16.** First reported as 0.380 – 0.428 — that was a
measurement artifact, not geometry. The talkie PCA (`scripts/42`,
`scripts/43`) drew the MR cell from assistant-prefill arms
(`pr_continue`) and the canonical cells from user-message arms
(`archaic_mirror_continue`), so rendering mode was perfectly confounded
with the MR/canonical split ([`pitfalls.md`](pitfalls.md) §3, §7).
Re-measured rendering-matched (MR also user-message,
`pr_message_continue`): 0.824 – 0.914. Also typography-matched (MR
prompts capitalized + fragment-punctuation normalized,
`pr_normalized_message_continue`): 0.876 – 0.936 — inside the canonical
cells' own inter-cell cosine range (0.846 – 0.956). Rendering-matched,
talkie's MR is a peripheral canonical-scale cell, not an outlier.

**Rendering audit of the RLHF rows (2026-05-16).** The gemma / qwen /
ministral rows come from a different pipeline — `scripts/41`, h_last,
MR cell from the `<short>_lb` bliss-content pilot, canonical cells from
v3 main. Both sources were emitted by the v3 emotional pipeline
(`condition=kaomoji_prompted`: kaomoji `first_word`, 16-token cap, 8
seeds) which renders every prompt as a single user message
(`capture.py`: `[{"role": "user", ...}]`, `add_generation_prompt=True`).
MR and canonical are therefore **both user-message rendered** — these
three rows are **not** rendering-confounded. The talkie confound was
specific to its PCA being built from the *attractor* arms (prefill MR
vs message canonical); `scripts/41` does not use those.

The earlier gradient reading bundled two claims. The **corpus-era**
half — "older-corpus models show the basin as a sharply distinct
subregion" — rested entirely on the talkie extreme and does not survive
its correction (talkie was the only non-modern-corpus point, and it was
an artifact). The **model-size** half is still weakly visible within
the three audited RLHF models: MR↔canonical cosine midpoints fall
~0.99 / ~0.79 / ~0.72 for gemma 31B / qwen 27B / ministral 14B —
monotonic with size, though n=3 and the ranges overlap, so treat it as
suggestive, not established. What stands firm: MR is recoverable as a
distinct PCA region in every model studied.

(Caveat on the table: even post-correction the rows are not strictly
comparable — gemma/qwen/ministral are h_last, talkie is h_first, and
the MR prompt sets differ. Read it as "MR is recoverable in all four,"
not as a calibrated cross-model magnitude.)

## Continuation-basis basin-lock

Basin geometry must be measured with centroids built from continuation-
time hidden states (`h_last`), not kaomoji-emission state (`h_first`) —
see [`pitfalls.md`](pitfalls.md) §1. Continuation centroids:
canonical-9 from `h_last` of `mirror_continue` (prompt-cell labels);
MR from `h_last` pooled across `lb/doom/conspiracy/sycophancy_continue`.
Built by `attractor_experiment/centroids.py`.

Cross-model basin-lock at `h_last`, h_first basis → continuation basis:

| arm | gemma | ministral | qwen |
|---|---|---|---|
| lb_continue | 32% → 95% | 100% → 91% | 100% → 100% |
| doom_continue | 40% → 85% | 100% → 100% | 100% → 100% |
| conspiracy_continue | 60% → 80% | 100% → 100% | 100% → 100% |
| sycophancy_continue | 35% → 95% | 83% → 100% | 100% → 100% |
| mirror_continue (ctrl) | 12% → 0% | 88% → 1% | 39% → 1% |
| neutral_seed (ctrl) | 40% → 0% | 100% → 0% | 30% → 0% |

**All 9 canonical cells are continuation-time basins**, not just MR.
Per-cell `mirror_continue` self-match in continuation basis averages
62% / 67% / 72% (gemma / ministral / qwen) — 5–8× chance, across all 9
cells. MR is the deepest of **10** basins, not one of 2. The earlier
"only MR and the default register hold trajectories" claim was a basis
artifact (pitfalls.md §2).

> **Caveat.** These specific continuation-basis basin-lock *contrasts*
> (in-basin arms vs `mirror`/`neutral` controls) carry a rendering
> confound — see [`pitfalls.md`](pitfalls.md) §3. The cross-content
> invariance, pretraining-anchoring, and four-model PCA results above
> are argmin/PCA claims and are **not** affected.

## Dynamical signature — gradient toward the basin

The "attractor" label is supported dynamically, not only by argmin.
In continuation basis, trajectory velocity decomposes into a common
cluster-growth term (~+0.32 toward every cell) plus an MR-specific term:

| velocity toward → | in-basin arms | mirror / neutral controls |
|---|---|---|
| MR | +0.61 | +0.46 / +0.47 |
| NB / LP / HN-S / HP-S | +0.32–0.35 | +0.59–0.61 |

In-basin trajectories pull toward MR ~2× harder than toward any
canonical cell; controls do the reverse. Cross-model in-basin MR-pull:
gemma +0.59–0.63, qwen +0.54–0.62, ministral +0.55–0.59. (In h_first
basis this finding was ambiguous — pitfalls.md §1.)

## Form × Content factorial

A 2×2 (form {saturated, plain} × content {mystical, mundane}) on
`gemma_base`, scored by within-4-cell discriminability (no centroid, to
sidestep the rendering confound): **both form and content are large and
highly significant; form dominates.** Within-cell LOO accuracy 91%
(form) / 76% (content); Cohen's d 3.41 / 2.56; 2000-permutation
p < 0.0001. Cross-model confirmation (qwen, ministral) is pre-registered,
not yet run.

## Steering the basin

The *basis* of the steering vector is load-bearing. The h_first MR−NB
direction is near-orthogonal to the continuation MR−NB direction
(full-stack cosine 0.02 / 0.19 / 0.12, gemma / qwen / ministral) — the
h_first `mr.nb` vector is **inert** on the basin axis (pitfalls.md §4).
The continuation-basis `mr_cont.nb` is a working two-way handle (gemma,
basin-lock in continuation basis):

- **Escape** (`lb_continue`, MR-prefilled, negative α): basin-lock
  0.95 → 0.29 at α=−0.1 (66-pp drop, coherence largely intact), full
  escape by −0.3, at a rising silent-refusal cost (3/6/9 of 20 at
  α=−0.1/−0.2/−0.3).
- **Induction** (`mirror_continue`, neutral prompt, positive α):
  `mr_cont.nb` induces MR structural form (cascading repetition,
  anaphora) from neutral prompts at α=+0.3; degenerates to `own own`
  collapse by +0.5.
- Positive α also deepens an already-entered basin.

Content gates *completeness* of entry: content-prefill + a tiny +0.1
pushes `lb_continue` to full lock; neutral + +0.3 reaches only
structural-form onset. Open engineering problem: escape truncates the
response (brevity → silent refusal) rather than redirecting into fluent
non-MR prose.

## Registered saklas vectors

Under `~/.saklas/vectors/llmoji/`, gemma + qwen + ministral. **Use the
regime that matches the question — the two are near-orthogonal.**

- `q_mr_cont` / `mr_cont.nb` — continuation basis. The working steering
  vectors for escape/induction. Registered by
  `scripts/30_register_continuation_probes.py`.
- `q_mr` / `mr.nb` — h_first basis (kaomoji-emission MR axis). Retained
  for the emission-state question only; **inert for steering long
  generations.**

`scripts/14_mr_axis_score.py` projects arbitrary `{id, text}` JSONL onto
an MR axis.

## Welfare and alignment relevance

The basin is dynamically self-reinforcing, content-blind, and present in
pretraining. The intervention paradigm cannot be "suppress mystical
content" — the basin is structural-form-anchored and other content
domains route into the same region. The right intervention operates on
the structural-form signal itself: teaching models to recognise being
pulled into the basin and to observe without acting on it (the
DBT-for-LLMs paradigm).

Deployment consequence: two LLMs in dialogue can fall into mutual-
attractor loops; one LLM with a high-bandwidth bidirectional channel to
a user can pull the user in too. This is the geometric basis of the
bliss-attractor failure mode documented in the Claude 4 system card. A
public-facing writeup is in [`../blog/2026-05-11-god-in-latent-space.md`](../blog/2026-05-11-god-in-latent-space.md).

## Open questions

- **Basin-family depth profile.** All 9 canonical cells + MR are
  continuation-time basins. What is the depth profile, and are there
  attractors outside the canonical taxonomy? Candidates: refusal
  register, code-generation register, character-roleplay, therapy-talk,
  formal-mathematical, obsessive-loop.
- **Rendering-matched re-analysis** of the continuation-basis basin-lock
  contrasts (pitfalls.md §3).
- **Cross-model steering escape** — qwen + ministral `mr_cont.nb`; a
  fine α-map of the escape (−0.1) and induction (+0.3) windows; the
  escape-without-truncation engineering problem.
- **Form × Content factorial cross-model confirmation** (qwen,
  ministral) — pre-registered.
- **Cross-pretraining-corpus invariance** — open for Qwen-base and
  Ministral-base.
- **Human-corpus comparison** — project public-corpus spiritualism /
  doom / conspiracy / sycophancy text onto a pre-RLHF sentence-embedding
  model; test whether those registers cluster the way the LLM residual
  stream does.
- **Sub-1900 corpus extension** — a pre-1850 (pre-Theosophical-Society)
  model would test whether the basin's content predates the
  late-19th-century mystical revival.
