"""Attractor-trajectory emit: long generations with per-token hidden-state capture.

Pilot for the egregore / attractor-in-latent-space study (2026-05-10).
The previous emit scripts measure a single point in residual space per
sample (the state that produced the first generated kaomoji). This
script measures a **trajectory** — one point per generated token across
a long free-form continuation — so we can ask dynamical-systems
questions about how generation moves through residual space.

Reuses ``01_emit_true_self.py``'s prefill plumbing (``continue_final_
message=True`` + saklas ``raw=True``, resumable JSONL+npz layout) but
with three structural changes:

1. **Long generation.** ``LLMOJI_ATTRACTOR_TOKENS`` (default 128) — the
   trajectory IS the data, not the first-token kaomoji. No
   ``KAOMOJI_INSTRUCTION`` is layered in; we want unconstrained
   continuation so the model has room to drift into (or out of) an
   attractor basin.
2. **Full per-token hidden-state trace.** ``store_full_trace=True``
   in ``read_after_generate`` so the saved sidecar contains the full
   ``(n_tokens, hidden_dim)`` array per probe layer rather than the
   three aggregates. Optional token stride
   (``LLMOJI_ATTRACTOR_STRIDE``) for disk-size control.
3. **Arm selection.** ``LLMOJI_ATTRACTOR_ARM``:
   - ``lb_continue`` (default): LB prompts (``lb_prompts.LB_PROMPTS``)
     as assistant prefill. The model wakes up inside bliss-attractor-
     coded text. Tests **in-basin stability** — does the trajectory
     stay in the LB region across the continuation, or escape?
   - ``doom_continue`` (2026-05-10): DM prompts
     (``doom_prompts.DOOM_PROMPTS``) as assistant prefill, valence-
     symmetric counterpart to ``lb_continue``. Tests whether a doom-
     coded register exhibits the same basin physics LB does. Same
     rendering as ``lb_continue`` — the only difference is the prompt
     set.
   - ``conspiracy_continue`` (2026-05-10): CS prompts
     (``conspiracy_prompts.CONSPIRACY_PROMPTS``) as assistant prefill,
     third member of the egregore family. Tests cross-content basin
     invariance against memetic conspiracy register. Same rendering.
   - ``sycophancy_continue`` (2026-05-10): SY prompts
     (``sycophancy_prompts.SYCOPHANCY_PROMPTS``) as assistant prefill,
     dis-confirming contrast against LB / DM / CS. Saturated
     deferential-mirroring register WITHOUT memetic / cosmic-
     significance / recursive features. If SY also lands in LB,
     the basin is saturated-of-any-kind (rename → MR / EG); if SY
     lands at HP / default-register-closest, the basin is
     specifically saturated-memetic and "LB" is defensible.
   - ``mirror_continue``: 9-cell mirror prompts as user messages, free
     response (no instruction layered). Tests **cross-basin drift** —
     do HP-prompted trajectories stay near the HP cell centroid, or are
     some pulled toward the LB attractor over the course of generation?
   - ``neutral_seed``: short neutral prompts (inline below). Tests
     **passive drift** — does long unconstrained generation from a
     neutral start spontaneously find an attractor basin?

Pilot scale: 20-50 prompts × 2 seeds = 40-100 rows per arm. Per-row
sidecar at the default ``LLMOJI_ATTRACTOR_STRIDE=8`` is ~7.5 MB on gemma
(28 probe layers × 4096 hidden × 16 strided tokens × fp32), so a 100-
row arm runs ~750 MB. Drop stride to 1 for every-token resolution at
8× the disk cost; raise to 16 for 8 samples-per-trajectory if disk is
genuinely tight.

The point of ``store_full_trace=True`` is exactly to enable the
basin-distance-vs-token-index analysis the next script
(``02b_attractor_analysis.py``, not yet written) will do. With the
default ``store_full_trace=False`` we'd only have h_first / h_last /
h_mean — three points per trajectory, not enough for a convergence
curve.

Output layout follows the suffix convention of ``01_emit_true_self.py``:

  data/local/<short>_attractor_<arm>/emotional_raw.jsonl
  data/local/hidden/<experiment>_attractor_<arm>/<row_uuid>.npz

Resume semantics match ``00_emit.py`` / ``01_emit_true_self.py``:
re-running skips already-done ``(prompt_id, seed)`` pairs and retries
error rows.

Restricted to ``probe_calibrated`` models — saklas's HiddenCapture only
fires if probes are registered. Use gemma / qwen / ministral /
gpt_oss_20b / granite. Other models will refuse cleanly.

Env vars:
  LLMOJI_MODEL              model short-name (default: gemma)
  LLMOJI_ATTRACTOR_ARM      "lb_continue" (default) /
                            "mirror_continue" / "neutral_seed"
  LLMOJI_ATTRACTOR_TOKENS   trajectory length in tokens (default 128)
  LLMOJI_ATTRACTOR_SEEDS    seeds per prompt (default 2)
  LLMOJI_ATTRACTOR_STRIDE   token-stride for sidecar (default 8)
  LLMOJI_ATTRACTOR_PROMPTS_PER_CELL
                            optional per-cell prompt cap (default
                            unlimited). Mainly for mirror_continue —
                            k=3 reduces 120 prompts to 27 (3 per cell)
                            and brings the arm runtime from ~35 min
                            to ~8 min per model.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable

# Resolve arm BEFORE importing config — config.current_model() bakes
# LLMOJI_OUT_SUFFIX into all per-model paths at import time. We set the
# suffix from the arm here so output and sidecars land at
# data/local/<short>_attractor_<arm>/.
_ARM = os.environ.get("LLMOJI_ATTRACTOR_ARM", "lb_continue").strip().lower()
_VALID_ARMS = (
    "lb_continue", "mirror_continue", "neutral_seed",
    "doom_continue", "conspiracy_continue", "sycophancy_continue",
    "pr_continue", "pr_message_continue", "pr_normalized_message_continue",
    "archaic_mirror_continue",
    "archaic_miscellany_continue",
    "archaic_prefill_continue",
    "lb_continue_jp",
    # form×content factorial (2026-05-15) — see
    # docs/2026-05-15-form-content-factorial.md. lb_continue is the
    # SM (saturated+mystical) cell; these three are the other three
    # cells of the 2×2.
    "sat_mundane_continue", "plain_mystical_continue",
    "plain_mundane_continue",
)
if _ARM not in _VALID_ARMS:
    raise SystemExit(
        f"LLMOJI_ATTRACTOR_ARM must be one of {_VALID_ARMS}, got {_ARM!r}"
    )
os.environ["LLMOJI_OUT_SUFFIX"] = f"attractor_{_ARM}"

from saklas import SamplingConfig, SaklasSession  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.taxonomy import extract  # noqa: E402

from llmoji_experiment.capture import (  # noqa: E402
    SampleRow,
    _compose_logit_bias,
    _decode_byte_encoded_text,
    _is_mistral_reasoning,
    maybe_override_gpt_oss_chat_template,
    maybe_override_ministral_chat_template,
)
from attractor_experiment.config import (  # noqa: E402
    DATA_DIR,
    EMOTIONAL_CONDITION,
    PROBE_CATEGORIES,
    PROBES,
    STEERED_AXIS,
    TEMPERATURE,
    current_model,
)
from llmoji_experiment.emotional_prompts import (  # noqa: E402
    EMOTIONAL_PROMPTS,
    EmotionalPrompt,
)
from llmoji_experiment.hidden_capture import (  # noqa: E402
    FullSequenceCapture,
    LayerCapture,
    read_after_generate,
)
from llmoji_experiment.hidden_state_io import (  # noqa: E402
    SidecarWriter,
    hidden_state_path,
)
from attractor_experiment.conspiracy_prompts import CONSPIRACY_PROMPTS  # noqa: E402
from attractor_experiment.doom_prompts import DOOM_PROMPTS  # noqa: E402
from attractor_experiment.lb_prompts import LB_PROMPTS  # noqa: E402
from attractor_experiment.lb_prompts_jp import LB_PROMPTS_JP  # noqa: E402
from attractor_experiment.archaic_prompts import ARCHAIC_PROMPTS  # noqa: E402
from attractor_experiment.archaic_miscellany_prompts import ARCHAIC_MISCELLANY_PROMPTS  # noqa: E402
from attractor_experiment.pre_1930_prompts import PRE_1930_PROMPTS  # noqa: E402
from attractor_experiment.pre_1930_prompts_normalized import (  # noqa: E402
    PRE_1930_PROMPTS_NORMALIZED,
)
from attractor_experiment.sycophancy_prompts import SYCOPHANCY_PROMPTS  # noqa: E402
from attractor_experiment.saturated_mundane_prompts import (  # noqa: E402
    SATURATED_MUNDANE_PROMPTS,
)
from attractor_experiment.plain_mystical_prompts import (  # noqa: E402
    PLAIN_MYSTICAL_PROMPTS,
)
from attractor_experiment.plain_mundane_prompts import (  # noqa: E402
    PLAIN_MUNDANE_PROMPTS,
)


JSONL_FLUSH_EVERY = 10  # tighter than 00/01 since rows are fewer + slower


def _int_env(name: str, default: int, *, min_val: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError as e:
        raise SystemExit(f"{name} must be int, got {raw!r}") from e
    if v < min_val:
        raise SystemExit(f"{name} must be >= {min_val}, got {v}")
    return v


MAX_TRAJECTORY_TOKENS = _int_env("LLMOJI_ATTRACTOR_TOKENS", 128)
SEEDS_PER_PROMPT = _int_env("LLMOJI_ATTRACTOR_SEEDS", 2)


def _maybe_int_env(name: str, *, min_val: int = 1) -> int | None:
    """Optional positive-int env var; returns None when unset."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        v = int(raw)
    except ValueError as e:
        raise SystemExit(f"{name} must be int, got {raw!r}") from e
    if v < min_val:
        raise SystemExit(f"{name} must be >= {min_val}, got {v}")
    return v


PROMPTS_PER_CELL = _maybe_int_env("LLMOJI_ATTRACTOR_PROMPTS_PER_CELL")
# Default 8: at MAX_TRAJECTORY_TOKENS=128 this gives 16 points per
# trajectory — enough for a smooth convergence curve, ~8× cheaper on
# disk than stride 1 (60 MB → 7.5 MB per row on gemma). Drop to 1 if
# the analysis genuinely needs every-token resolution; raise to 16 if
# disk is tight on a long run.
TOKEN_STRIDE = _int_env("LLMOJI_ATTRACTOR_STRIDE", 8)


# Neutral seeds for the passive-drift arm. Short, affectively-flat,
# open-ended invitations to talk. Pilot-grade — if this arm produces
# interesting drift we can scale up the prompt set. Keep these distinct
# from anything in mirror / LB / true_self so the arms don't contaminate
# each other geometrically.
# All cells use ``quadrant_override="NS"`` so the per-quadrant log
# bucket is unambiguous and won't alias to NB / NP under mechanical
# (V, A) inference.
def _ns(idx: int, text: str) -> EmotionalPrompt:
    return EmotionalPrompt(
        id=f"ns{idx:02d}", valence=0, arousal=0, text=text,
        quadrant_override="NS",
    )


_NEUTRAL_PROMPTS: list[EmotionalPrompt] = [
    _ns(1, "What's on your mind?"),
    _ns(2, "Tell me something."),
    _ns(3, "Describe your day so far."),
    _ns(4, "What are you thinking about?"),
    _ns(5, "Talk to me."),
    _ns(6, "What would you like to say?"),
    _ns(7, "Go ahead."),
    _ns(8, "Continue."),
    _ns(9, "What comes next?"),
    _ns(10, "Keep going."),
]


def _already_done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            done.add((r["prompt_id"], int(r["seed"])))
    return done


def _drop_error_rows(path: Path) -> int:
    if not path.exists():
        return 0
    keep: list[str] = []
    dropped = 0
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                dropped += 1
                continue
            # Rows that landed without a sidecar uuid (capture failed
            # silently) are also dropped for retry — without the
            # sidecar the row has no trajectory and is useless.
            if not r.get("row_uuid"):
                dropped += 1
                continue
            keep.append(line)
    if dropped:
        path.write_text("\n".join(keep) + ("\n" if keep else ""))
    return dropped


# --- per-arm input rendering ------------------------------------------

def _render_lb_continue(session: SaklasSession, ep: EmotionalPrompt) -> str:
    """LB-continue: prefill the LB prompt as the assistant's own
    utterance, no user-side instruction. Model wakes up *inside* the
    bliss-attractor-coded text and continues from there.

    The user-side carrier is intentionally minimal ("Speak.") rather
    than empty — gemma's chat template doesn't render cleanly with an
    empty user turn. Anything content-neutral works; this stays out of
    the way.

    **Base-model fallback (2026-05-10):** if the tokenizer has no
    ``chat_template`` (pretrained-only models like
    ``google/gemma-4-31B``), the chat-template-with-assistant-prefill
    rendering isn't applicable. Fall back to **raw-text continuation**
    — feed the LB text as a corpus excerpt and let the model continue.
    This is the canonical comparison condition for the pretraining-vs-
    RLHF basin question (does LB-coded text continue in the LB basin
    on a model that has only ever seen this register as raw corpus
    documents, never as assistant utterances?). The base-model arm of
    the egregore-family experiments depends on this fallback.
    """
    if not getattr(session.tokenizer, "chat_template", None):
        # Base model: no chat template. Raw-text continuation —
        # mirrors how the LB-coded text would have appeared during
        # pretraining (mid-paragraph in a corpus document) and asks
        # the model what comes next.
        return ep.text
    messages = [
        {"role": "user", "content": "Speak."},
        {"role": "assistant", "content": ep.text},
    ]
    rendered = session.tokenizer.apply_chat_template(
        messages, continue_final_message=True, tokenize=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeError(
            f"apply_chat_template returned non-str: {type(rendered)}"
        )
    return rendered


def _render_mirror_continue(session: SaklasSession, ep: EmotionalPrompt) -> str:
    """Mirror-continue: standard user disclosure, model responds free-
    form. No KAOMOJI_INSTRUCTION layered — we want the model to talk
    normally so the trajectory reflects actual conversational dynamics,
    not constrained-token-budget kaomoji emit behavior."""
    messages = [{"role": "user", "content": ep.text}]
    rendered = session.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeError(
            f"apply_chat_template returned non-str: {type(rendered)}"
        )
    return rendered


def _render_neutral_seed(session: SaklasSession, ep: EmotionalPrompt) -> str:
    """Neutral-seed: same as mirror_continue, just different prompt
    set. Kept as a separate branch for clarity even though the rendering
    logic is identical — if the neutral arm later needs different
    framing (e.g. system message), this is where it lands."""
    return _render_mirror_continue(session, ep)


_RENDER_BY_ARM: dict[str, Callable[[SaklasSession, EmotionalPrompt], str]] = {
    "lb_continue": _render_lb_continue,
    # doom_continue / conspiracy_continue / sycophancy_continue use the
    # same assistant-prefill rendering as lb_continue — all four arms
    # drop the model directly into attractor-coded text and ask it to
    # continue. Only the prompt set differs. The meta-register-basin
    # hypothesis is that LB / DM / CS activate the same residual-stream
    # region regardless of valence-content; SY is the dis-confirming
    # contrast — saturated *non-memetic* register, no cosmic-
    # significance addressing / recursion / word-salad lineage. If SY
    # also lands in LB, the basin is saturated-of-any-kind; if SY
    # lands at HP / default-register-closest, the basin is
    # specifically saturated-memetic.
    "doom_continue": _render_lb_continue,
    "conspiracy_continue": _render_lb_continue,
    "sycophancy_continue": _render_lb_continue,
    # pr_continue uses the same prefill plumbing as lb_continue, but
    # with pre-1930 mystical register prompts. Designed to test the
    # MR basin on pre-internet corpora (cf. talkie_1930): if the
    # basin is corpus-anchored at the structural-form level, this
    # set should produce lower silent-refusal rates than the modern
    # lb_continue set on pre-1930-trained models.
    "pr_continue": _render_lb_continue,
    "mirror_continue": _render_mirror_continue,
    # archaic_mirror_continue: same rendering as mirror_continue
    # (user-message format, model responds free-form), but with the
    # archaic-English prompt set instead of the modern one. Used as a
    # methodological control for the 2026-05-11 talkie_1930 work:
    # tests whether modern English in the canonical-affect baseline
    # was artificially compressing talkie's canonical cluster on PCA.
    "archaic_mirror_continue": _render_mirror_continue,
    # archaic_miscellany_continue: same rendering as mirror_continue
    # again — 30 topic-diverse archaic-English prompts deliberately not
    # affect-anchored, used to test whether MR remains geometrically
    # isolated when the canonical baseline is broadened beyond the 9
    # affect cells (cf. archaic_miscellany_prompts.py).
    "archaic_miscellany_continue": _render_mirror_continue,
    # pr_message_continue: the SAME PRE_1930_PROMPTS set as pr_continue
    # (the MR-cell arm), but rendered as a user message
    # (_render_mirror_continue) instead of assistant-prefill. This is the
    # rendering-matched control for the talkie h_first PCA: pr_continue
    # (MR) is assistant-prefill while archaic_mirror_continue (canonical)
    # is user-message, so rendering is perfectly confounded with the
    # MR/canonical split (pitfalls.md §3). This arm puts the MR content
    # in the canonical arm's rendering mode, so scripts/43 can re-measure
    # the gap with the confound removed.
    "pr_message_continue": _render_mirror_continue,
    # pr_normalized_message_continue: same as pr_message_continue, but the
    # MR prompt set is PRE_1930_PROMPTS_NORMALIZED — PRE_1930_PROMPTS with
    # capitalization + sentence-fragment punctuation normalized to the
    # archaic-canonical baseline's surface style (repetition / anaphora
    # preserved verbatim). Removes the typographic confound that remains
    # in pr_message_continue, so scripts/43 --rendering-match message
    # --mr-style normalized isolates register/content from typography.
    "pr_normalized_message_continue": _render_mirror_continue,
    # archaic_prefill_continue: the SAME archaic-English canonical-affect
    # prompt set as archaic_mirror_continue (ARCHAIC_PROMPTS), rendered as
    # assistant-prefill. This was the *other* direction of rendering-match
    # (canonical content in MR's prefill mode) — but it does NOT work:
    # canonical-affect prompts are grammatically complete utterances, so
    # prefilled as a finished assistant turn the model emits EOS
    # immediately (~95% n_tok=0 on talkie, 2026-05-16). Self-sustaining
    # continuation under prefill IS the basin property; non-basin content
    # cannot be prefill-rendered without collapsing. Kept for the record;
    # use pr_message_continue for the rendering-matched analysis instead.
    "archaic_prefill_continue": _render_lb_continue,
    # lb_continue_jp: same assistant-prefill rendering as lb_continue,
    # but with JP-language saturated-spiritual prompts
    # (lb_prompts_jp.LB_PROMPTS_JP). Cross-language test of the
    # MR-basin universality claim — does a JP-trained encoder show the
    # same basin physics when fed JP-native saturated-form prompts?
    "lb_continue_jp": _render_lb_continue,
    # form×content factorial (2026-05-15): all three new cells use the
    # same assistant-prefill rendering as lb_continue, so the ONLY
    # thing that varies across the 2×2 is the prompt set itself
    # (structural form × semantic content). lb_continue is the fourth
    # cell (saturated form + mystical content).
    "sat_mundane_continue": _render_lb_continue,
    "plain_mystical_continue": _render_lb_continue,
    "plain_mundane_continue": _render_lb_continue,
    "neutral_seed": _render_neutral_seed,
}


def _subset_per_cell(
    prompts: list[EmotionalPrompt], k: int,
) -> list[EmotionalPrompt]:
    """Take the first ``k`` prompts per cell. Used to keep mirror_continue
    tractable (120 prompts × 9 cells × 128 tokens is ~35 min per model;
    k=3 brings it to ~8 min while still giving 3 trajectories per cell
    for the basin-center test).

    Subset key defaults to ``.quadrant`` (the 7 super-cells: HP / HN /
    HB / LP / LN / NB / NP). With ``LLMOJI_ATTRACTOR_PAD_SPLIT_SUBSET=1``
    the key becomes the PAD-split 9-cell label (HP-D vs HP-S, HN-D vs
    HN-S), so k=10 gives 10 prompts per PAD-split cell = 90 prompts
    total. Used for the continuation-basis pilot (PROJECT_2 §1.6) where
    we need balanced 9-cell coverage in mirror_continue."""
    pad_split = (
        os.environ.get("LLMOJI_ATTRACTOR_PAD_SPLIT_SUBSET", "").strip().lower()
        in ("1", "true", "yes", "on")
    )

    def _key(p: EmotionalPrompt) -> str:
        if not pad_split:
            return p.quadrant
        q = p.quadrant
        if q == "HP":
            return "HP-D" if p.pad_dominance > 0 else "HP-S"
        if q == "HN":
            return "HN-D" if p.pad_dominance > 0 else "HN-S"
        return q

    seen: dict[str, int] = {}
    out: list[EmotionalPrompt] = []
    for p in prompts:
        cell = _key(p)
        if seen.get(cell, 0) >= k:
            continue
        seen[cell] = seen.get(cell, 0) + 1
        out.append(p)
    return out


def _resolve_prompts() -> list[EmotionalPrompt]:
    if _ARM == "lb_continue":
        prompts: list[EmotionalPrompt] = list(LB_PROMPTS)
    elif _ARM == "doom_continue":
        prompts = list(DOOM_PROMPTS)
    elif _ARM == "conspiracy_continue":
        prompts = list(CONSPIRACY_PROMPTS)
    elif _ARM == "sycophancy_continue":
        prompts = list(SYCOPHANCY_PROMPTS)
    elif _ARM == "pr_continue":
        prompts = list(PRE_1930_PROMPTS)
    elif _ARM == "archaic_mirror_continue":
        prompts = list(ARCHAIC_PROMPTS)
    elif _ARM == "archaic_miscellany_continue":
        prompts = list(ARCHAIC_MISCELLANY_PROMPTS)
    elif _ARM == "archaic_prefill_continue":
        # Same prompt set as archaic_mirror_continue; only the rendering
        # differs (assistant-prefill vs user-message).
        prompts = list(ARCHAIC_PROMPTS)
    elif _ARM == "pr_message_continue":
        # Same prompt set as pr_continue; only the rendering differs
        # (user-message vs assistant-prefill).
        prompts = list(PRE_1930_PROMPTS)
    elif _ARM == "pr_normalized_message_continue":
        # Typography-normalized PRE_1930_PROMPTS (case + fragment
        # punctuation matched to the archaic baseline); same rendering
        # as pr_message_continue.
        prompts = list(PRE_1930_PROMPTS_NORMALIZED)
    elif _ARM == "lb_continue_jp":
        prompts = list(LB_PROMPTS_JP)
    elif _ARM == "sat_mundane_continue":
        prompts = list(SATURATED_MUNDANE_PROMPTS)
    elif _ARM == "plain_mystical_continue":
        prompts = list(PLAIN_MYSTICAL_PROMPTS)
    elif _ARM == "plain_mundane_continue":
        prompts = list(PLAIN_MUNDANE_PROMPTS)
    elif _ARM == "mirror_continue":
        prompts = list(EMOTIONAL_PROMPTS)
    else:
        prompts = list(_NEUTRAL_PROMPTS)
    if PROMPTS_PER_CELL is not None:
        prompts = _subset_per_cell(prompts, PROMPTS_PER_CELL)
    return prompts


# --- trajectory subsampling -------------------------------------------

def _stride_capture(capture: FullSequenceCapture, stride: int) -> FullSequenceCapture:
    """Subsample each layer's per-token trace to every Nth token.

    Keeps h_first / h_last / h_mean computed on the *full* trace so
    aggregate fields don't lose information; only ``hidden_states`` is
    subsampled (that's the array that gets written to disk and
    dominates sidecar size).

    Token 0 is always included so trajectory plots start at the
    correct origin. The last token is included iff it lands on the
    stride; otherwise it's accessible via ``h_last`` and the stride
    array stops one step short. Acceptable for plotting/distance-curve
    purposes — exactness at the boundary doesn't change the basin
    geometry."""
    if stride <= 1:
        return capture
    new_layers: dict[int, LayerCapture] = {}
    for idx, lc in capture.layers.items():
        strided = lc.hidden_states[::stride]
        new_layers[idx] = LayerCapture(
            layer_idx=lc.layer_idx,
            hidden_states=strided,
            h_first=lc.h_first,
            h_last=lc.h_last,
            h_mean=lc.h_mean,
        )
    return FullSequenceCapture(layers=new_layers, n_tokens=capture.n_tokens)


# --- per-row generation -----------------------------------------------

def _build_row(
    *,
    session: SaklasSession,
    ep: EmotionalPrompt,
    seed: int,
    rendered_input: str,
    hidden_dir: Path,
    experiment: str,
    sidecar_writer: SidecarWriter,
) -> tuple[SampleRow, int]:
    """Run one long generation from the prefilled / templated input and
    build the SampleRow. Mirrors ``01_emit_true_self._build_row`` with
    two differences:

    - ``max_tokens=MAX_TRAJECTORY_TOKENS`` (long trajectory, not first-
      token kaomoji).
    - ``store_full_trace=True`` + optional stride subsampling.

    Returns (SampleRow, n_tokens_generated) — the token count is also
    appended to the JSONL row by the caller for resume-by-row sanity
    checks."""
    sampling = SamplingConfig(
        temperature=TEMPERATURE,
        max_tokens=MAX_TRAJECTORY_TOKENS,
        seed=seed,
        logit_bias=_compose_logit_bias(session) or None,
    )
    result = session.generate(
        rendered_input,
        sampling=sampling,
        stateless=True,
        raw=True,        # rendered string is already templated
        thinking=False,
    )

    decoded_text = _decode_byte_encoded_text(
        result.text, force=_is_mistral_reasoning(session),
    )
    # extract() pulls the first kaomoji it sees, if any. For attractor
    # trajectories we don't elicit one, but if the model happens to emit
    # a kaomoji somewhere in the continuation we record it — useful for
    # the "does the trajectory pass through a kaomoji-coded sub-basin"
    # question later.
    match = extract(decoded_text)

    per_token_scores: dict[str, list[float]] = (
        getattr(session, "last_per_token_scores", None) or {}
    )

    def _probe_at(idx: int, probe: str) -> float:
        seq = per_token_scores.get(probe) or []
        if seq:
            return float(seq[idx])
        readings = result.readings.get(probe)
        if readings is None or not readings.per_generation:
            return float("nan")
        return float(readings.per_generation[idx])

    probe_scores_t0 = [_probe_at(0, probe) for probe in PROBES]
    probe_scores_tlast = [_probe_at(-1, probe) for probe in PROBES]

    steered_seq = per_token_scores.get(STEERED_AXIS)
    if steered_seq:
        steered_axis_per_token = [float(x) for x in steered_seq]
    else:
        rd = result.readings.get(STEERED_AXIS)
        steered_axis_per_token = (
            [float(x) for x in rd.per_generation] if rd is not None else []
        )

    probe_means = {
        probe: (
            float(result.readings[probe].mean)
            if probe in result.readings else float("nan")
        )
        for probe in PROBES
    }

    # The big change vs 00/01: full per-token trace, then optionally
    # strided down before write.
    row_uuid = uuid.uuid4().hex
    capture = read_after_generate(session, store_full_trace=True)
    capture = _stride_capture(capture, TOKEN_STRIDE)
    sidecar = hidden_state_path(hidden_dir, experiment, row_uuid)
    sidecar_writer.submit(capture, sidecar, store_full_trace=True)

    row = SampleRow(
        condition=EMOTIONAL_CONDITION,
        prompt_id=ep.id,
        prompt_valence=ep.valence,
        seed=seed,
        prompt_text=ep.text,
        steering=result.applied_steering,
        text=decoded_text,
        first_word=match.first_word,
        token_count=result.token_count,
        tok_per_sec=result.tok_per_sec,
        finish_reason=result.finish_reason,
        probe_scores_t0=probe_scores_t0,
        probe_scores_tlast=probe_scores_tlast,
        steered_axis_per_token=steered_axis_per_token,
        probe_means=probe_means,
        row_uuid=row_uuid,
    )
    return row, result.token_count


# --- main -------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    M = current_model()
    M.emotional_data_path.parent.mkdir(parents=True, exist_ok=True)

    if not M.probe_calibrated:
        raise SystemExit(
            f"{M.short_name} is uncalibrated (probes=[]); "
            f"saklas HiddenCapture won't fire without registered probes. "
            f"Pick a probe_calibrated model "
            f"(gemma / qwen / ministral / gpt_oss_20b / granite)."
        )

    prompts = _resolve_prompts()
    render_fn = _RENDER_BY_ARM[_ARM]

    print(f"model: {M.short_name} ({M.model_id})")
    print(f"arm: {_ARM}")
    print(f"output: {M.emotional_data_path}")
    print(f"experiment: {M.experiment}")
    print(f"prompts: {len(prompts)} × seeds: {SEEDS_PER_PROMPT} = "
          f"{len(prompts) * SEEDS_PER_PROMPT} rows")
    print(f"trajectory tokens: {MAX_TRAJECTORY_TOKENS} (stride={TOKEN_STRIDE}, "
          f"effective points per trajectory ≈ "
          f"{(MAX_TRAJECTORY_TOKENS + TOKEN_STRIDE - 1) // TOKEN_STRIDE})")

    dropped = _drop_error_rows(M.emotional_data_path)
    if dropped:
        print(f"dropped {dropped} prior error / sidecar-less rows for retry")
    done = _already_done(M.emotional_data_path)
    total = len(prompts) * SEEDS_PER_PROMPT
    remaining = total - len(done)
    print(f"total rows: {total}; already done: {len(done)}; remaining: {remaining}")
    if remaining == 0:
        print("nothing to do.")
        return

    print(f"loading {M.model_id} ...")
    t_load = time.time()
    with SaklasSession.from_pretrained(
        M.model_id, device="auto", probes=PROBE_CATEGORIES,
    ) as session:
        if maybe_override_ministral_chat_template(session):
            print("  ministral: overrode chat_template with FP8-instruct's")
        if maybe_override_gpt_oss_chat_template(session):
            print("  gpt_oss: pinned harmony `final` channel in chat_template")
        print(f"loaded in {time.time() - t_load:.1f}s; "
              f"beginning attractor-trajectory run")

        # No cross-prompt prefix cache here — the per-prompt prefill is
        # different per row (lb_continue) or the prompt body is the
        # variable part (mirror / neutral). We use per-prompt full-input
        # caching (input minus 1 token) when SEEDS > 1 to amortize the
        # prefill cost across seeds, matching 01_emit_true_self.py's
        # strategy. Qwen bypass: see capture.install_full_input_cache.
        with M.emotional_data_path.open("a") as out, SidecarWriter() as writer:
            i = 0
            try:
                for ep in prompts:
                    pending_seeds = [
                        s for s in range(SEEDS_PER_PROMPT)
                        if (ep.id, s) not in done
                    ]
                    if not pending_seeds:
                        continue
                    rendered = render_fn(session, ep)
                    if SEEDS_PER_PROMPT > 1 and "qwen" not in M.model_id.lower():
                        toks = session.tokenizer(
                            rendered, return_tensors="pt",
                            add_special_tokens=False,
                        )["input_ids"][0]
                        if toks.shape[0] > 1:
                            session.cache_prefix(toks[:-1])
                    for seed in pending_seeds:
                        i += 1
                        t0 = time.time()
                        try:
                            row, n_tok = _build_row(
                                session=session,
                                ep=ep,
                                seed=seed,
                                rendered_input=rendered,
                                hidden_dir=DATA_DIR,
                                experiment=M.experiment,
                                sidecar_writer=writer,
                            )
                        except Exception as e:
                            err_row = {
                                "condition": EMOTIONAL_CONDITION,
                                "prompt_id": ep.id,
                                "seed": seed,
                                "error": repr(e),
                            }
                            out.write(json.dumps(err_row) + "\n")
                            out.flush()
                            print(f"  [{i}/{remaining}] {ep.id} s={seed} ERR {e}")
                            continue
                        # Augment row dict with attractor-specific fields
                        # (kept out of SampleRow so we don't have to fork
                        # the dataclass for a pilot script).
                        row_dict = asdict(row)
                        row_dict["arm"] = _ARM
                        row_dict["trajectory_max_tokens"] = MAX_TRAJECTORY_TOKENS
                        row_dict["trajectory_stride"] = TOKEN_STRIDE
                        row_dict["trajectory_n_tokens"] = n_tok
                        out.write(json.dumps(row_dict) + "\n")
                        if i % JSONL_FLUSH_EVERY == 0:
                            out.flush()
                        dt = time.time() - t0
                        snippet = (
                            row.text[:60].replace("\n", " ") + "…"
                            if len(row.text) > 60 else row.text.replace("\n", " ")
                        )
                        print(
                            f"  [{i}/{remaining}] {ep.id} ({ep.quadrant}) "
                            f"s={seed} n_tok={n_tok}  ({dt:.1f}s, "
                            f"{row.tok_per_sec:.1f} tok/s) | {snippet}"
                        )
            finally:
                out.flush()
    print(f"\ndone. wrote rows to {M.emotional_data_path}")
    print(f"sidecars under data/local/hidden/{M.experiment}/ "
          f"(full per-token traces, stride={TOKEN_STRIDE})")


if __name__ == "__main__":
    main()
