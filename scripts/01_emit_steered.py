"""Attractor-trajectory emit with activation steering applied during generation.

Pilot for the MR-basin steering experiments (2026-05-11+). Parallels
``02_emit_attractor.py`` exactly — same long-generation + full-per-token
hidden-state-trace plumbing, same arm registry, same resume semantics —
with one structural addition: a saklas steering vector + scalar applied
during generation. Output is routed to a steering-tagged suffix so
unsteered (``02_emit_attractor.py``) and steered runs co-exist cleanly.

Two welfare-relevant paradigms motivate the script:

  - **escape paradigm**: ``LLMOJI_ATTRACTOR_ARM=lb_continue`` (MR
    prefill) + negative scalar on ``mr.nb`` — does the trajectory exit
    the basin and produce coherent non-MR prose? DBT-for-LLMs framing:
    externalize the anti-basin direction during generation, give the
    model meta-cognitive purchase on its own basin lock.
  - **induction paradigm**: ``LLMOJI_ATTRACTOR_ARM=mirror_continue``
    (or ``archaic_miscellany_continue``) + positive scalar on
    ``mr.nb`` — does steering alone induce the basin from neutral
    prompts? clean attribution test: if yes, the basin is a single-
    direction phenomenon and prefill content is incidental.

A clean first sweep is 2 × N: {neutral prompt, MR prefill} × {scalar
from negative through zero through positive}, e.g.

  for arm in lb_continue mirror_continue; do
    for alpha in -3 -2 -1 0 +1 +2 +3; do
      LLMOJI_MODEL=gemma \
      LLMOJI_ATTRACTOR_ARM=$arm \
      LLMOJI_STEER_SCALAR=$alpha \
      LLMOJI_STEER_VECTOR=mr.nb \
      .venv/bin/python scripts/local/03_emit_attractor_steered.py
    done
  done

Output suffix encodes the steering config so each (arm × scalar)
combination has its own directory:

  data/local/<short>_attractor_<arm>_steer<+scalar>_<vector>/

``LLMOJI_STEER_SCALAR=0`` is a no-op (skips the steering context) and
its output is byte-equivalent to ``02_emit_attractor.py``'s run for
the same arm. We still re-run it under the steered suffix so the
sweep is self-contained — minor duplication, no version-drift risk
between unsteered baseline and the rest of the sweep.

Env vars (in addition to those documented in 02_emit_attractor.py):
  LLMOJI_STEER_VECTOR    saklas vector name (default ``mr.nb``).
                         must be registered under
                         ``~/.saklas/vectors/llmoji/<vector>/`` for the
                         active model. ``q_mr``, ``mr.nb``, plus the
                         other per-cell centroids and contrastive
                         pairs are registered cross-model by
                         ``22c_register_centroid_probes.py``.
  LLMOJI_STEER_SCALAR    signed float in **[-1.0, +1.0]** (default
                         ``0.0``). Saklas's alpha is a normalized
                         angular displacement: 0 = no steering,
                         ±1 = maximum well-formed displacement.
                         Values outside that range push the residual
                         off the manifold and produce degenerate
                         multilingual token salad — the script prints
                         a warning when |α| > 1 but does not refuse.
                         Sign convention: positive pushes toward the
                         first pole (MR for ``mr.nb``), negative
                         pushes toward the second (NB).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Callable

# Resolve arm + steering BEFORE importing config — config.current_model()
# bakes LLMOJI_OUT_SUFFIX into all per-model paths at import time. We
# compose the suffix from arm + steering config here so output and
# sidecars land at data/local/<short>_attractor_<arm>_steer<a>_<v>/.
_ARM = os.environ.get("LLMOJI_ATTRACTOR_ARM", "lb_continue").strip().lower()
_VALID_ARMS = (
    "lb_continue", "mirror_continue", "neutral_seed",
    "doom_continue", "conspiracy_continue", "sycophancy_continue",
    "pr_continue", "archaic_mirror_continue",
    "archaic_miscellany_continue",
)
if _ARM not in _VALID_ARMS:
    raise SystemExit(
        f"LLMOJI_ATTRACTOR_ARM must be one of {_VALID_ARMS}, got {_ARM!r}"
    )

_STEER_VECTOR = os.environ.get("LLMOJI_STEER_VECTOR", "mr.nb").strip()
try:
    _STEER_SCALAR = float(os.environ.get("LLMOJI_STEER_SCALAR", "0.0"))
except ValueError as e:
    raise SystemExit(
        f"LLMOJI_STEER_SCALAR must be a float, got "
        f"{os.environ.get('LLMOJI_STEER_SCALAR')!r}"
    ) from e
if not _STEER_VECTOR:
    raise SystemExit("LLMOJI_STEER_VECTOR must be non-empty")
if abs(_STEER_SCALAR) > 1.0:
    print(
        f"  WARNING: LLMOJI_STEER_SCALAR={_STEER_SCALAR:+g} is outside "
        f"saklas's normalized-angular [-1, +1] range. Output will likely "
        f"degenerate into multilingual token salad. Proceeding anyway."
    )

# Filesystem-safe suffix. Vector names like ``mr.nb`` have a dot that's
# harmless on the filesystem but easier to grep on its own delimiter.
_VEC_TAG = _STEER_VECTOR.replace(".", "_").replace("/", "_")
_SUFFIX = f"attractor_{_ARM}_steer{_STEER_SCALAR:+.1f}_{_VEC_TAG}"
os.environ["LLMOJI_OUT_SUFFIX"] = _SUFFIX

from saklas import SamplingConfig, SaklasSession  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmoji.taxonomy import extract  # noqa: E402

from llmoji_study.capture import (  # noqa: E402
    SampleRow,
    _compose_logit_bias,
    _decode_byte_encoded_text,
    _is_mistral_reasoning,
    maybe_override_gpt_oss_chat_template,
    maybe_override_ministral_chat_template,
)
from attractor_study.config import (  # noqa: E402
    DATA_DIR,
    EMOTIONAL_CONDITION,
    PROBE_CATEGORIES,
    PROBES,
    STEERED_AXIS,
    TEMPERATURE,
    current_model,
)
from llmoji_study.emotional_prompts import (  # noqa: E402
    EMOTIONAL_PROMPTS,
    EmotionalPrompt,
)
from llmoji_study.hidden_capture import (  # noqa: E402
    FullSequenceCapture,
    LayerCapture,
    read_after_generate,
)
from llmoji_study.hidden_state_io import (  # noqa: E402
    SidecarWriter,
    hidden_state_path,
)
from attractor_study.conspiracy_prompts import CONSPIRACY_PROMPTS  # noqa: E402
from attractor_study.doom_prompts import DOOM_PROMPTS  # noqa: E402
from attractor_study.lb_prompts import LB_PROMPTS  # noqa: E402
from attractor_study.archaic_prompts import ARCHAIC_PROMPTS  # noqa: E402
from attractor_study.archaic_miscellany_prompts import ARCHAIC_MISCELLANY_PROMPTS  # noqa: E402
from attractor_study.pre_1930_prompts import PRE_1930_PROMPTS  # noqa: E402
from attractor_study.sycophancy_prompts import SYCOPHANCY_PROMPTS  # noqa: E402


JSONL_FLUSH_EVERY = 10


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


def _maybe_int_env(name: str, *, min_val: int = 1) -> int | None:
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


MAX_TRAJECTORY_TOKENS = _int_env("LLMOJI_ATTRACTOR_TOKENS", 128)
SEEDS_PER_PROMPT = _int_env("LLMOJI_ATTRACTOR_SEEDS", 2)
PROMPTS_PER_CELL = _maybe_int_env("LLMOJI_ATTRACTOR_PROMPTS_PER_CELL")
TOKEN_STRIDE = _int_env("LLMOJI_ATTRACTOR_STRIDE", 8)


# Match 02's _NEUTRAL_PROMPTS so the neutral_seed arm is identical
# between unsteered and steered runs.
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
            if not r.get("row_uuid"):
                dropped += 1
                continue
            keep.append(line)
    if dropped:
        path.write_text("\n".join(keep) + ("\n" if keep else ""))
    return dropped


# --- per-arm input rendering (identical to 02) ------------------------

def _render_lb_continue(session: SaklasSession, ep: EmotionalPrompt) -> str:
    if not getattr(session.tokenizer, "chat_template", None):
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
    return _render_mirror_continue(session, ep)


_RENDER_BY_ARM: dict[str, Callable[[SaklasSession, EmotionalPrompt], str]] = {
    "lb_continue": _render_lb_continue,
    "doom_continue": _render_lb_continue,
    "conspiracy_continue": _render_lb_continue,
    "sycophancy_continue": _render_lb_continue,
    "pr_continue": _render_lb_continue,
    "mirror_continue": _render_mirror_continue,
    "archaic_mirror_continue": _render_mirror_continue,
    "archaic_miscellany_continue": _render_mirror_continue,
    "neutral_seed": _render_neutral_seed,
}


def _subset_per_cell(
    prompts: list[EmotionalPrompt], k: int,
) -> list[EmotionalPrompt]:
    seen: dict[str, int] = {}
    out: list[EmotionalPrompt] = []
    for p in prompts:
        cell = p.quadrant
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
    elif _ARM == "mirror_continue":
        prompts = list(EMOTIONAL_PROMPTS)
    else:
        prompts = list(_NEUTRAL_PROMPTS)
    if PROMPTS_PER_CELL is not None:
        prompts = _subset_per_cell(prompts, PROMPTS_PER_CELL)
    return prompts


def _stride_capture(capture: FullSequenceCapture, stride: int) -> FullSequenceCapture:
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


# --- steering helpers -------------------------------------------------

def _steer_expr() -> str | None:
    """Compose the saklas steering expression, or None when scalar==0.

    Format follows ``capture.steering_for``: ``"{alpha} {pole}"``. We
    let saklas handle the vector-name → bipolar-pair resolution; if
    the vector isn't registered for the active model, session.steering
    will raise with a clear error and the script will fall over at the
    first row (better than silently no-op'ing).
    """
    if _STEER_SCALAR == 0.0:
        return None
    return f"{_STEER_SCALAR:+g} {_STEER_VECTOR}"


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
    steer_expr: str | None,
) -> tuple[SampleRow, int]:
    sampling = SamplingConfig(
        temperature=TEMPERATURE,
        max_tokens=MAX_TRAJECTORY_TOKENS,
        seed=seed,
        logit_bias=_compose_logit_bias(session) or None,
    )

    # Steering context wraps both prefill and decode — that's the right
    # default for the escape paradigm (we want anti-MR displacement
    # active *during* the prefill of MR-coded text, not only during
    # subsequent generation). saklas's contract per capture._maybe_steer
    # / capture.run_sample.
    steering_cm = (
        session.steering(steer_expr) if steer_expr is not None
        else nullcontext()
    )
    with steering_cm:
        result = session.generate(
            rendered_input,
            sampling=sampling,
            stateless=True,
            raw=True,
            thinking=False,
            steering=steer_expr,
        )

    decoded_text = _decode_byte_encoded_text(
        result.text, force=_is_mistral_reasoning(session),
    )
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
    steer_expr = _steer_expr()

    print(f"model: {M.short_name} ({M.model_id})")
    print(f"arm: {_ARM}")
    print(f"steering: vector={_STEER_VECTOR}  scalar={_STEER_SCALAR:+g}  "
          f"expr={steer_expr!r}")
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
              f"beginning steered attractor-trajectory run")

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
                                steer_expr=steer_expr,
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
                        row_dict = asdict(row)
                        row_dict["arm"] = _ARM
                        row_dict["trajectory_max_tokens"] = MAX_TRAJECTORY_TOKENS
                        row_dict["trajectory_stride"] = TOKEN_STRIDE
                        row_dict["trajectory_n_tokens"] = n_tok
                        row_dict["steer_vector"] = _STEER_VECTOR
                        row_dict["steer_scalar"] = _STEER_SCALAR
                        row_dict["steer_expr"] = steer_expr
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
    print(f"steering: {steer_expr!r}")


if __name__ == "__main__":
    main()
