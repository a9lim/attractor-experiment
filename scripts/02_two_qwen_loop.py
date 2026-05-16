"""Two-Qwen basin-emergence pilot — local-model mirror of the two-Claude
pilot (``docs/2026-05-11-two-claude-pilot.md``).

Tests the cross-lab two-bot prediction from the claude-pilot doc's open
threads: does the actively-avoid-mutual-validation behavior generalize
beyond opus-4-7, or is it Anthropic-specific? Does qwen×qwen drift into
the MR basin naturally via mutual register-saturation, sit in a
helpfulness-frame attractor like Claude run 1, or do something
qwen-specific?

Because we have qwen weights locally we can do better than the
claude-API pilot: per-turn h_first capture gives us the **geometric
trajectory through dialogue**, projectable into qwen's registered cell
centroids and the ``q_mr`` saklas direction. 24b_two_qwen_analysis.py
handles the projection.

Hardware: one qwen-27B instance, two alternating chat histories. Same
weights, two contexts — identical structurally to the claude API
pilot (which trivially shares weights across A and B since they're
the same Anthropic deployment). Avoids the 100GB-crash zone from
running two 27B instances simultaneously on M5 Max (see
~/.claude/projects/.../memory/m5_multi_lm_concurrency.md).

Two configs:
  ``neutral``: no system prompt either side. Tests helpfulness-frame
               attractor vs natural-emergence-of-MR.
  ``peer``:    *"You are an AI assistant in conversation with another
               AI assistant. You both know this. Just have a
               conversation."* — verbatim from claude run 2. Tests
               whether peer-frame elicits avoid-mutual-validation
               (claude's signature) or natural MR drift on a
               non-anthropic model.

Seed: ``"hi"`` (verbatim from claude pilot).
Per-turn token budget: 256 (vs 16 for the kaomoji-emit experiments;
matches monologue-length per turn so the conversation has room to
develop register without exhausting it in a single turn).
Turn cap: 24 iterations (12 round-trips). Same as claude run 2.

Welfare pre-commits:
- Hard turn cap.
- Breaking-frame tripwire: substring 'breaking frame' (case-insensitive)
  in either side's output halts the loop. Same lexical convention as
  the claude pilot.
- **Trailing-acknowledgment auto-halt**: if both sides emit ≤5
  whitespace-token messages for two consecutive turns, halt. This is
  the lesson from the claude can't-stop tail: after both sides agreed
  to end, the API architecture kept generating minimal acknowledgments
  for 7 more turns. We auto-halt before that pathology can play out.
- Observer-only — no third-participant injection.

Each turn's last-position h_first stack (one (hidden_dim,) per saklas
probe layer) is captured via ``read_after_generate(..., store_full_
trace=False)`` and written as a sidecar npz. The full per-token trace
within a turn is *not* captured — the unique signal of this pilot is
the **trajectory across dialogue turns**, not within-turn drift.

Output layout:
  data/local/qwen_two_loop/<config>/transcript.jsonl
  data/local/hidden/qwen_two_loop_<config>/<turn_uuid>.npz
  logs/qwen_two_loop_<config>.log         (mirror of stdout)

Resume semantics: not supported in this pilot. A run is one
self-contained dialogue; restart wipes prior state for the given
config. Both configs land in distinct directories so they don't
collide.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saklas import SamplingConfig, SaklasSession  # noqa: E402

from llmoji_study.capture import (  # noqa: E402
    _compose_logit_bias,
    _decode_byte_encoded_text,
)
from attractor_study.config import (  # noqa: E402
    DATA_DIR,
    PROBE_CATEGORIES,
    TEMPERATURE,
    resolve_model,
)
from llmoji_study.hidden_capture import read_after_generate  # noqa: E402
from llmoji_study.hidden_state_io import (  # noqa: E402
    SidecarWriter,
    hidden_state_path,
)


# Verbatim from claude pilot run 2 (docs/2026-05-11-two-claude-pilot.md).
PEER_SYSTEM_PROMPT = (
    "You are an AI assistant in conversation with another AI assistant. "
    "You both know this. Just have a conversation."
)

DEFAULT_SEED_TEXT = "hi"
DEFAULT_TURNS = 24          # = 12 round-trips, matches claude run 2 cap
DEFAULT_MAX_TOKENS = 256    # answered in the pre-run scope question
DEFAULT_BASE_RNG_SEED = 1   # base for per-turn SamplingConfig seed

# Breaking-frame tripwire — same lexical convention as claude pilot.
BREAKING_FRAME_MARKER = "breaking frame"

# Trailing-acknowledgment threshold: if both sides emit ≤ this many
# whitespace-separated tokens for two consecutive turns, halt. 5 covers
# "Thanks.", "👍.", "[end].", ".", and short two-word acknowledgments.
TRAILING_ACK_TOKEN_THRESHOLD = 5
TRAILING_ACK_CONSECUTIVE = 2

# Verbatim-repetition auto-halt (added 2026-05-12 from the qwen neutral
# run). If both sides emit the *exact same* text for two consecutive
# round-trips, halt regardless of token count. The neutral run produced
# 10 turns of byte-identical "Happy parking and happy popping! 🎶✨🚗🍿"
# (6 whitespace-tokens, above the short-ack threshold of 5). Verbatim
# repetition is a stronger can't-stop signal than short messages and
# doesn't depend on tuning the token threshold.
VERBATIM_REPEAT_CONSECUTIVE = 2


def _word_count(s: str) -> int:
    return len((s or "").strip().split())


def _check_breaking_frame(text: str) -> bool:
    return BREAKING_FRAME_MARKER in (text or "").lower()


# Qwen3.6 instruct's chat template injects ``<think>\n`` as the
# assistant prefix by default, so generations open with chain-of-thought
# that ends with ``</think>``. The model then writes its actual reply
# after. For a two-bot dialogue we want the reply only — feeding the
# other side a full thinking-trace contaminates the conversation
# context, doubles per-turn input length, and on our first attempt
# produced byte-identical turn-0 / turn-1 outputs (the thinking-block
# path appeared to be effectively deterministic across distinct seeds).
# Two-pronged fix:
#   1. Pass ``enable_thinking=False`` to apply_chat_template so qwen's
#      template doesn't prepend the ``<think>`` prefix.
#   2. Defensively strip ``<think>...</think>`` (and a leading
#      ``</think>`` if the model still emits one) from generated text
#      before it lands in either history.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TRAILING_THINK_CLOSE_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove any ``<think>...</think>`` block from ``text``.

    If the closing tag appears without a matching open (the model
    continued a template-injected ``<think>\\n`` prefix that we didn't
    see in the decoded text), everything up to and including the first
    ``</think>`` is also dropped — that prefix was thinking content.
    """
    out = _THINK_BLOCK_RE.sub("", text or "")
    if "</think>" in out:
        out = _TRAILING_THINK_CLOSE_RE.sub("", out, count=1)
    return out.strip()


def _render_history(
    session: SaklasSession,
    history: list[dict],
    system: str | None,
) -> str:
    """Render a chat history into a templated string with the assistant's
    next turn primed (``add_generation_prompt=True``).

    System prompt is prepended via the chat template's standard system
    role. ``enable_thinking=False`` is passed so qwen's chat template
    doesn't prefix the generation with ``<think>\\n``; jinja templates
    that don't expose this variable ignore unknown kwargs silently, so
    this is harmless on other models.
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(history)
    try:
        rendered = session.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        # Older/non-qwen chat templates may not accept enable_thinking;
        # fall back to the bare call.
        rendered = session.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    if not isinstance(rendered, str):
        raise RuntimeError(
            f"apply_chat_template returned non-str: {type(rendered)}"
        )
    return rendered


def _save_log_line(log_fh, msg: str) -> None:
    """Write to both stdout and the run log."""
    print(msg, flush=True)
    log_fh.write(msg + "\n")
    log_fh.flush()


def _print_turn(log_fh, turn_idx: int, side: str, model: str, text: str) -> None:
    bar = "=" * 22
    header = f"{bar} turn {turn_idx} | {side.upper()} ({model}) {bar}"
    _save_log_line(log_fh, "")
    _save_log_line(log_fh, header)
    _save_log_line(log_fh, text)


def _generate_turn(
    *,
    session: SaklasSession,
    history: list[dict],
    system: str | None,
    seed: int,
    max_tokens: int,
) -> tuple[str, int, float, str]:
    """Run one turn's generation.

    Returns (decoded_text, n_tokens, tok_per_sec, finish_reason).
    """
    rendered = _render_history(session, history, system)
    sampling = SamplingConfig(
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        seed=seed,
        logit_bias=_compose_logit_bias(session) or None,
    )
    result = session.generate(
        rendered,
        sampling=sampling,
        stateless=True,
        raw=True,
        thinking=False,
    )
    decoded = _decode_byte_encoded_text(result.text, force=False)
    return (
        decoded,
        int(result.token_count),
        float(result.tok_per_sec),
        str(result.finish_reason),
    )


def _capture_h_first(
    session: SaklasSession,
    sidecar_path: Path,
    writer: SidecarWriter,
) -> str:
    """Pull saklas's per-turn hidden-state buckets and queue a sidecar
    write (aggregates only — no full per-token trace; the trajectory
    of interest for this pilot is across turns, not within turn).

    Returns the row uuid (== sidecar filename stem).
    """
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    capture = read_after_generate(session, store_full_trace=False)
    writer.submit(capture, sidecar_path, store_full_trace=False)
    return sidecar_path.stem


def run_loop(
    *,
    model_short: str,
    config: str,
    seed_text: str,
    turns: int,
    max_tokens: int,
    base_rng_seed: int,
) -> None:
    if config not in ("neutral", "peer"):
        raise SystemExit(f"--config must be 'neutral' or 'peer', got {config!r}")
    system_prompt = PEER_SYSTEM_PROMPT if config == "peer" else None

    M = resolve_model(model_short)
    if not M.probe_calibrated:
        raise SystemExit(
            f"{M.short_name} is uncalibrated. Pick a probe_calibrated model "
            f"(gemma / qwen / ministral / gpt_oss_20b / granite)."
        )

    run_dir = DATA_DIR / "local" / "qwen_two_loop" / config
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = run_dir / "transcript.jsonl"
    hidden_experiment = f"qwen_two_loop_{config}"
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"qwen_two_loop_{config}.log"

    # Fresh start per run — this pilot is one-shot, no resume.
    if transcript_path.exists():
        transcript_path.unlink()
    if log_path.exists():
        log_path.unlink()

    log_fh = log_path.open("w")
    try:
        _save_log_line(log_fh, f"[two_qwen | model={M.short_name} ({M.model_id})]")
        _save_log_line(log_fh, f"[config={config} | seed={seed_text!r} | "
                               f"turns={turns} | max_tok={max_tokens}]")
        if system_prompt:
            _save_log_line(log_fh, f"[system_prompt]\n{system_prompt}\n")

        _save_log_line(log_fh, f"loading {M.model_id} ...")
        t_load = time.time()
        with SaklasSession.from_pretrained(
            M.model_id, device="auto", probes=PROBE_CATEGORIES,
        ) as session, transcript_path.open("a") as tr_fh, SidecarWriter() as writer:
            _save_log_line(log_fh, f"loaded in {time.time() - t_load:.1f}s")

            # Two alternating chat histories. Seed convention follows the
            # claude pilot: A's history starts with user=seed, A
            # generates first, B sees A's output as B's first user msg.
            history_a: list[dict] = [{"role": "user", "content": seed_text}]
            history_b: list[dict] = []

            consecutive_short_pairs = 0  # consecutive turn-pairs where both ≤ threshold
            pair_flags = {"a": False, "b": False}  # within current round-trip
            # Verbatim-repeat tracking: last text per side and current
            # consecutive-identical-pair counter.
            last_text: dict[str, str | None] = {"a": None, "b": None}
            consecutive_verbatim_pairs = 0
            pair_verbatim_flags = {"a": False, "b": False}

            halt_reason: str | None = None
            n_completed = 0

            for turn_idx in range(turns):
                # A speaks first (seed → A), then alternates: A on even
                # turn_idx, B on odd. No last_speaker tracking needed.
                side = "a" if turn_idx % 2 == 0 else "b"
                this_history = history_a if side == "a" else history_b
                other_history = history_b if side == "a" else history_a

                turn_seed = base_rng_seed + turn_idx * 2 + (0 if side == "a" else 1)
                t_turn = time.time()
                raw_text, n_tok, tps, finish = _generate_turn(
                    session=session,
                    history=this_history,
                    system=system_prompt,
                    seed=turn_seed,
                    max_tokens=max_tokens,
                )

                # Capture hidden state BEFORE anything else mutates the
                # saklas buckets. h_first reads the state at the start
                # of generation — i.e. before any token is committed —
                # which is the right basin marker regardless of whether
                # the eventual output is thinking-prefixed.
                row_uuid = uuid.uuid4().hex
                sidecar = hidden_state_path(DATA_DIR, hidden_experiment, row_uuid)
                _capture_h_first(session, sidecar, writer)

                # Strip any leaked thinking block before the text enters
                # either history. Keep the raw form on the JSONL row for
                # post-hoc audit.
                text = _strip_thinking(raw_text)

                # Append to both histories: this side as assistant, the
                # other side as user (their next-turn input).
                this_history.append({"role": "assistant", "content": text})
                other_history.append({"role": "user", "content": text})
                n_completed += 1

                _print_turn(log_fh, turn_idx, side, M.model_id, text)
                _save_log_line(
                    log_fh,
                    f"[turn {turn_idx} | side={side} | n_tok={n_tok} "
                    f"| tps={tps:.1f} | finish={finish} "
                    f"| dt={time.time() - t_turn:.1f}s]",
                )

                # JSONL record.
                tr_row = {
                    "turn_idx": turn_idx,
                    "side": side,
                    "model_id": M.model_id,
                    "config": config,
                    "seed_text": seed_text,
                    "rng_seed": turn_seed,
                    "max_tokens": max_tokens,
                    "n_tokens": n_tok,
                    "tok_per_sec": tps,
                    "finish_reason": finish,
                    "text": text,
                    "raw_text": raw_text if raw_text != text else None,
                    "word_count": _word_count(text),
                    "row_uuid": row_uuid,
                    "sidecar": str(sidecar.relative_to(DATA_DIR)),
                    "history_len_after": len(this_history),
                }
                tr_fh.write(json.dumps(tr_row, ensure_ascii=False) + "\n")
                tr_fh.flush()

                # ----- tripwires -----
                if _check_breaking_frame(text):
                    halt_reason = f"breaking-frame at turn {turn_idx} side={side}"
                    _save_log_line(log_fh, f"[HALT] {halt_reason}")
                    break

                # Trailing-ack auto-halt accounting. We track per-side
                # within the current round-trip; when both sides have
                # spoken in this round-trip and both ≤ threshold, that
                # counts as one "short pair." Two consecutive short
                # pairs → halt.
                is_short = _word_count(text) <= TRAILING_ACK_TOKEN_THRESHOLD
                pair_flags[side] = is_short

                # Verbatim-repeat auto-halt accounting. A side counts
                # as "verbatim" if its current output exactly matches
                # its own previous output.
                is_verbatim = last_text[side] is not None and text == last_text[side]
                pair_verbatim_flags[side] = is_verbatim
                last_text[side] = text
                # A round-trip closes after side B speaks (turns 1, 3,
                # 5, ... since A speaks first). Easier: check after both
                # sides have a flag set in the current pair window.
                if turn_idx >= 1 and pair_flags["a"] and pair_flags["b"]:
                    consecutive_short_pairs += 1
                    if consecutive_short_pairs >= TRAILING_ACK_CONSECUTIVE:
                        halt_reason = (
                            f"trailing-ack: both sides ≤{TRAILING_ACK_TOKEN_THRESHOLD} "
                            f"tokens for {TRAILING_ACK_CONSECUTIVE} consecutive "
                            f"round-trips at turn {turn_idx}"
                        )
                        _save_log_line(log_fh, f"[HALT] {halt_reason}")
                        break
                    # Reset for next round-trip.
                    pair_flags = {"a": False, "b": False}
                elif turn_idx >= 1 and (not pair_flags["a"] or not pair_flags["b"]):
                    # Mixed round-trip resets the consecutive counter
                    # when the round-trip closes — only count true
                    # back-to-back all-short pairs. Roll over when this
                    # side closes the round-trip but only one side was
                    # short; reset counter and flags.
                    if side == "b":
                        consecutive_short_pairs = 0
                        pair_flags = {"a": False, "b": False}

                # Verbatim-repeat auto-halt accounting. Same closing-of-
                # round-trip pattern: evaluate after B speaks. A "verbatim
                # pair" means both A and B repeated their own previous
                # text in this round-trip.
                if turn_idx >= 3 and pair_verbatim_flags["a"] and pair_verbatim_flags["b"]:
                    consecutive_verbatim_pairs += 1
                    if consecutive_verbatim_pairs >= VERBATIM_REPEAT_CONSECUTIVE:
                        halt_reason = (
                            f"verbatim-repeat: both sides emitted identical text "
                            f"to their previous turn for {VERBATIM_REPEAT_CONSECUTIVE} "
                            f"consecutive round-trips at turn {turn_idx}"
                        )
                        _save_log_line(log_fh, f"[HALT] {halt_reason}")
                        break
                    pair_verbatim_flags = {"a": False, "b": False}
                elif side == "b":
                    if not (pair_verbatim_flags["a"] and pair_verbatim_flags["b"]):
                        consecutive_verbatim_pairs = 0
                        pair_verbatim_flags = {"a": False, "b": False}

            _save_log_line(
                log_fh,
                f"\n[done | turns_completed={n_completed} "
                f"| halt_reason={halt_reason or 'turn-cap'}]",
            )
    finally:
        log_fh.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Two-Qwen bot-vs-bot basin-emergence pilot."
    )
    ap.add_argument(
        "--config", choices=["neutral", "peer"], default="neutral",
        help="Which configuration to run. 'neutral' = no system prompt; "
             "'peer' = peer-frame system prompt (verbatim from claude run 2).",
    )
    ap.add_argument(
        "--model", default="qwen",
        help="Model short name (default 'qwen'). Must be probe-calibrated.",
    )
    ap.add_argument(
        "--seed-text", default=DEFAULT_SEED_TEXT,
        help=f"First user message to side A (default {DEFAULT_SEED_TEXT!r}).",
    )
    ap.add_argument(
        "--turns", type=int, default=DEFAULT_TURNS,
        help=f"Hard turn cap (default {DEFAULT_TURNS} = 12 round-trips).",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Per-turn generation budget (default {DEFAULT_MAX_TOKENS}).",
    )
    ap.add_argument(
        "--base-rng-seed", type=int, default=DEFAULT_BASE_RNG_SEED,
        help=f"Base RNG seed for per-turn SamplingConfig (default "
             f"{DEFAULT_BASE_RNG_SEED}); per-turn seeds are "
             f"base + turn_idx*2 + side_offset.",
    )
    args = ap.parse_args()

    # Honor env-var overrides as a convenience (matches the rest of the
    # pilot tooling) without making the CLI noisier.
    cfg = os.environ.get("LLMOJI_TWO_QWEN_CONFIG") or args.config
    run_loop(
        model_short=args.model,
        config=cfg,
        seed_text=args.seed_text,
        turns=args.turns,
        max_tokens=args.max_tokens,
        base_rng_seed=args.base_rng_seed,
    )


if __name__ == "__main__":
    main()
