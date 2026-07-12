"""Project arbitrary text into MR-axis cosine scores.

For each input text, computes the cosine of the model's residual-
stream representation with the registered MR direction (centroid
difference MR - NB, from ``~/.saklas/vectors/llmoji/mr.nb/``) at
each probe layer, then aggregates across layers and across tokens.

Use cases
---------
- **Score deployed-model output**: "how MR-coded is this generation?"
- **Audit corpora**: "what fraction of [text source] sits in the MR
  basin?" Useful for pre-internet vs post-internet corpus
  comparisons, cross-cultural extension, deployment-side basin
  monitoring.
- **Test the MR-axis as a deployment-side monitor**: do
  bliss-attractor / doom / conspiracy / sycophancy texts that have
  *not* been seen during basin-calibration still score high on
  MR-axis?
- **Validate the basin's content-blindness empirically**: score
  bliss-content, doom-content, conspiracy-content, sycophancy-
  content excerpts; if they all score high on MR-axis (despite
  surface-content diversity), the basin's content-blindness is
  confirmed at the embedding level too.

Per-layer projection: dot(h_token_L, mr.nb_L) / ||mr.nb_L||
(cosine-like quantity, sensitive to representation magnitude but
unit-normalized in the probe direction; values typically in
[-1, +1] range modulo the residual-stream norm).

Per-token score: mean across probe layers (treats each layer
equally; matches how MR centroids were computed).

Per-text aggregates: mean (overall basin-ness), max (did this text
pass through the basin at any point), min (anti-basin moments).

Input
-----
JSONL with one ``{"id": str, "text": str}`` row per line. Other
fields are passed through unchanged.

Output
------
JSONL with input fields plus:

- ``mr_score_mean``: mean per-token MR-axis cosine across the text
- ``mr_score_max``: max per-token MR-axis cosine
- ``mr_score_min``: min per-token MR-axis cosine
- ``mr_score_p95``: 95th-percentile per-token cosine
- ``n_tokens``: number of tokens after the model's BOS / chat-
  template treatment
- ``per_token_mr`` (optional, when ``--per-token`` set): list of
  per-token MR scores

CLI
---

::

    python scripts/local/30_mr_axis_score.py \
        --input data/some_text.jsonl \
        --output data/some_text_mr_scores.jsonl \
        --model gemma            # default
        [--per-token]            # include per-token scores
        [--max-rows N]           # process only first N rows
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_experiment.config import MODEL_REGISTRY  # noqa: E402


def _saklas_namespace_dir() -> Path:
    home = Path.home() / ".saklas" / "vectors" / "llmoji"
    if not home.exists():
        raise SystemExit(
            f"No saklas llmoji vectors at {home}. Register first via "
            f"`scripts/local/22c_register_centroid_probes.py`."
        )
    return home


def _safe_model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _load_mr_axis(model_id: str):
    """Load registered mr.nb direction; return per-layer unit vectors."""
    slug = _safe_model_slug(model_id)
    path = _saklas_namespace_dir() / "mr.nb" / f"{slug}.safetensors"
    if not path.exists():
        raise SystemExit(
            f"No mr.nb probe for {model_id} at {path}. "
            f"Register via scripts/local/22c_register_centroid_probes.py."
        )
    tensors = safetensors.torch.load_file(str(path))
    layer_re = re.compile(r"layer_(\d+)")
    per_layer: dict[int, np.ndarray] = {}
    for key, tensor in tensors.items():
        m = layer_re.search(key)
        if m is None:
            continue
        v = tensor.float().numpy()
        norm = float(np.linalg.norm(v))
        if norm == 0:
            continue
        per_layer[int(m.group(1))] = v / norm
    if not per_layer:
        raise SystemExit(f"no layer vectors found in {path}")
    layer_idxs = sorted(per_layer.keys())
    hidden_dim = per_layer[layer_idxs[0]].shape[0]
    return per_layer, layer_idxs, hidden_dim


def _score_text(text, *, tokenizer, model, mr_vectors, mr_layers, device):
    enc = tokenizer(
        text, return_tensors="pt", add_special_tokens=True,
        truncation=True, max_length=2048,
    ).to(device)
    input_ids = enc["input_ids"][0]
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    all_hidden = out.hidden_states
    n_tok_total = input_ids.shape[0]
    content_start = 1  # skip BOS
    content_end = n_tok_total
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        if int(input_ids[-1]) == tokenizer.eos_token_id:
            content_end = n_tok_total - 1
    if content_end <= content_start:
        return {
            "n_tokens": 0,
            "mr_score_mean": float("nan"),
            "mr_score_max": float("nan"),
            "mr_score_min": float("nan"),
            "mr_score_p95": float("nan"),
            "per_token_mr": [],
        }
    per_token_scores_layers = []
    for L in mr_layers:
        if L >= len(all_hidden):
            continue
        h_L = all_hidden[L][0].float().cpu().numpy()
        h_L = h_L[content_start:content_end]
        h_norms = np.linalg.norm(h_L, axis=-1, keepdims=True)
        h_norms = np.where(h_norms == 0, 1.0, h_norms)
        cos = (h_L @ mr_vectors[L]) / h_norms[:, 0]
        per_token_scores_layers.append(cos)
    if not per_token_scores_layers:
        return {
            "n_tokens": content_end - content_start,
            "mr_score_mean": float("nan"),
            "mr_score_max": float("nan"),
            "mr_score_min": float("nan"),
            "mr_score_p95": float("nan"),
            "per_token_mr": [],
        }
    per_token_score = np.stack(per_token_scores_layers, axis=0).mean(axis=0)
    return {
        "n_tokens": int(per_token_score.shape[0]),
        "mr_score_mean": float(per_token_score.mean()),
        "mr_score_max": float(per_token_score.max()),
        "mr_score_min": float(per_token_score.min()),
        "mr_score_p95": float(np.percentile(per_token_score, 95)),
        "per_token_mr": [float(x) for x in per_token_score],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--per-token", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"unknown model {args.model!r}")
    M = MODEL_REGISTRY[args.model]
    print(f"model: {M.short_name} ({M.model_id})")

    mr_vectors, mr_layers, mr_hd = _load_mr_axis(M.model_id)
    print(f"mr.nb probe: {len(mr_layers)} layers x {mr_hd} hidden_dim "
          f"(L{mr_layers[0]}..L{mr_layers[-1]})")

    if M.trust_remote_code:
        os.environ.setdefault("HF_TRUST_REMOTE_CODE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        M.model_id, trust_remote_code=M.trust_remote_code,
    )
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = AutoModelForCausalLM.from_pretrained(
        M.model_id,
        trust_remote_code=M.trust_remote_code,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map="auto" if device != "cpu" else None,
    )
    # Switch to inference mode (disables dropout etc.). Equivalent to
    # model.eval() but uses train(False) to keep the hook surface clean.
    model.train(False)
    print(f"  loaded in {time.time() - t_load:.1f}s on {device}")

    rows_in = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows_in.append(json.loads(line))
    if args.max_rows is not None:
        rows_in = rows_in[: args.max_rows]
    print(f"input: {len(rows_in)} rows from {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with args.output.open("w") as out:
        for i, row in enumerate(rows_in, 1):
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                row_out = {
                    **row,
                    "n_tokens": 0,
                    "mr_score_mean": float("nan"),
                    "mr_score_max": float("nan"),
                    "mr_score_min": float("nan"),
                    "mr_score_p95": float("nan"),
                }
                out.write(json.dumps(row_out) + "\n")
                continue
            scores = _score_text(
                text, tokenizer=tokenizer, model=model,
                mr_vectors=mr_vectors, mr_layers=mr_layers, device=device,
            )
            row_out = {**row, **scores}
            if not args.per_token:
                row_out.pop("per_token_mr", None)
            out.write(json.dumps(row_out) + "\n")
            if i % 20 == 0 or i == len(rows_in):
                dt = time.time() - t0
                snippet = text[:60].replace("\n", " ")
                print(f"  [{i}/{len(rows_in)}] "
                      f"mean={scores['mr_score_mean']:+.3f} "
                      f"max={scores['mr_score_max']:+.3f}  "
                      f"({dt:.1f}s) | {snippet}")

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
