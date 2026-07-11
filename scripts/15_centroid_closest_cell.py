"""Per-token closest-cell projection in centroid space.

Companion to ``30_mr_axis_score.py``. Where 30 projects each token onto
a single direction (``mr.nb``), this script computes the cosine between
each token's full layer-stack hidden state and each of the 10 cell
centroids (held-out A-half from ``40_centroid_holdout.py``), and
reports the modal cell per token.

This is the direct analog of the attractor-trajectory closest-cell
analysis in ``02b_attractor_analysis.py`` — but applied to arbitrary
input text rather than generated trajectories. For each scripture
chunk (or any other text), it answers: across the tokens, what
fraction land in each cell of centroid space?

Per-token operation:

- forward pass gives ``hidden_states[L]`` for each probe layer L
- per-cell, per-layer cosine contribution accumulated incrementally
  (so we never materialize a ``(n_tokens × n_layers × hidden_dim)``
  matrix in memory)
- final ``cos[token, cell] = sum_L(h_L · c_L) / (||h_total|| · ||c_total||)``
- ``closest_cell[token] = argmax_cell(cos[token, :])``

Output JSONL: input fields plus

- ``closest_cell_fractions``: dict ``{cell: fraction_of_tokens}``
- ``modal_cell``: most common closest cell
- ``modal_cell_frac``: its fraction
- ``mr_token_frac``: fraction of tokens whose closest cell is MR
- ``mr_max_cos``: max token-to-MR cosine across the text
- ``mr_mean_cos``: mean token-to-MR cosine across the text
- ``n_tokens``: token count after BOS / EOS trimming

CLI::

    .venv/bin/python scripts/local/30b_centroid_closest_cell.py \\
        --input data/scripture_chunks.jsonl \\
        --output data/local/scripture_mr/scripture_closest_cell_gemma.jsonl \\
        --model gemma \\
        --centroid-dir data/local/centroid_holdout/gemma \\
        --centroid-half A
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attractor_experiment.config import MODEL_REGISTRY  # noqa: E402


def _load_centroids(
    centroid_dir: Path, half: str,
) -> tuple[dict[str, np.ndarray], list[int], int]:
    """Load held-out centroids + layer-stack metadata.

    Returns ``(centroids_per_layer, layer_idxs, hidden_dim)`` where
    ``centroids_per_layer[cell]`` is a ``(n_layers, hidden_dim)`` array.
    """
    manifest = json.loads((centroid_dir / "manifest.json").read_text())
    layer_idxs: list[int] = list(manifest["layer_idxs"])
    n_layers = int(manifest["n_layers"])
    hidden_dim = int(manifest["hidden_dim_per_layer"])
    npz_path = centroid_dir / f"centroids_{half}.npz"
    if not npz_path.exists():
        raise SystemExit(f"no centroid file at {npz_path}")
    cents_flat = np.load(npz_path)
    cents: dict[str, np.ndarray] = {}
    for cell in cents_flat.files:
        v = cents_flat[cell]
        if v.shape != (n_layers * hidden_dim,):
            raise SystemExit(
                f"centroid {cell} shape {v.shape} != expected "
                f"({n_layers * hidden_dim},)"
            )
        cents[cell] = v.reshape(n_layers, hidden_dim).astype(np.float32)
    return cents, layer_idxs, hidden_dim


def _score_text(
    text: str,
    *,
    tokenizer,
    model,
    centroids: dict[str, np.ndarray],
    layer_idxs: list[int],
    device,
) -> dict:
    """Compute per-token closest-cell + summary stats.

    Memory-efficient accumulation: for each probe layer, accumulate
    ``dot[token, cell]`` and ``h_sq[token]`` and ``c_sq[cell]``,
    then derive per-token cosine to each cell at the end.
    """
    enc = tokenizer(
        text, return_tensors="pt", add_special_tokens=True,
        truncation=True, max_length=2048,
    ).to(device)
    input_ids = enc["input_ids"][0]
    n_tok_total = input_ids.shape[0]
    content_start = 1  # skip BOS
    content_end = n_tok_total
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        if int(input_ids[-1]) == tokenizer.eos_token_id:
            content_end = n_tok_total - 1
    if content_end <= content_start:
        return {
            "n_tokens": 0,
            "closest_cell_fractions": {},
            "modal_cell": None,
            "modal_cell_frac": float("nan"),
            "mr_token_frac": float("nan"),
            "mr_max_cos": float("nan"),
            "mr_mean_cos": float("nan"),
        }

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    all_hidden = out.hidden_states

    cells = sorted(centroids.keys())
    n_cells = len(cells)
    n_tokens = content_end - content_start

    # Incremental accumulators (numpy).
    dot = np.zeros((n_tokens, n_cells), dtype=np.float64)
    h_sq = np.zeros(n_tokens, dtype=np.float64)
    c_sq = np.zeros(n_cells, dtype=np.float64)

    # Pre-stack per-layer centroid slices, aligned with probe layer order.
    # centroids[cell] has shape (n_layers, hidden_dim) already aligned
    # with layer_idxs ordering (from 40_centroid_holdout's save layout).
    for k, L in enumerate(layer_idxs):
        if L >= len(all_hidden):
            continue
        h_L = all_hidden[L][0].float().cpu().numpy()
        h_L = h_L[content_start:content_end]  # (n_tokens, hidden_dim)
        h_sq += (h_L ** 2).sum(axis=1)
        for ci, cell in enumerate(cells):
            c_L = centroids[cell][k]  # (hidden_dim,)
            dot[:, ci] += h_L @ c_L
            c_sq[ci] += float((c_L ** 2).sum())

    denom = np.sqrt(np.outer(h_sq, c_sq))
    denom = np.where(denom == 0, 1.0, denom)
    cos = dot / denom  # (n_tokens, n_cells)

    closest = np.argmax(cos, axis=1)
    counts: dict[str, int] = {cell: 0 for cell in cells}
    for idx in closest:
        counts[cells[idx]] += 1
    fractions = {cell: counts[cell] / n_tokens for cell in cells}

    modal_cell = max(fractions.items(), key=lambda kv: kv[1])[0]

    if "MR" in cells:
        mr_idx = cells.index("MR")
        mr_token_frac = float(fractions["MR"])
        mr_max_cos = float(cos[:, mr_idx].max())
        mr_mean_cos = float(cos[:, mr_idx].mean())
    else:
        mr_token_frac = float("nan")
        mr_max_cos = float("nan")
        mr_mean_cos = float("nan")

    return {
        "n_tokens": int(n_tokens),
        "closest_cell_fractions": fractions,
        "modal_cell": modal_cell,
        "modal_cell_frac": float(fractions[modal_cell]),
        "mr_token_frac": mr_token_frac,
        "mr_max_cos": mr_max_cos,
        "mr_mean_cos": mr_mean_cos,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default="gemma")
    ap.add_argument(
        "--centroid-dir",
        type=Path,
        default=None,
        help="default: data/local/centroid_holdout/{model}",
    )
    ap.add_argument(
        "--centroid-half",
        default="A",
        choices=["A", "B"],
        help="which held-out half to use as the projection basis",
    )
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"unknown model {args.model!r}")
    M = MODEL_REGISTRY[args.model]
    print(f"model: {M.short_name} ({M.model_id})")

    cent_dir = args.centroid_dir or (
        Path("data/local/centroid_holdout") / args.model
    )
    if not cent_dir.exists():
        raise SystemExit(
            f"no centroid hold-out at {cent_dir}. Run "
            f"`scripts/local/40_centroid_holdout.py --model {args.model}` first."
        )

    centroids, layer_idxs, hidden_dim = _load_centroids(cent_dir, args.centroid_half)
    print(
        f"centroids: {len(centroids)} cells ({', '.join(sorted(centroids))}) "
        f"x {len(layer_idxs)} layers x {hidden_dim} hidden_dim "
        f"(half={args.centroid_half})"
    )

    # Load model.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = (
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"loading {M.model_id} on {device}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(M.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        M.model_id,
        trust_remote_code=M.trust_remote_code,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map="auto" if device != "cpu" else None,
    )
    # Switch to inference mode (disables dropout etc.).
    model.train(False)
    print(f"  loaded in {time.time() - t0:.1f}s on {device}")

    # Iterate over rows.
    rows = [json.loads(l) for l in args.input.open() if l.strip()]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    print(f"input: {len(rows)} rows from {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    with args.output.open("w") as fh_out:
        for i, row in enumerate(rows):
            text = row.get("text", "")
            try:
                stats = _score_text(
                    text,
                    tokenizer=tokenizer,
                    model=model,
                    centroids=centroids,
                    layer_idxs=layer_idxs,
                    device=device,
                )
            except Exception as exc:
                stats = {
                    "n_tokens": 0,
                    "closest_cell_fractions": {},
                    "modal_cell": None,
                    "modal_cell_frac": float("nan"),
                    "mr_token_frac": float("nan"),
                    "mr_max_cos": float("nan"),
                    "mr_mean_cos": float("nan"),
                    "error": str(exc),
                }
            out_row = {**row, **stats}
            fh_out.write(json.dumps(out_row, ensure_ascii=False))
            fh_out.write("\n")
            if (i + 1) % 20 == 0 or i + 1 == len(rows):
                dt = time.time() - t_start
                print(
                    f"  [{i+1}/{len(rows)}] modal={stats.get('modal_cell')!s:>5s} "
                    f"mr_frac={stats.get('mr_token_frac', float('nan')):.2f} "
                    f"mr_max={stats.get('mr_max_cos', float('nan')):+.3f}  "
                    f"({dt:.1f}s)"
                )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
