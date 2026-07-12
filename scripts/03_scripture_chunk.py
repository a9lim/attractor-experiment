"""Chunk the scripture corpus into JSONL for MR-axis scoring.

PROJECT_2 phase 3.1 prep step. Reads each scripture file at
``../a9lim.github.io/scripture/text/<source>.txt``, strips metadata
lines (WORK / BOOK / CHAPTER / license/copyright header), concatenates
verses, splits into ~300-word chunks at sentence boundaries, and
writes a JSONL of ``{id, source, text}`` rows.

The output is consumed by ``scripts/local/30_mr_axis_score.py``.

Output format::

    {"id": "ot_0042", "source": "ot", "text": "..."}

Per-scripture chunk count is capped at ``--chunks-per-source`` (default
50) via uniform random sampling across the source. Seed is fixed for
reproducibility.

Usage::

    python scripts/local/44a_scripture_chunk.py \\
        --corpus-dir ../a9lim.github.io/scripture/text \\
        --output data/scripture_chunks.jsonl \\
        [--chunks-per-source 50] \\
        [--words-per-chunk 300] \\
        [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# Skip these scripture files (license-only / not actually scripture).
SKIP_FILES = {"LICENSE"}

# Metadata-line prefixes to strip from scripture text.
METADATA_PATTERNS = [
    re.compile(r"^\s*This structured text is licensed"),
    re.compile(r"^\s*https://creativecommons"),
    re.compile(r"^\s*Source text is public domain"),
    re.compile(r"^\s*WORK:"),
    re.compile(r"^\s*BOOK:"),
    re.compile(r"^\s*CHAPTER:"),
    re.compile(r"^\s*VERSE:"),
    re.compile(r"^\s*©"),
    re.compile(r"^\s*\(c\)"),
]

# Sentence-ending punctuation, allowing for paragraph breaks.
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _strip_metadata(text: str) -> str:
    """Remove license, work/book/chapter markers; keep verse content."""
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(p.match(stripped) for p in METADATA_PATTERNS):
            continue
        out_lines.append(stripped)
    return " ".join(out_lines)


def _chunk_words(text: str, target_words: int) -> list[str]:
    """Split a flat text into ~target_words chunks at sentence boundaries.

    Greedy: accumulate sentences until reaching target_words, then emit.
    Doesn't break mid-sentence.
    """
    sentences = SENTENCE_RE.split(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_wc = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        wc = len(sent.split())
        if buf and buf_wc + wc > target_words:
            chunks.append(" ".join(buf))
            buf = []
            buf_wc = 0
        buf.append(sent)
        buf_wc += wc
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def process_source(
    path: Path, source: str, words_per_chunk: int
) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    clean = _strip_metadata(raw)
    chunks = _chunk_words(clean, words_per_chunk)
    rows = [
        {"id": f"{source}_{i:04d}", "source": source, "text": chunk}
        for i, chunk in enumerate(chunks)
    ]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("../a9lim.github.io/scripture/text"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/scripture_chunks.jsonl"),
    )
    ap.add_argument("--chunks-per-source", type=int, default=50)
    ap.add_argument("--words-per-chunk", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpus_dir = args.corpus_dir.resolve()
    if not corpus_dir.exists():
        raise SystemExit(f"corpus dir not found: {corpus_dir}")

    rng = random.Random(args.seed)
    txt_files = sorted(corpus_dir.glob("*.txt"))
    print(f"corpus: {len(txt_files)} txt files at {corpus_dir}")

    all_rows: list[dict] = []
    summary: list[tuple[str, int, int]] = []  # (source, n_total, n_kept)

    for txt in txt_files:
        if txt.stem in SKIP_FILES:
            continue
        source = txt.stem
        rows = process_source(txt, source, args.words_per_chunk)
        n_total = len(rows)
        if n_total > args.chunks_per_source:
            rows = rng.sample(rows, args.chunks_per_source)
            rows.sort(key=lambda r: r["id"])
        n_kept = len(rows)
        all_rows.extend(rows)
        summary.append((source, n_total, n_kept))
        print(
            f"  {source:>12s}  total={n_total:5d}  kept={n_kept:4d}  "
            f"(~{args.words_per_chunk}w/chunk)"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")

    print(f"\nwrote {len(all_rows)} chunks to {args.output}")
    print("per-source summary:")
    for source, n_total, n_kept in sorted(summary, key=lambda x: -x[2]):
        print(f"  {source:>12s}  {n_kept:4d} / {n_total:5d}")


if __name__ == "__main__":
    main()
