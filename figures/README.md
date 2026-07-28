# Figures

All result figures in this repository are tracked. The main families are:

- `local/{gemma,qwen,ministral}_attractor/` — cross-content trajectories and
  distance curves.
- `local/{gemma,qwen,ministral}_attractor_lda_continuation/` —
  continuation-basis trajectories and centroids.
- `local/gemma_attractor_steered/` and `local/qwen_attractor_steered/` —
  steering dose response.
- `local/*_h_last_pca/` — PCA views of the continuation state.
- `local/qwen_two_loop/` — two-agent loop pilots.

The compact numbers behind the claim summary are in
[`data/summary/results.json`](../data/summary/results.json), with detailed
per-analysis JSON under `data/local/basin_stats/`. Legacy `lb` filenames
predate the rename to MR and are retained for provenance.
