# Artifacts & reproducibility manifest

What is committed to this repo, and where every excluded (large/licensed) input comes from.

## In the repo (code + results)
- **Code:** `src/transport_study/` (driver, operators, decoders, evaluation, gate),
  `tests/`, `scripts/state/` (STATE-phase scripts).
- **Docs:** `docs/*.md` — study contract, amendment, results, recovery analysis, STATE
  comparison; `docs/dashboard/results.html` (self-contained results dashboard).
- **Result tables** (`artifacts/metrics/`, small parquet/JSON):
  - `dong_metrics.parquet` — full 700-manifest Stack run (4 methods × 3 metrics).
  - `GATE.json` — the four-contrast gate verdict (fails).
  - `floor_baseline.parquet` — Stack-free gene-space floors (no-op, gene/PCA mean-shift).
  - `state_metrics.parquet` — early Dong STATE run (superseded; kept for provenance).
  - `parse_fair_metrics.parquet` — **the corrected in-distribution Parse comparison**
    (our steering vs STATE transition vs floor, STATE's own decoder for both).
  - `CELL_COUNT_AUDIT.json`, `LOCK.json` — sampling audit + lock.
- **Paper:** `2025.06.26.661135v2.full.pdf` (STATE preprint, CC-BY).

## Not in the repo — external downloads (too large / licensed)

| artifact | source | notes |
|---|---|---|
| Dong raw data (`data/Integrated_raw.h5ad`, `Integrated_.h5ad`) | Zenodo record 8180343 | the raw counts + annotations |
| Stack model (`artifacts/models/Stack-Large-Aligned/`) | HF `arcinstitute/Stack-Large-Aligned` rev `b09f085…` | checkpoint + gene pkl |
| STATE embedding (`artifacts/models/SE-600M/`) | HF `arcinstitute/SE-600M` | ~14 GB incl. `se600m_epoch16.ckpt` |
| STATE transition (`artifacts/models/ST-SE-Parse/`) | HF `arcinstitute/ST-SE-Parse` | `zeroshot/split_0` + `split_4` eval adatas + checkpoints |
| Parse full data (optional) | HF `arcinstitute/State-Parse-Filtered` | 52 GB; **not needed** — per-split eval adatas suffice |
| cell-eval, Stack, cinema-ot | git submodules (`.gitmodules`) | `git submodule update --init` |

## Regenerable (deterministic — not committed)
- **Manifests** (`artifacts/manifests/*.json`, 1442 files / 451 MB): `transport-study build-manifests`
  (seed 1729). Only the audit + lock summaries are committed.
- **SE embeddings** (`*_emb.h5ad`): `state emb transform --model-folder artifacts/models/SE-600M ...`.
- **STATE predictions / intermediate `.npz`**: `scripts/state/parse_fair_steering.py` (needs `.venv-state`).

## Environments
- Main `.venv`: `uv sync` (Stack + cell-eval + torch 2.5.1 cu121).
- Isolated `.venv-state`: `uv venv .venv-state && VIRTUAL_ENV=.venv-state uv pip install arc-state`
  (its triton 3.7 conflicts with the main env's torch — keep them separate).

## Scripts (`scripts/state/`)
- `floor_baseline.py` — gene-space floors.
- `state_mmd.py`, `state_transition.py` — early Dong STATE run (superseded).
- `parse_fair_steering.py` — **loads STATE's real model, MMD-steers SE embeddings, decodes with
  STATE's own `gene_decoder`** (the fair, corrected comparison).
- `parse_fair_eval.py` — cell-eval the Parse predictions.
- `parse_ridge_superseded.py` — the first Parse attempt with a ridge decoder (why it was unfair).
