# Transport Steering Study

Research code for comparing:

- transport steering in Stack latent space versus Stack in-context learning;
- Stack-space transport versus the same operator in matched PCA space; and
- global versus initial-state-conditioned transport across held-out populations.

The study design is in [`docs/DONG_CONTRACT.md`](docs/DONG_CONTRACT.md).

## Environment

```bash
git submodule update --init --recursive
uv sync --locked --python 3.10 --extra test
source .venv/bin/activate
```

The lock installs the pinned Stack submodule and the tested PyTorch 2.5.1 CUDA
12.1 build.

## Local inputs

The current checkout already contains these ignored local assets:

```text
data/Integrated_raw.h5ad
data/Integrated_.h5ad
artifacts/models/Stack-Large-Aligned/bc_large_aligned.ckpt
artifacts/models/Stack-Large-Aligned/basecount_1000per_15000max.pkl
```

The Dong files come from <https://zenodo.org/records/8180343>. The Stack files
come from revision `b09f085dac03d170b078a5c72f550ae93686e544` of
<https://huggingface.co/arcinstitute/Stack-Large-Aligned>.

## Prepare Dong metadata

The deposited raw file contains counts but not cell-type annotations. The
annotated file has the labels but transformed expression. Join only their
metadata while retaining the raw row positions and gene order:

```bash
transport-study prepare-dong-metadata \
  --raw-adata data/Integrated_raw.h5ad \
  --annotated-adata data/Integrated_.h5ad \
  --out artifacts/dong_metadata
```

Then construct the experimental cell manifests:

```bash
transport-study build-manifests \
  --metadata artifacts/dong_metadata \
  --out artifacts/manifests
```

The contract preregisters role sizes 512, 256, and 128; none is feasible on the
deposited Dong data (too few B and Dendritic cells). Per the recorded amendment in
[`docs/AMENDMENT_ROLE_SIZE.md`](docs/AMENDMENT_ROLE_SIZE.md), the tier list is
extended and the unchanged "largest passing tier" rule now selects role size **40**
(17 of 18 two-donor units eligible, every class and cytokine covered). The command
writes the cell-count audit and 700 eligible fine-matched manifests, exiting 0.

## Tests

```bash
pytest -q
```

## Run the study

The end-to-end driver (`src/transport_study/run.py`) loads Stack, extracts post-L9
embeddings via the parity-verified manual path, fits the transport maps, runs
deterministic in-context generation, decodes (native NB head and matched ridge),
applies the anchored synthetic-control correction, and writes the metric table:

```bash
transport-study run \
  --manifests artifacts/manifests --metadata artifacts/dong_metadata \
  --raw-adata data/Integrated_raw.h5ad \
  --ckpt artifacts/models/Stack-Large-Aligned/bc_large_aligned.ckpt \
  --genes artifacts/models/Stack-Large-Aligned/basecount_1000per_15000max.pkl \
  --transport-configs artifacts/crossfit/selected_mmd.json \
  --out artifacts/metrics/dong_metrics.parquet

transport-study gate --metrics artifacts/metrics/dong_metrics.parquet \
  --out artifacts/metrics/GATE.json
```

`selected_mmd.json` must contain independently selected settings for both target
donors. The selection objective must be decided and recorded without using the
sealed target donor. The runner validates that every value belongs to the registered
grid. This example illustrates the schema only; its numbers are not selected results:

```json
{
  "H2D2": {"selected_on_donor": "H3D2", "rank": 8, "alpha": 1.0, "bandwidth_multiplier": 1.0, "movement": 0.01},
  "H3D2": {"selected_on_donor": "H2D2", "rank": 8, "alpha": 1.0, "bandwidth_multiplier": 1.0, "movement": 0.01}
}
```

The runner intentionally refuses to substitute fixed defaults because that would
not constitute the contract's donor-cross-fit primary analysis.

All four methods run in Stack's 15012-gene space. Steering is applied at the final
Stack layer (after block 9): the NB head is per-cell, so transporting the post-L9
embedding and decoding it equals a layer-9 intervention (verified to 0.0 against the
native forward). `--limit N` runs the first N eligible manifests for a pilot.

Evaluation uses the pinned **cell-eval `v0.6.6`** (`vendor/cell-eval`): `pearson_delta`
for delta-Pearson and `de_spearman_lfc_sig` for DE-LFC correlation, with the DE gene
sets from Arc's `pdex` (Wilcoxon). Energy distance uses cell-eval's E-distance formula
in the fixed 50-component control-only evaluator space. Predictions are frozen (anchored
synthetic-control corrected) before cell-eval sees any Y1 expression.

Pinned revisions:

- Stack code: `cacc2e4b09435c3e536d46237d10b50f222dd144`
- Stack model: `b09f085dac03d170b078a5c72f550ae93686e544`
- cell-eval primary: `v0.6.6`
- cell-eval secondary: `v0.8.1`
