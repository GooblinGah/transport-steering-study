# Transport Steering Study

This project implements the revised Dong test.

The project has two result tables.

- The end-to-end table compares complete methods.
- The representation-controlled table compares Stack and PCA with matched decoders.

The project keeps treated target expression sealed during all fit and selection steps.
The project records hashes for manifests, windows, predictions, decoders, gene panels, and metrics.

## Terms

- **Sealed data** is data that a fit or selection step cannot use.
- **Gate** is the set of requirements that controls the next study stage.
- **Expected counts** are deterministic mean counts from a prediction model.
- **Technical resample** is a repeated calculation with a new cell sample or seed.

## Install the project

Activate the Python environment.

```bash
source /venv/main/bin/activate
```

Install the project and the test packages.

```bash
uv pip install -e '.[test]'
```

## Check the storage

Run the storage check.

```bash
transport-study preflight
```

For a production run, use a persistent volume with 0.5 TB to 1 TB of space.
The current `/workspace` directory is not persistent.

## Make the Dong manifests

Use the deposited raw-count file.

```bash
transport-study build-manifests \
  --adata data/Integrated_raw.h5ad \
  --celltype-col cell_type0528 \
  --out artifacts/manifests
```

The command audits cell counts before it selects a role size.
The command tests role sizes of 512, 256, and 128 cells.
The command selects one role size for all eligible tasks.

The command stops if no role size meets the coverage requirements.
Do not add a role size of 64 cells without a recorded study amendment.

## Run the tests

```bash
pytest -q
```

## Apply the gate

Use the locked metric records.

```bash
transport-study gate \
  --metrics artifacts/metrics/locked.parquet \
  --out artifacts/gate/dong.json
```

Do not start the SciPlex3 or STATE stage unless the output contains `"passed": true`.

## Pinned revisions

- Stack code: `cacc2e4b09435c3e536d46237d10b50f222dd144`
- Stack-Large-Aligned: `b09f085dac03d170b078a5c72f550ae93686e544`
- Primary cell-eval version: `v0.6.6`
- Secondary cell-eval version: `v0.8.1`

Read [the Dong contract](docs/DONG_CONTRACT.md) before you run a model.
