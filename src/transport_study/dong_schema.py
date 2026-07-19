"""Join the deposited Dong annotations to the raw-count cell order."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


BATCH_ALIASES = {
    "Donor 1 acute": "H1D2",
    "Donor 1 chronic": "H1D7",
    "Donor 2 acute": "H2D2",
    "Donor 2 chronic": "H2D7",
    "Donor 3 acute": "H3D2",
    "Donor 3 chronic": "H3D7",
    "H1D2": "H1D2",
    "H1D7": "H1D7",
    "H2D2": "H2D2",
    "H2D7": "H2D7",
    "H3D2": "H3D2",
    "H3D7": "H3D7",
}

CONDITION_ALIASES = {
    "No stimulation": "No stimulation",
    "IFNa2": "IFN-α2",
    "IFNb": "IFN-β",
    "IFNg": "IFN-γ",
    "IFN-III": "IFN-III/IL-29",
    "IL-6": "IL-6",
    "TNFa": "TNF-α",
    "IFNb + TNFa": "IFN-β + TNF-α",
    "IFNb + IFNg": "IFN-β + IFN-γ",
    "IFNb+ IFNg": "IFN-β + IFN-γ",
    "IFNb + IL-6": "IFN-β + IL-6",
    "IFNb+IL-6": "IFN-β + IL-6",
}
CONDITION_ALIASES.update({value: value for value in CONDITION_ALIASES.values()})


@dataclass(frozen=True)
class DongMetadata:
    obs: pd.DataFrame
    genes: tuple[str, ...]
    report: dict


def _read_obs(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        return ad.io.read_elem(handle["obs"])


def _read_genes(path: Path) -> tuple[str, ...]:
    with h5py.File(path, "r") as handle:
        return tuple(map(str, ad.io.read_elem(handle["var"]).index))


def _map_labels(values: pd.Series, aliases: dict[str, str], name: str) -> pd.Series:
    text = values.astype(str)
    unknown = sorted(set(text) - set(aliases))
    if unknown:
        raise ValueError(f"unknown {name} labels: {unknown}")
    return text.map(aliases)


def prepare_dong_metadata(
    raw_adata: Path,
    annotated_adata: Path,
    *,
    celltype_col: str = "cell_type0528",
) -> DongMetadata:
    """Transfer annotations after verifying identical deposited row order."""
    raw_adata = Path(raw_adata)
    annotated_adata = Path(annotated_adata)
    raw = _read_obs(raw_adata)
    annotated = _read_obs(annotated_adata)

    for frame, name, columns in (
        (raw, "raw", ("batch", "perturbation")),
        (annotated, "annotated", ("batch", "perturbation", celltype_col)),
    ):
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(f"{name} H5AD is missing obs columns: {missing}")

    raw_barcodes = raw.index.astype(str).to_numpy()
    annotated_barcodes = annotated.index.astype(str).to_numpy()
    if not np.array_equal(raw_barcodes, annotated_barcodes):
        raise ValueError("raw and annotated H5AD observation order differs")

    raw_batches = _map_labels(raw["batch"], BATCH_ALIASES, "batch")
    annotated_batches = _map_labels(annotated["batch"], BATCH_ALIASES, "batch")
    if not np.array_equal(raw_batches.to_numpy(), annotated_batches.to_numpy()):
        raise ValueError("raw and annotated H5AD batches differ")

    raw_conditions = _map_labels(raw["perturbation"], CONDITION_ALIASES, "condition")
    annotated_conditions = _map_labels(
        annotated["perturbation"], CONDITION_ALIASES, "condition"
    )
    if not np.array_equal(raw_conditions.to_numpy(), annotated_conditions.to_numpy()):
        raise ValueError("raw and annotated H5AD conditions differ")

    cell_ids = pd.Index(
        [f"{batch}:{barcode}" for batch, barcode in zip(raw_batches, raw_barcodes)],
        name="cell_id",
    )
    if not cell_ids.is_unique:
        raise ValueError("batch and barcode do not uniquely identify Dong cells")

    genes = _read_genes(raw_adata)
    if len(genes) != len(set(genes)):
        raise ValueError("raw H5AD gene identifiers are not unique")

    obs = pd.DataFrame(
        {
            "raw_position": np.arange(len(raw), dtype=np.int64),
            "barcode": raw_barcodes,
            "sample": raw_batches.to_numpy(),
            "cytokine": raw_conditions.to_numpy(),
            "cell_type0528": annotated[celltype_col].astype(str).to_numpy(),
        },
        index=cell_ids,
    )
    report = {
        "raw_adata": str(raw_adata),
        "annotated_adata": str(annotated_adata),
        "n_obs": len(obs),
        "n_vars": len(genes),
        "contains_expression": False,
    }
    return DongMetadata(obs, genes, report)


def write_dong_metadata(metadata: DongMetadata, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata.obs.to_parquet(out / "obs.parquet")
    (out / "genes.json").write_text(json.dumps(metadata.genes) + "\n")
    (out / "PREPARATION.json").write_text(json.dumps(metadata.report, indent=2) + "\n")
    return metadata.report


def read_dong_metadata(path: Path) -> DongMetadata:
    path = Path(path)
    obs = pd.read_parquet(path / "obs.parquet")
    genes = tuple(json.loads((path / "genes.json").read_text()))
    report = json.loads((path / "PREPARATION.json").read_text())
    if not obs.index.is_unique or len(genes) != len(set(genes)):
        raise ValueError("prepared Dong metadata has duplicate cell or gene identifiers")
    if report["n_obs"] != len(obs) or report["n_vars"] != len(genes):
        raise ValueError("prepared Dong dimensions do not match PREPARATION.json")
    return DongMetadata(obs, genes, report)
