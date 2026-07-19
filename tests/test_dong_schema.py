import anndata as ad
import numpy as np
import pandas as pd
import pytest

from transport_study.dong_schema import prepare_dong_metadata, read_dong_metadata, write_dong_metadata
from transport_study.manifests import build_manifests_from_metadata


def _dong_pair(tmp_path, *, reorder=False):
    barcodes = ["a", "a", "b"]
    raw_obs = pd.DataFrame(
        {"batch": ["Donor 1 acute", "Donor 2 acute", "Donor 2 acute"],
         "perturbation": ["IFNa2", "No stimulation", "TNFa"]},
        index=barcodes,
    )
    annotated_obs = pd.DataFrame(
        {"batch": ["H1D2", "H2D2", "H2D2"],
         "perturbation": ["IFNa2", "No stimulation", "TNFa"],
         "cell_type0528": ["CD4 T", "B", "Monocyte"]},
        index=barcodes,
    )
    if reorder:
        annotated_obs = annotated_obs.iloc[[1, 0, 2]]
    with pytest.warns(UserWarning, match="Observation names are not unique"):
        raw = ad.AnnData(np.ones((3, 2)), obs=raw_obs, var=pd.DataFrame(index=["g1", "g2"]))
    with pytest.warns(UserWarning, match="Observation names are not unique"):
        annotated = ad.AnnData(np.ones((3, 2)), obs=annotated_obs, var=pd.DataFrame(index=["g1", "g2"]))
    raw_path, annotated_path = tmp_path / "raw.h5ad", tmp_path / "annotated.h5ad"
    raw.write_h5ad(raw_path)
    annotated.write_h5ad(annotated_path)
    return raw_path, annotated_path


def test_prepare_dong_metadata_maps_labels_and_roundtrips(tmp_path):
    raw, annotated = _dong_pair(tmp_path)
    metadata = prepare_dong_metadata(raw, annotated)
    assert metadata.obs.index.tolist() == ["H1D2:a", "H2D2:a", "H2D2:b"]
    assert metadata.obs["cytokine"].tolist() == ["IFN-α2", "No stimulation", "TNF-α"]
    write_dong_metadata(metadata, tmp_path / "prepared")
    loaded = read_dong_metadata(tmp_path / "prepared")
    pd.testing.assert_frame_equal(loaded.obs, metadata.obs)
    assert loaded.genes == metadata.genes
    lock = build_manifests_from_metadata(loaded.obs, loaded.genes, tmp_path / "manifests", n_resamples=1)
    assert lock["manifest_count"] == 0
    assert (tmp_path / "manifests" / "CELL_COUNT_AUDIT.json").exists()


def test_prepare_dong_metadata_rejects_mismatched_rows(tmp_path):
    raw, annotated = _dong_pair(tmp_path, reorder=True)
    with pytest.raises(ValueError, match="batches differ"):
        prepare_dong_metadata(raw, annotated)
