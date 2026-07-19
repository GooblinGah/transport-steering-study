import numpy as np
import pandas as pd
import pytest
import anndata as ad

from transport_study.decoders import fit_matched_decoders
from transport_study.evaluation import contrast_summary, gate
from transport_study.manifests import _quota_from_control, build_manifests, standardize_fine_label
from transport_study.operators import sinkhorn_coupling
from transport_study.output_scale import (
    library_sizes, log1p_to_expected_counts, normalize_log1p,
    prepare_for_evaluation, synthetic_control_correct,
)
from transport_study.stack_context import build_query_windows, build_source_fit_windows, build_icl_windows, manual_no_mask_forward, official_icl_schedule
from transport_study.panels import fit_control_only_panel

DONORS=("H2D2","H3D2")
CYTOKINES=("IFN-α2","IFN-β","IFN-γ","IFN-III/IL-29","IL-6","TNF-α")
CLASSES=("B","T","Myeloid")


def contrast_frame(effect=.021, table="end_to_end", method="stack_mmd_native", comparator="stack_icl_native", metrics=("delta_pearson",)):
    rows=[]
    decoder=table=="representation_controlled"
    for donor in DONORS:
        for cytokine in CYTOKINES:
            for cls in CLASSES:
                for resample in range(20):
                    common=dict(donor=donor,cytokine=cytokine,heldout_class=cls,resample=resample,generation_seed=0,
                        window_seed=0,
                        eligible=True,evaluator_panel="control_only_fixed",comparison_table=table,
                        manifest_hash=f"{donor}-{cytokine}-{cls}-{resample}",panel_hash="panel",
                        evaluator_hash="evaluator",transport_config_hash="transport" if decoder else "",
                        deg_count=20,deg_set_hash=f"deg-{donor}-{cytokine}-{cls}",
                        decoder_train_ids_hash="train" if decoder else "",expression_basis_hash="basis" if decoder else "",
                        sampling_mode="fine_matched")
                    for metric in metrics:
                        base=1.0 if metric=="energy_distance" else .5
                        # Lower is better only for energy.
                        moved=base-effect if metric=="energy_distance" else base+effect
                        rows += [common|{"method":comparator,"metric":metric,"value":base},common|{"method":method,"metric":metric,"value":moved}]
    return pd.DataFrame(rows)


def test_output_scale_roundtrip_and_anchored_correction():
    counts=np.array([[2,3,5],[0,4,6]],float); libs=library_sizes(counts)
    reconstructed=log1p_to_expected_counts(normalize_log1p(counts),libs)
    assert np.allclose(reconstructed,counts)
    pert=counts+np.array([[2,-1,-1],[3,-2,-1]])
    corrected,diag=synthetic_control_correct(counts,pert,counts,return_diagnostics=True)
    assert np.all(corrected>=0) and np.allclose(corrected.sum(1),libs)
    assert diag["clipped_fraction"]>=0
    assert np.allclose(prepare_for_evaluation(counts),normalize_log1p(counts))


def test_control_only_panel_is_deterministic_and_sealed():
    counts=np.array([[1,0,3],[2,4,0],[3,1,1]]);ids=["a","b","c"];genes=["g1","g2","g3"]
    left=fit_control_only_panel(counts,ids,genes,n_genes=2,sealed_y1=["z"])
    right=fit_control_only_panel(counts,ids,genes,n_genes=2,sealed_y1=["z"])
    assert left.panel_hash==right.panel_hash and len(left.genes)==2
    with pytest.raises(ValueError,match="sealed Y1"):fit_control_only_panel(counts,ids,genes,sealed_y1=["b"])


def test_fine_aliases_and_control_composition_quota():
    assert standardize_fine_label("CD4 T") == "CD4"
    assert standardize_fine_label("CD8-positive T cell") == "CD8"
    assert standardize_fine_label("Dendritic") == "Dendritic"
    assert standardize_fine_label("NK") is None
    quota=_quota_from_control({"Monocyte":996,"Dendritic":4},{"Monocyte":900,"Dendritic":4},128)
    assert quota=={"Monocyte":127,"Dendritic":1}
    assert _quota_from_control({"Monocyte":997,"Dendritic":3},{"Monocyte":900,"Dendritic":3},128) is None


def test_sinkhorn_regularization_has_balanced_marginals():
    rng=np.random.default_rng(3);a=rng.normal(size=(12,4));b=rng.normal(size=(10,4))
    coupling=sinkhorn_coupling(a,b,.05)
    assert np.allclose(coupling.sum(1),1/len(a),atol=2e-5)
    assert np.allclose(coupling.sum(0),1/len(b),atol=2e-5)


def test_matched_decoder_hashes_and_y1_seal():
    rng=np.random.default_rng(4);counts=rng.poisson(3,size=(30,20));stack=rng.normal(size=(30,8));pca=rng.normal(size=(30,6));ids=[f"c{i}" for i in range(30)]
    ds,dp=fit_matched_decoders(stack,pca,counts,train_ids=ids,y1_ids=["sealed"],n_expression_pcs=5)
    assert ds.decoder_train_ids_hash_==dp.decoder_train_ids_hash_
    assert ds.expression_basis_hash_==dp.expression_basis_hash_
    with pytest.raises(ValueError,match="sealed Y1"):
        fit_matched_decoders(stack,pca,counts,train_ids=ids,y1_ids=["c1"],n_expression_pcs=5)


@pytest.mark.parametrize("effect,passes",[(.019999,False),(.02,True)])
def test_numeric_effect_threshold(effect,passes):
    frame=contrast_frame(effect)
    result=contrast_summary(frame,table="end_to_end",method="stack_mmd_native",comparator="stack_icl_native",metric="delta_pearson",n_bootstrap=50)
    assert result["criteria"]["donor_averaged_difference_at_least_0.02"] is passes


def test_missing_technical_pair_fails_and_duplicate_raises():
    frame=contrast_frame()
    missing=frame.drop(frame.index[(frame.method=="stack_icl_native")][0])
    result=contrast_summary(missing,table="end_to_end",method="stack_mmd_native",comparator="stack_icl_native",metric="delta_pearson",n_bootstrap=20)
    assert not result["criteria"]["technical_pairing_complete"]
    duplicate=pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match="duplicate"):
        contrast_summary(duplicate,table="end_to_end",method="stack_mmd_native",comparator="stack_icl_native",metric="delta_pearson",n_bootstrap=20)


def test_complete_four_contrast_gate_and_secondary_panel_cannot_rescue():
    native=contrast_frame(metrics=("delta_pearson","de_lfc_spearman","energy_distance"))
    matched=contrast_frame(table="representation_controlled",method="stack_mmd_matched_ridge",comparator="pca_mmd_matched_ridge",metrics=("delta_pearson","de_lfc_spearman","energy_distance"))
    frame=pd.concat([native,matched],ignore_index=True)
    assert gate(frame,n_bootstrap=30)["passed"]
    fixed=frame.copy(); fixed.loc[(fixed.method=="stack_mmd_native")&(fixed.metric=="delta_pearson"),"value"]-=.1
    secondary=frame.copy();secondary["evaluator_panel"]="q0_y1_hvg"
    assert not gate(pd.concat([fixed,secondary],ignore_index=True),n_bootstrap=30)["passed"]


def test_manifest_audit_locks_global_tier_and_matches_fine_labels(tmp_path):
    labels={"B":520,"CD4 T":340,"CD8 T":240,"Monocyte":500,"Dendritic":20}
    obs=[]
    conditions=("No stimulation",)+CYTOKINES
    for donor in DONORS:
        for condition in conditions:
            for label,n in labels.items():
                obs += [{"sample":donor,"cytokine":condition,"cell_type0528":label} for _ in range(n)]
    frame=pd.DataFrame(obs,index=[f"cell{i}" for i in range(len(obs))])
    counts=np.zeros((len(frame),3),dtype=np.int32);counts[:,0]=1
    lock=build_manifests(ad.AnnData(counts,obs=frame),tmp_path,n_resamples=1)
    assert lock["global_role_size"]==512 and lock["coverage"]["gate_coverage_possible"]
    files=list(tmp_path.glob("*fine_matched*.json"));assert len(files)==36
    import json
    manifest=json.loads(files[0].read_text())
    assert manifest["fine_label_counts"]["p0"]==manifest["fine_label_counts"]["p1"]
    assert manifest["fine_label_counts"]["q0"]==manifest["fine_label_counts"]["y1"]
    assert len(manifest["p0"])==512


def test_stack_window_contract_and_official_schedule():
    p0=[f"a{i}" for i in range(256)];p1=[f"b{i}" for i in range(256)];fine=["CD4"]*128+["CD8"]*128
    windows=build_source_fit_windows(p0,p1,fine,fine,seed=7,sealed_y1=["sealed"])
    assert len(windows)==4 and all(len(w.cell_ids)==512 for w in windows)
    assert all(w.actual_focal_or_query_start==333 and w.query_position_start==332 for w in windows)
    schedule=official_icl_schedule()
    assert [x["source_rows"] for x in schedule]==[231,256,282,308,333]
    assert [x["query_position_start"] for x in schedule]==[230,256,281,307,332]


def test_query_window_seed_changes_query_order():
    context=[f"p{i}" for i in range(333)];query=[f"q{i}" for i in range(256)]
    assert build_query_windows(context,query,seed=1)[0].cell_ids[333:] != build_query_windows(context,query,seed=2)[0].cell_ids[333:]


def test_icl_windows_follow_every_registered_step():
    context=[f"p{i}" for i in range(400)];query=[f"q{i}" for i in range(300)]
    schedule=official_icl_schedule()
    for step, spec in enumerate(schedule, 1):
        windows=build_icl_windows(context,query,step=step,seed=10,sealed_y1=["sealed"])
        assert all(w.source_rows==spec["source_rows"] for w in windows)
        assert all(w.focal_or_query_rows==spec["query_rows"] for w in windows)
        assert all(w.query_position_start==spec["query_position_start"] for w in windows)


def test_manual_stack_forward_intervenes_on_query_only():
    import torch
    class Block(torch.nn.Module):
        def forward(self,x,pos,mask,return_attn):return x+1,None
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(0.));self.query_pos_embedding=torch.nn.Parameter(torch.ones(2,2));self.gene_pos_embedding=torch.nn.Parameter(torch.zeros(2,2));self.layers=torch.nn.ModuleList([Block(),Block()])
        def _reduce_and_tokenize(self,x):return x.reshape(x.shape[0],x.shape[1],2,2)
        def _compute_nb_parameters(self,h,lib):
            scale=torch.softmax(h[...,:4],-1);return scale*lib,torch.ones_like(scale),scale
    model=Fake();counts=torch.ones(1,512,4)
    base=manual_no_mask_forward(model,counts,return_activations=True)
    moved=manual_no_mask_forward(model,counts,intervention_layer=1,intervention=lambda x:x+torch.tensor([1.,0,0,0]))
    assert len(base["activations"])==3
    assert torch.allclose(base["nb_mean"][:,:333],moved["nb_mean"][:,:333])
    assert not torch.allclose(base["nb_mean"][:,333:],moved["nb_mean"][:,333:])
    assert torch.allclose(moved["nb_mean"].sum(-1),counts.sum(-1))
