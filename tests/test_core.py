import numpy as np
import pandas as pd
import pytest
from transport_study.contracts import TaskManifest, validate_manifest_payload
from transport_study.manifests import assert_raw_counts
from transport_study.operators import LowRankMMD, mean_shift
import anndata as ad

def manifest():
    ids=lambda x: tuple(f"{x}{i}" for i in range(128))
    return TaskManifest("H2D2","IL-6","B",ids("a"),ids("b"),ids("c"),ids("d"),{},("g1","g2"),1,0)

def test_manifest_hash_and_sealing():
    m=manifest(); m.validate(); assert len(m.manifest_hash)==64
    bad=TaskManifest(**(m.payload()|{"y1":m.q0}))
    with pytest.raises(ValueError,match="overlap"): bad.validate()

def test_loaded_manifest_is_authenticated():
    m=manifest(); body=m.payload()|{"task_id":m.task_id,"manifest_hash":m.manifest_hash}
    assert validate_manifest_payload(body)==m
    body["q0"]=["tampered"]*128
    with pytest.raises(ValueError): validate_manifest_payload(body)

def test_raw_counts_guard():
    assert_raw_counts(ad.AnnData(np.array([[0,1],[2,3]],dtype=float)))
    with pytest.raises(ValueError,match="integer"): assert_raw_counts(ad.AnnData(np.array([[.2,1]])))

def test_alpha_zero_is_noop():
    rng=np.random.default_rng(2); a=rng.normal(size=(32,8)); b=a+1; q=rng.normal(size=(11,8))
    assert np.array_equal(mean_shift(a,b,q,alpha=0),q)
    model=LowRankMMD(steps=5,seed=2).fit(a,b)
    assert np.allclose(model.transform(q,alpha=0),q,atol=1e-5)

def test_mmd_source_loss_improves():
    rng=np.random.default_rng(4); a=rng.normal(size=(80,6)); b=a+np.array([1,0,0,0,0,0])
    model=LowRankMMD(steps=100,seed=4).fit(a,b)
    assert model.loss_after_ < model.loss_before_
