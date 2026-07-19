"""STATE-SE steering (state_mmd): identical LowRankMMD math as Stack, on SE embeddings,
decoded by the same matched ridge decoder. Evaluated on the same panel with cell-eval,
so state_mmd is directly comparable to stack_mmd_matched / pca_mmd_matched."""
import warnings, logging, json, glob, pickle, os, sys
warnings.filterwarnings("ignore"); logging.disable(logging.INFO); os.environ["TQDM_DISABLE"]="1"
import numpy as np, pandas as pd, anndata as ad, torch
from transport_study.run import GeneAligner, evaluate_task, PANEL_GENES, MMD_CONFIG
from transport_study.operators import LowRankMMD
from transport_study.decoders import MatchedRidgeDecoder
from transport_study.output_scale import library_sizes, prepare_for_evaluation, synthetic_control_correct
from transport_study.panels import fit_control_only_panel
from transport_study.evaluation import FixedEvaluatorSpace

SCRATCH="/tmp/claude-0/-workspace/de16e521-5da6-4dcb-b532-7b4a0ce6cbd8/scratchpad"
se=ad.read_h5ad(f"{SCRATCH}/state_cells_emb.h5ad")
SE={str(c):se.obsm["X_state"][i] for i,c in enumerate(se.obs_names)}
print("SE embeddings loaded:", se.obsm["X_state"].shape)

stack_genes=[str(g) for g in pickle.load(open("artifacts/models/Stack-Large-Aligned/basecount_1000per_15000max.pkl","rb"))]
obs=pd.read_parquet("artifacts/dong_metadata/obs.parquet"); pos=obs["raw_position"].to_dict()
raw=ad.read_h5ad("data/Integrated_raw.h5ad"); X=raw.X.tocsr()
al=GeneAligner([str(g) for g in raw.var_names], stack_genes)
def cl(c): r=pos[c]; row=X[r]; return np.asarray(row.todense()).ravel()
dev="cuda"; to_dev=lambda x: torch.as_tensor(np.asarray(x),dtype=torch.float32,device=dev)

files=sorted(f for f in glob.glob("artifacts/manifests/*fine_matched*r00.json") if json.load(open(f))["eligible"])
rows=[]
for i,f in enumerate(files):
    m=json.load(open(f)); roles={r:list(m[r]) for r in ("p0","p1","q0","y1")}
    ids=sum(roles.values(),[]); aligned=al.align(np.stack([cl(c) for c in ids])); cnt={c:aligned[j] for j,c in enumerate(ids)}
    C=lambda r: np.stack([cnt[c] for c in roles[r]]); p0c,p1c,q0c,y1c=C("p0"),C("p1"),C("q0"),C("y1")
    se_p0,se_p1,se_q0=(np.stack([SE[c] for c in roles[r]]) for r in ("p0","p1","q0"))
    q0_lib=library_sizes(q0c)
    mmd=LowRankMMD(**MMD_CONFIG).fit(to_dev(se_p0),to_dev(se_p1)); tq=mmd.transform(se_q0)
    dec=MatchedRidgeDecoder().fit_expression_basis(np.vstack([p0c,p1c,q0c])).fit_representation(np.vstack([se_p0,se_p1,se_q0]))
    pred=synthetic_control_correct(q0c, dec.decode_expected_counts(tq,q0_lib), dec.decode_expected_counts(se_q0,q0_lib))
    panel=fit_control_only_panel(np.vstack([p0c,q0c]),roles["p0"]+roles["q0"],[f"g{k}" for k in range(al.n)],n_genes=PANEL_GENES,sealed_y1=roles["y1"])
    pidx=np.array([int(g[1:]) for g in panel.genes])
    ev=FixedEvaluatorSpace(50).fit(prepare_for_evaluation(np.vstack([p0c,p1c,q0c]))[:,pidx],gene_ids=panel.genes)
    met,_,_=evaluate_task({"state_mmd_matched":pred}, q0c,y1c,pidx,ev,f"{SCRATCH}/ce_state",num_threads=16)
    for name,v in met.items():
        base=dict(task_id=m["task_id"],donor=m["donor"],cytokine=m["cytokine"],heldout_class=m["heldout_class"],method=name,resample=0)
        for metric in ("delta_pearson","de_lfc_spearman","energy_distance"): rows.append(base|{"metric":metric,"value":v[metric]})
    print(f"[{i+1}/{len(files)}] {m['task_id']}",flush=True)
pd.DataFrame(rows).to_parquet(f"{SCRATCH}/state_mmd.parquet"); print("DONE", len(rows))
