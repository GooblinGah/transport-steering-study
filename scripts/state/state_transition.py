"""state_transition: STATE's native ST model prediction, decoded by the SAME matched
ridge as state_mmd. Only the transformation differs (our MMD vs their trained ST) -> fair."""
import warnings, logging, json, glob, pickle, os
warnings.filterwarnings("ignore"); logging.disable(logging.INFO); os.environ["TQDM_DISABLE"]="1"
import numpy as np, pandas as pd, anndata as ad
from transport_study.run import GeneAligner, evaluate_task, PANEL_GENES
from transport_study.decoders import MatchedRidgeDecoder
from transport_study.output_scale import library_sizes, prepare_for_evaluation, synthetic_control_correct
from transport_study.panels import fit_control_only_panel
from transport_study.evaluation import FixedEvaluatorSpace

SCRATCH="/tmp/claude-0/-workspace/de16e521-5da6-4dcb-b532-7b4a0ce6cbd8/scratchpad"
CYTO={"IFN-α2":"IFN-alpha1","IFN-β":"IFN-beta","IFN-γ":"IFN-gamma","IFN-III/IL-29":"IFN-lambda1","IL-6":"IL-6","TNF-α":"TNF-alpha"}

# ST predictions: (orig_id, cytokine) -> predicted perturbed SE embedding
stp=ad.read_h5ad(f"{SCRATCH}/st_pred_all.h5ad")
PRED={}
oid=stp.obs["orig_id"].to_numpy(); cyt=stp.obs["cytokine"].to_numpy(); Xs=stp.obsm["X_state"]
for i in range(stp.n_obs): PRED[(str(oid[i]),str(cyt[i]))]=Xs[i]
print("ST prediction entries:", len(PRED))

se=ad.read_h5ad(f"{SCRATCH}/state_cells_emb.h5ad"); SE={str(c):se.obsm["X_state"][i] for i,c in enumerate(se.obs_names)}
stack_genes=[str(g) for g in pickle.load(open("artifacts/models/Stack-Large-Aligned/basecount_1000per_15000max.pkl","rb"))]
obs=pd.read_parquet("artifacts/dong_metadata/obs.parquet"); pos=obs["raw_position"].to_dict()
raw=ad.read_h5ad("data/Integrated_raw.h5ad"); X=raw.X.tocsr(); al=GeneAligner([str(g) for g in raw.var_names],stack_genes)
def cl(c): return np.asarray(X[pos[c]].todense()).ravel()

files=sorted(f for f in glob.glob("artifacts/manifests/*fine_matched*r00.json") if json.load(open(f))["eligible"])
rows=[]; skipped=0
for i,f in enumerate(files):
    m=json.load(open(f)); roles={r:list(m[r]) for r in ("p0","p1","q0","y1")}; mc=CYTO.get(m["cytokine"])
    if any((c,mc) not in PRED or (c,"PBS") not in PRED for c in roles["q0"]): skipped+=1; print("skip",m["task_id"]); continue
    ids=sum(roles.values(),[]); aligned=al.align(np.stack([cl(c) for c in ids])); cnt={c:aligned[j] for j,c in enumerate(ids)}
    C=lambda r: np.stack([cnt[c] for c in roles[r]]); p0c,p1c,q0c,y1c=C("p0"),C("p1"),C("q0"),C("y1")
    se_p0,se_p1,se_q0=(np.stack([SE[c] for c in roles[r]]) for r in ("p0","p1","q0"))
    st_pert=np.stack([PRED[(c,mc)] for c in roles["q0"]]); st_ctrl=np.stack([PRED[(c,"PBS")] for c in roles["q0"]])
    dec=MatchedRidgeDecoder().fit_expression_basis(np.vstack([p0c,p1c,q0c])).fit_representation(np.vstack([se_p0,se_p1,se_q0]))
    q0_lib=library_sizes(q0c)
    pred=synthetic_control_correct(q0c, dec.decode_expected_counts(st_pert,q0_lib), dec.decode_expected_counts(st_ctrl,q0_lib))
    panel=fit_control_only_panel(np.vstack([p0c,q0c]),roles["p0"]+roles["q0"],[f"g{k}" for k in range(al.n)],n_genes=PANEL_GENES,sealed_y1=roles["y1"]); pidx=np.array([int(g[1:]) for g in panel.genes])
    ev=FixedEvaluatorSpace(50).fit(prepare_for_evaluation(np.vstack([p0c,p1c,q0c]))[:,pidx],gene_ids=panel.genes)
    met,_,_=evaluate_task({"state_transition":pred}, q0c,y1c,pidx,ev,f"{SCRATCH}/ce_sttx",num_threads=16)
    for name,v in met.items():
        base=dict(task_id=m["task_id"],donor=m["donor"],cytokine=m["cytokine"],heldout_class=m["heldout_class"],method=name,resample=0)
        for metric in ("delta_pearson","de_lfc_spearman","energy_distance"): rows.append(base|{"metric":metric,"value":v[metric]})
    print(f"[{i+1}/{len(files)}] {m['task_id']}",flush=True)
pd.DataFrame(rows).to_parquet(f"{SCRATCH}/state_transition.parquet"); print("DONE rows",len(rows),"skipped",skipped)
