"""In-distribution Parse comparison (STATE's home turf): held-out cell type B_Intermediate_Memory.
Our MMD steering on SE embeddings vs STATE's OWN transition predictions vs the perturbation-mean
floor -- all evaluated against real Parse perturbed cells in the shared 2000-HVG space.
Genes are anonymized but consistent across every method, so cell-eval works directly."""
import warnings, logging, os
warnings.filterwarnings("ignore"); logging.disable(logging.INFO); os.environ["TQDM_DISABLE"]="1"
import numpy as np, pandas as pd, anndata as ad
from sklearn.linear_model import Ridge
from cell_eval import MetricsEvaluator
from transport_study.operators import LowRankMMD
from transport_study.run import MMD_CONFIG
from transport_study.evaluation import energy_distance, FixedEvaluatorSpace
import torch

SC="/tmp/claude-0/-workspace/de16e521-5da6-4dcb-b532-7b4a0ce6cbd8/scratchpad"
D="artifacts/models/ST-SE-Parse/zeroshot"
tgt=ad.read_h5ad(f"{D}/split_0/eval_best.ckpt/adata_real.h5ad")   # B_Int_Memory real (Q0/Y1)
src=ad.read_h5ad(f"{D}/split_4/eval_best.ckpt/adata_real.h5ad")   # CD14_Mono real (P0/P1 source)
stp=ad.read_h5ad(f"{D}/split_0/eval_best.ckpt/adata_pred.h5ad")   # STATE's B_Int_Memory prediction
def dense(a): return np.asarray(a.todense()) if hasattr(a,"todense") else np.asarray(a)

# Strong SE->2000HVG ridge decoder for OUR steering (fit on a large pooled Parse sample).
rng=np.random.default_rng(0)
pool=[];
for A in (tgt,src):
    idx=rng.choice(A.n_obs, min(4000,A.n_obs), replace=False); pool.append((A.obsm["X_state"][idx], dense(A.X[idx])))
Xse=np.vstack([p[0] for p in pool]); Xhvg=np.vstack([p[1] for p in pool])
ridge=Ridge(alpha=10.0).fit(Xse, Xhvg)
print("ridge SE->2000HVG R2(train):", round(ridge.score(Xse,Xhvg),3))

CYTOS=["IFN-alpha1","IFN-beta","IFN-gamma","IFN-lambda1","IL-6","TNF-alpha","IL-2","IL-4","IL-10","IFN-omega","IL-21","GM-CSF"]
N=120; dev="cuda"; to=lambda x: torch.as_tensor(np.asarray(x),dtype=torch.float32,device=dev)
def samp(A,cy,n):
    i=np.where(A.obs.cytokine.values==cy)[0]; i=rng.choice(i,min(n,len(i)),replace=False); return i
def celleval(pred, q0, y1, ev):
    def A(ctrl,pert):
        X=np.vstack([ctrl,pert]).astype(np.float32)
        return ad.AnnData(X=X, obs=pd.DataFrame({"perturbation":["control"]*len(ctrl)+["pert"]*len(pert)}), var=pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])]))
    m=MetricsEvaluator(A(q0,pred),A(q0,y1),control_pert="control",pert_col="perturbation",de_method="wilcoxon",num_threads=16,outdir=f"{SC}/ce_parse")
    r,_=m.compute(profile="full",skip_metrics=["discrimination_score_l1","discrimination_score_l2","discrimination_score_cosine","pearson_edistance","clustering_agreement"],write_csv=False)
    row=r.to_pandas().set_index("perturbation").loc["pert"]
    return row.get("pearson_delta"), row.get("de_spearman_lfc_sig"), energy_distance(pred,y1,ev)

rows=[]
for cy in CYTOS:
    qi=samp(tgt,"PBS",N); yi=samp(tgt,cy,N); p0i=samp(src,"PBS",N); p1i=samp(src,cy,N); si=samp(stp,cy,N)
    if min(len(qi),len(yi),len(p0i),len(p1i),len(si))<20: print("skip",cy); continue
    Q0=dense(tgt.X[qi]); Y1=dense(tgt.X[yi]); P0=dense(src.X[p0i]); P1=dense(src.X[p1i])
    seP0,seP1,seQ0=src.obsm["X_state"][p0i],src.obsm["X_state"][p1i],tgt.obsm["X_state"][qi]
    ev=FixedEvaluatorSpace(50).fit(np.vstack([P0,P1,Q0]))
    # our steering (anchored in lognorm 2000HVG via ridge)
    tq=LowRankMMD(**MMD_CONFIG).fit(to(seP0),to(seP1)).transform(seQ0)
    our=np.maximum(Q0+(ridge.predict(tq)-ridge.predict(seQ0)),0)
    # STATE native + perturbation-mean floor
    state=dense(stp.X[si]); floor=np.maximum(Q0+(P1.mean(0)-P0.mean(0)),0)
    for name,pred in [("state_transition",state),("our_mmd_steering",our),("perturbation_mean_floor",floor)]:
        dp,lfc,en=celleval(pred,Q0,Y1,ev)
        rows.append(dict(cytokine=cy,method=name,delta_pearson=dp,de_lfc=lfc,energy=en))
    print(f"done {cy}",flush=True)
df=pd.DataFrame(rows); df.to_parquet(f"{SC}/parse_compare.parquet")
print("\n=== Parse (in-distribution), mean over",df.cytokine.nunique(),"cytokines ===")
print(df.groupby("method")[["delta_pearson","de_lfc","energy"]].mean().round(4).to_string())
print("DONE")
