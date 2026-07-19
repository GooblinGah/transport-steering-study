"""cell-eval the fair Parse predictions (main venv). Every method's control+perturbed are
already in the shared 2000-HVG log-norm space; ours and STATE both went through STATE's
exact gene_decoder. Reuses real DE per cytokine."""
import warnings, logging, os
warnings.filterwarnings("ignore"); logging.disable(logging.INFO); os.environ["TQDM_DISABLE"]="1"
import numpy as np, pandas as pd, anndata as ad
from cell_eval import MetricsEvaluator
from transport_study.evaluation import energy_distance, FixedEvaluatorSpace

SC="/tmp/claude-0/-workspace/de16e521-5da6-4dcb-b532-7b4a0ce6cbd8/scratchpad"
Z=np.load(f"{SC}/parse_fair.npz")
cytos=sorted({k.split("__")[0] for k in Z.files})
def A(ctrl,pert):
    X=np.vstack([ctrl,pert]).astype(np.float32)
    return ad.AnnData(X=X,obs=pd.DataFrame({"perturbation":["control"]*len(ctrl)+["pert"]*len(pert)}),var=pd.DataFrame(index=[f"g{i}" for i in range(2000)]))
rows=[]
for cy in cytos:
    g=lambda k: Z[f"{cy}__{k}"]
    real=A(g("real_ctrl"),g("real_pert")); de_real=None
    ev=FixedEvaluatorSpace(50).fit(np.vstack([g("real_ctrl"),g("real_pert")]))
    for name,ck,pk in [("our_mmd_steering","our_ctrl","our_pert"),("state_transition","state_ctrl","state_pert"),("perturbation_mean_floor","floor_ctrl","floor_pert")]:
        m=MetricsEvaluator(A(g(ck),g(pk)),real.copy(),de_real=de_real,control_pert="control",pert_col="perturbation",de_method="wilcoxon",num_threads=24,outdir=f"{SC}/ce_pf")
        if de_real is None: de_real=m.de_comparison.real.data
        r,_=m.compute(profile="full",skip_metrics=["discrimination_score_l1","discrimination_score_l2","discrimination_score_cosine","pearson_edistance","clustering_agreement"],write_csv=False)
        row=r.to_pandas().set_index("perturbation").loc["pert"]
        rows.append(dict(cytokine=cy,method=name,delta_pearson=float(row.get("pearson_delta")),
                         de_lfc=float(row["de_spearman_lfc_sig"]) if row.get("de_spearman_lfc_sig") is not None and np.isfinite(row.get("de_spearman_lfc_sig")) else np.nan,
                         energy=energy_distance(g(pk),g("real_pert"),ev)))
    print("evaluated",cy,flush=True)
df=pd.DataFrame(rows); df.to_parquet(f"{SC}/parse_fair_metrics.parquet")
print("\n=== FAIR Parse (in-distribution, STATE's own decoder for BOTH), mean over",df.cytokine.nunique(),"cytokines ===")
print(df.groupby("method")[["delta_pearson","de_lfc","energy"]].mean().round(4).to_string())
print("\n=== per-cytokine delta_pearson ===")
print(df.pivot_table(index="cytokine",columns="method",values="delta_pearson").round(3).to_string())
print("DONE")
