import warnings, logging, json, glob, pickle, os
warnings.filterwarnings("ignore"); logging.disable(logging.INFO); os.environ["TQDM_DISABLE"]="1"
import numpy as np, pandas as pd, anndata as ad
from transport_study.run import GeneAligner, evaluate_task, PANEL_GENES
from transport_study.operators import gene_mean_shift_counts, pca_mean_shift_counts
from transport_study.output_scale import prepare_for_evaluation
from transport_study.panels import fit_control_only_panel
from transport_study.evaluation import FixedEvaluatorSpace

stack_genes=[str(g) for g in pickle.load(open("artifacts/models/Stack-Large-Aligned/basecount_1000per_15000max.pkl","rb"))]
obs=pd.read_parquet("artifacts/dong_metadata/obs.parquet"); pos=obs["raw_position"].to_dict()
raw=ad.read_h5ad("data/Integrated_raw.h5ad"); X=raw.X.tocsr()
al=GeneAligner([str(g) for g in raw.var_names], stack_genes)
def cl(c):
    r=pos[c]; row=X[r]; return np.asarray(row.todense()).ravel()

files=sorted(f for f in glob.glob("artifacts/manifests/*fine_matched*r00.json") if json.load(open(f))["eligible"])
rows=[]
for i,f in enumerate(files):
    m=json.load(open(f)); roles={r:list(m[r]) for r in ("p0","p1","q0","y1")}
    ids=sum(roles.values(),[]); aligned=al.align(np.stack([cl(c) for c in ids])); cnt={c:aligned[j] for j,c in enumerate(ids)}
    C=lambda r: np.stack([cnt[c] for c in roles[r]]); p0c,p1c,q0c,y1c=C("p0"),C("p1"),C("q0"),C("y1")
    panel=fit_control_only_panel(np.vstack([p0c,q0c]),roles["p0"]+roles["q0"],[f"g{k}" for k in range(al.n)],n_genes=PANEL_GENES,sealed_y1=roles["y1"])
    pidx=np.array([int(g[1:]) for g in panel.genes])
    ev=FixedEvaluatorSpace(50).fit(prepare_for_evaluation(np.vstack([p0c,p1c,q0c]))[:,pidx],gene_ids=panel.genes)
    preds={"noop":q0c.astype(float),"gene_shift":gene_mean_shift_counts(p0c,p1c,q0c),"pca50_shift":pca_mean_shift_counts(p0c,p1c,q0c)}
    met,_,_=evaluate_task(preds,q0c,y1c,pidx,ev,"/tmp/claude-0/-workspace/de16e521-5da6-4dcb-b532-7b4a0ce6cbd8/scratchpad/ce_base",num_threads=16)
    for name,v in met.items(): rows.append({"task_id":m["task_id"],"cytokine":m["cytokine"],"heldout_class":m["heldout_class"],"method":name,"delta_pearson":v["delta_pearson"],"energy":v["energy_distance"]})
    print(f"[{i+1}/{len(files)}] {m['task_id']}", flush=True)
pd.DataFrame(rows).to_parquet("/tmp/claude-0/-workspace/de16e521-5da6-4dcb-b532-7b4a0ce6cbd8/scratchpad/baseline.parquet")
print("DONE")
