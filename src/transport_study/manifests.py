from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np
import anndata as ad
from scipy import sparse
from .contracts import TaskManifest

DONORS = ("H2D2", "H3D2")
CYTOKINES = ("IFN-α2", "IFN-β", "IFN-γ", "IFN-III/IL-29", "IL-6", "TNF-α")
CLASSES = ("B", "T", "Myeloid")
FINE_BY_CLASS = {"B": ("B",), "T": ("CD4", "CD8"), "Myeloid": ("Monocyte", "Dendritic")}
ROLE_SIZES = (512, 256, 128)
MIN_AVAILABLE_FINE = 4
MIN_SELECTED_FINE = 1


def assert_raw_counts(adata: ad.AnnData) -> None:
    x = adata.X
    values = x.data if sparse.issparse(x) else np.asarray(x).ravel()
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("X must contain finite nonnegative raw counts")
    if values.size and not np.allclose(values, np.rint(values)):
        raise ValueError("X is not integer-valued raw counts")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("cell and gene identifiers must be unique")


def standardize_fine_label(value: str) -> str | None:
    """Apply the registered fine-label map. Return no label for NK and plasma cells."""
    text=re.sub(r"[^a-z0-9]+"," ",str(value).lower()).strip()
    if "plasma" in text or re.search(r"\bnk\b|natural killer",text): return None
    if re.search(r"\bcd4\b",text): return "CD4"
    if re.search(r"\bcd8\b",text): return "CD8"
    if "dendritic" in text or re.search(r"(^| )dc($| )",text): return "Dendritic"
    if "mono" in text: return "Monocyte"
    if re.search(r"(^| )b( cell| lymphocyte|$)",text): return "B"
    return None


def _quota_from_control(left_counts: dict[str,int], capacities: dict[str,int], total: int):
    """Make the nearest integer quota to the control composition. Treat treated-cell counts as limits."""
    labels=tuple(capacities)
    if any(left_counts[k]<MIN_AVAILABLE_FINE or capacities[k]<MIN_AVAILABLE_FINE for k in labels): return None
    if sum(capacities.values())<total or total<MIN_SELECTED_FINE*len(labels): return None
    if len(labels)==1: return {labels[0]:total} if capacities[labels[0]]>=total else None
    if len(labels)!=2: raise ValueError("quota solver is preregistered for one or two fine labels")
    a,b=labels; prop=left_counts[a]/max(sum(left_counts.values()),1)
    lower=max(MIN_SELECTED_FINE,total-capacities[b]); upper=min(capacities[a],total-MIN_SELECTED_FINE)
    if lower>upper:return None
    qa=int(np.clip(round(total*prop),lower,upper))
    return {a:qa,b:total-qa}


def _sample_strata(rng, frame, quotas):
    out=[]
    for fine,n in quotas.items():
        ids=frame.index[frame["_fine"]==fine].to_numpy()
        if len(ids)<n: raise ValueError(f"{fine} has {len(ids)}, needs {n}")
        out.extend(rng.choice(ids,n,replace=False).tolist())
    return tuple(sorted(out))


def _matched_quotas(left, right, broad, total):
    fine=FINE_BY_CLASS[broad]
    left_counts={f:int((left["_fine"]==f).sum()) for f in fine}
    caps={f:min(int((left["_fine"]==f).sum()),int((right["_fine"]==f).sum())) for f in fine}
    return _quota_from_control(left_counts,caps,total),caps


def _task_pools(obs, donor, cytokine, donor_col, condition_col, control):
    base=obs[donor_col].astype(str)==donor
    return {
        "p0_all":obs[base&(obs[condition_col].astype(str)==control)],
        "p1_all":obs[base&(obs[condition_col].astype(str)==cytokine)],
    }


def audit_task(obs, donor, cytokine, heldout, donor_col, condition_col, control):
    pools=_task_pools(obs,donor,cytokine,donor_col,condition_col,control)
    source=tuple(c for c in CLASSES if c!=heldout)
    raw={role:{f:int((frame["_fine"]==f).sum()) for f in sum((FINE_BY_CLASS[c] for c in CLASSES),())} for role,frame in pools.items()}
    chosen=None; reason=""; all_quotas={}; candidates={}
    for n in ROLE_SIZES:
        if n%len(source): continue
        candidate={}; ok=True
        for broad in source:
            left=pools["p0_all"][pools["p0_all"]["_broad"]==broad]; right=pools["p1_all"][pools["p1_all"]["_broad"]==broad]
            q,_=_matched_quotas(left,right,broad,n//len(source)); candidate[f"source:{broad}"]=q
            if q is None: ok=False
        q0=pools["p0_all"][pools["p0_all"]["_broad"]==heldout]; y1=pools["p1_all"][pools["p1_all"]["_broad"]==heldout]
        q,_=_matched_quotas(q0,y1,heldout,n); candidate["target"]=q
        if q is None: ok=False
        candidates[str(n)]={"eligible":bool(ok),"matched_quotas":candidate if ok else {},"reason":"" if ok else "insufficient fine-stratum availability/capacity"}
        if ok and chosen is None: chosen=n; all_quotas=candidate
    if chosen is None: reason=f"no fine-matched role size >=128 with raw availability >= {MIN_AVAILABLE_FINE} per required fine label"
    return {"donor":donor,"cytokine":cytokine,"heldout_class":heldout,"eligible":chosen is not None,"role_size":chosen,"reason":reason,"raw_fine_counts":raw,"matched_quotas":all_quotas,"candidates":candidates}


def _sample_matched(obs,audit,rng,donor_col,condition_col,control):
    donor,cytokine,heldout=audit["donor"],audit["cytokine"],audit["heldout_class"]
    pools=_task_pools(obs,donor,cytokine,donor_col,condition_col,control); source=tuple(c for c in CLASSES if c!=heldout)
    p0=[]; p1=[]
    for broad in source:
        quotas=audit["matched_quotas"][f"source:{broad}"]
        p0.extend(_sample_strata(rng,pools["p0_all"][pools["p0_all"]["_broad"]==broad],quotas))
        p1.extend(_sample_strata(rng,pools["p1_all"][pools["p1_all"]["_broad"]==broad],quotas))
    quotas=audit["matched_quotas"]["target"]
    q0=_sample_strata(rng,pools["p0_all"][pools["p0_all"]["_broad"]==heldout],quotas)
    y1=_sample_strata(rng,pools["p1_all"][pools["p1_all"]["_broad"]==heldout],quotas)
    return tuple(sorted(p0)),tuple(sorted(p1)),q0,y1


def _sample_natural(obs,audit,rng,donor_col,condition_col,control):
    donor,cytokine,heldout,n=audit["donor"],audit["cytokine"],audit["heldout_class"],audit["role_size"]
    pools=_task_pools(obs,donor,cytokine,donor_col,condition_col,control); source=tuple(c for c in CLASSES if c!=heldout)
    def take(frame,count):
        ids=frame.index.to_numpy()
        if len(ids)<count: raise ValueError("natural-proportion pool too small")
        return rng.choice(ids,count,replace=False).tolist()
    # Keep source labels matched. Keep natural proportions only in held-out marginals.
    p0=[];p1=[]
    for broad in source:
        quotas=audit["matched_quotas"][f"source:{broad}"]
        p0+=list(_sample_strata(rng,pools["p0_all"][pools["p0_all"]["_broad"]==broad],quotas))
        p1+=list(_sample_strata(rng,pools["p1_all"][pools["p1_all"]["_broad"]==broad],quotas))
    return tuple(sorted(p0)),tuple(sorted(p1)),tuple(sorted(take(pools["p0_all"][pools["p0_all"]["_broad"]==heldout],n))),tuple(sorted(take(pools["p1_all"][pools["p1_all"]["_broad"]==heldout],n)))


def _counts(obs,ids,column): return {str(k):int(v) for k,v in obs.loc[list(ids),column].value_counts().sort_index().items()}


def build_manifests(adata,out:Path,*,donor_col="sample",condition_col="cytokine",celltype_col="cell_type0528",control="No stimulation",n_resamples=20,seed=1729):
    assert_raw_counts(adata); obs=adata.obs.copy()
    for col in (donor_col,condition_col,celltype_col):
        if col not in obs: raise ValueError(f"missing obs column {col!r}; available={list(obs.columns)}")
    obs["_fine"]=obs[celltype_col].map(standardize_fine_label)
    broad={f:c for c,labels in FINE_BY_CLASS.items() for f in labels}; obs["_broad"]=obs["_fine"].map(broad)
    genes=tuple(map(str,adata.var_names)); out.mkdir(parents=True,exist_ok=True)
    audits=[audit_task(obs,d,c,h,donor_col,condition_col,control) for d in DONORS for c in CYTOKINES for h in CLASSES]
    tier_coverage={}; global_size=None
    for n in ROLE_SIZES:
        units=[]
        for c in CYTOKINES:
            for h in CLASSES:
                pair=[a for a in audits if a["cytokine"]==c and a["heldout_class"]==h]
                if len(pair)==2 and all(a["candidates"][str(n)]["eligible"] for a in pair):units.append((c,h))
        per_class={h:sum(xh==h for _,xh in units) for h in CLASSES}; per_cytokine={c:sum(xc==c for xc,_ in units) for c in CYTOKINES}
        ok=len(units)>=12 and all(v>=4 for v in per_class.values()) and all(v>=2 for v in per_cytokine.values())
        tier_coverage[str(n)]={"paired_units":units,"n_units":len(units),"per_class":per_class,"per_cytokine":per_cytokine,"passes":ok}
        if ok and global_size is None:global_size=n
    for audit in audits:
        if global_size is None:
            audit.update(eligible=False,role_size=None,matched_quotas={},reason="no preregistered global role-size tier passes coverage")
        else:
            choice=audit["candidates"][str(global_size)]
            audit.update(eligible=choice["eligible"],role_size=global_size,matched_quotas=choice["matched_quotas"],reason=choice["reason"])
    (out/"CELL_COUNT_AUDIT.json").write_text(json.dumps({"global_role_size":global_size,"tier_coverage":tier_coverage,"tasks":audits},indent=2,sort_keys=True)+"\n")
    eligible_units=[]
    for c in CYTOKINES:
        for h in CLASSES:
            pair=[a for a in audits if a["cytokine"]==c and a["heldout_class"]==h]
            if len(pair)==2 and all(a["eligible"] for a in pair): eligible_units.append((c,h))
    coverage={"eligible_donor_tasks":sum(a["eligible"] for a in audits),"total_donor_tasks":36,"eligible_two_donor_units":len(eligible_units),"total_units":18,"every_cytokine":set(c for c,_ in eligible_units)==set(CYTOKINES),"every_class":set(h for _,h in eligible_units)==set(CLASSES),"minimum_units":12}
    coverage["gate_coverage_possible"]=coverage["eligible_two_donor_units"]>=12 and coverage["every_cytokine"] and coverage["every_class"]
    written=0
    for resample in range(n_resamples):
        for audit in audits:
            d,c,h=audit["donor"],audit["cytokine"],audit["heldout_class"]
            task_seed=seed+resample*1000+DONORS.index(d)*100+CYTOKINES.index(c)*10+CLASSES.index(h)
            for mode_i,mode in enumerate(("fine_matched","natural_proportion")):
                rng=np.random.default_rng(task_seed+mode_i*100_000)
                if audit["eligible"]:
                    roles=_sample_matched(obs,audit,rng,donor_col,condition_col,control) if mode=="fine_matched" else _sample_natural(obs,audit,rng,donor_col,condition_col,control)
                else: roles=((),(),(),())
                p0,p1,q0,y1=roles
                manifest=TaskManifest(d,c,h,p0,p1,q0,y1,
                    {r:_counts(obs,ids,"_broad") for r,ids in zip(("p0","p1"),roles[:2])},genes,task_seed,resample,
                    fine_label_counts={r:_counts(obs,ids,"_fine") for r,ids in zip(("p0","p1","q0","y1"),roles)},
                    sampling_mode=mode,eligible=bool(audit["eligible"]),eligibility_reason=audit["reason"],
                    preprocessing={"input":"raw_counts","fine_label_column":celltype_col,"minimum_raw_fine_availability":MIN_AVAILABLE_FINE,"minimum_selected_fine_cells":MIN_SELECTED_FINE,"y1_label_use":"stratification_only"})
                manifest.write(out/f"{manifest.task_id}.json"); written+=1
    lock={"donor_tasks":36,"global_role_size":global_size,"tier_coverage":tier_coverage,"sampling_modes":["fine_matched","natural_proportion"],"resamples":n_resamples,"manifest_count":written,"seed":seed,"coverage":coverage,"fine_mapping":FINE_BY_CLASS,"excluded":["NK","plasma"]}
    (out/"LOCK.json").write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n")
    return lock
