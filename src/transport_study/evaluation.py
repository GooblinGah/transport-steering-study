from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from hashlib import sha256
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

EXPECTED_CYTOKINES = {"IFN-α2", "IFN-β", "IFN-γ", "IFN-III/IL-29", "IL-6", "TNF-α"}
EXPECTED_CLASSES = {"B", "T", "Myeloid"}
EXPECTED_DONORS = {"H2D2", "H3D2"}
GATE_PANEL = "control_only_fixed"
CONTRAST_SPECS = (
    ("end_to_end", "stack_mmd_native", "stack_icl_native"),
    ("representation_controlled", "stack_mmd_matched_ridge", "pca_mmd_matched_ridge"),
)


def delta_metrics(pred, q0, y1):
    dp=np.mean(pred,0)-np.mean(q0,0); dy=np.mean(y1,0)-np.mean(q0,0)
    pearson=float(pearsonr(dp,dy).statistic) if np.std(dp)>0 and np.std(dy)>0 else np.nan
    spearman=float(spearmanr(dp,dy).statistic) if np.std(dp)>0 and np.std(dy)>0 else np.nan
    return {"delta_pearson":pearson,"delta_spearman":spearman,"delta_r2":float(1-np.sum((dp-dy)**2)/max(np.sum((dy-dy.mean())**2),1e-12))}


class FixedEvaluatorSpace:
    """Make one control-only evaluator basis. Use it for all methods."""
    def __init__(self,n_components=50): self.n_components=n_components
    def fit(self,fit_expression,gene_ids=()):
        x=np.asarray(fit_expression); self.pca=PCA(min(self.n_components,len(x)-1,x.shape[1]),random_state=0).fit(x)
        payload=np.ascontiguousarray(self.pca.components_).tobytes()+"\0".join(map(str,gene_ids)).encode()
        self.evaluator_hash=sha256(payload).hexdigest(); return self
    def transform(self,x): return self.pca.transform(x)


def energy_distance(pred, y1, evaluator_fit):
    evaluator=evaluator_fit if isinstance(evaluator_fit,FixedEvaluatorSpace) else FixedEvaluatorSpace().fit(evaluator_fit)
    x,y=evaluator.transform(pred),evaluator.transform(y1)
    return float(2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean())


def _require_columns(frame: pd.DataFrame) -> None:
    required={"donor","cytokine","heldout_class","method","metric","value","eligible","evaluator_panel","comparison_table","resample","generation_seed","window_seed","manifest_hash","panel_hash","evaluator_hash","deg_count","deg_set_hash","decoder_train_ids_hash","expression_basis_hash","transport_config_hash"}
    missing=required-set(frame.columns)
    if missing:
        raise ValueError(f"metric records missing columns: {sorted(missing)}")


def average_technical(records: pd.DataFrame) -> pd.DataFrame:
    """Average technical resamples and seeds before a biological contrast."""
    _require_columns(records)
    frame=records.loc[records["eligible"].astype(bool)].copy()
    keys=["donor","cytokine","heldout_class","method","metric","evaluator_panel","comparison_table"]
    return frame.groupby(keys,as_index=False,observed=True)["value"].mean()


def paired_technical_differences(records, *, table, method, comparator, metric, panel):
    """Pair technical records first. Then, average their differences."""
    _require_columns(records)
    part=records.loc[
        records["eligible"].astype(bool)&(records.comparison_table==table)&
        (records.metric==metric)&(records.evaluator_panel==panel)&records.method.isin([method,comparator])
    ].copy()
    if metric=="de_lfc_spearman":
        part=part.loc[(part.deg_count.fillna(0)>=10)&(part.deg_set_hash.astype(str)!="")]
    keys=["donor","cytokine","heldout_class","resample","generation_seed","window_seed","manifest_hash","panel_hash","evaluator_hash"]
    if metric=="de_lfc_spearman":keys.append("deg_set_hash")
    if table=="representation_controlled":
        if (part.decoder_train_ids_hash.astype(str)=="").any() or (part.expression_basis_hash.astype(str)=="").any() or (part.transport_config_hash.astype(str)=="").any():
            return pd.Series(dtype=float,name="difference"),False
        keys += ["decoder_train_ids_hash","expression_basis_hash","transport_config_hash"]
    if part.duplicated(keys+["method"]).any():
        raise ValueError("duplicate metric record for one technical key/method")
    pivot=part.pivot(index=keys,columns="method",values="value")
    pairing_complete=method in pivot and comparator in pivot and not pivot[[method,comparator]].isna().any().any()
    if method not in pivot or comparator not in pivot:
        return pd.Series(dtype=float,name="difference"),False
    diff=(pivot[method]-pivot[comparator]).dropna()
    biological=diff.groupby(level=["donor","cytokine","heldout_class"]).mean()
    return biological,pairing_complete


def _difference_grid(averaged, table, method, comparator, metric, panel):
    part=averaged.loc[
        (averaged.comparison_table==table)&(averaged.metric==metric)&
        (averaged.evaluator_panel==panel)&averaged.method.isin([method,comparator])
    ]
    pivot=part.pivot(index=["donor","cytokine","heldout_class"],columns="method",values="value")
    if method not in pivot or comparator not in pivot:
        return pd.Series(dtype=float,name="difference")
    return (pivot[method]-pivot[comparator]).dropna().rename("difference")


def descriptive_two_way_bootstrap(differences: pd.Series, n=10_000, seed=1729):
    """Calculate a descriptive interval across cytokine and class clusters. Keep donors fixed."""
    cytokines=sorted(set(differences.index.get_level_values("cytokine")))
    classes=sorted(set(differences.index.get_level_values("heldout_class")))
    donors=sorted(set(differences.index.get_level_values("donor")))
    arr=np.full((len(donors),len(cytokines),len(classes)),np.nan)
    for di,d in enumerate(donors):
        for ci,c in enumerate(cytokines):
            for ki,k in enumerate(classes):
                try: arr[di,ci,ki]=differences.loc[(d,c,k)]
                except KeyError: pass
    rng=np.random.default_rng(seed); draws=[]
    for _ in range(n):
        rows=rng.integers(0,len(cytokines),len(cytokines)); cols=rng.integers(0,len(classes),len(classes))
        sample=arr[:,rows][:,:,cols]
        if np.isfinite(sample).any(): draws.append(float(np.nanmean(sample)))
    if not draws:
        return {"lower_95":None,"upper_95":None,"bootstrap_kind":"descriptive_two_way_cluster"}
    return {"lower_95":float(np.quantile(draws,.025)),"upper_95":float(np.quantile(draws,.975)),"bootstrap_kind":"descriptive_two_way_cluster"}


def contrast_summary(averaged, *, table, method, comparator, metric, panel=GATE_PANEL, minimum_effect=.02, minimum_wins=12, n_bootstrap=10_000):
    diff,pairing_complete=paired_technical_differences(averaged,table=table,method=method,comparator=comparator,metric=metric,panel=panel)
    # A unit is eligible when both donor records contain the method pair.
    counts=diff.groupby(level=["cytokine","heldout_class"]).count() if len(diff) else pd.Series(dtype=int)
    units=diff.groupby(level=["cytokine","heldout_class"]).mean()[counts==len(EXPECTED_DONORS)] if len(diff) else pd.Series(dtype=float)
    valid=set(units.index.tolist()) if len(units) else set()
    joint=diff[[tuple(x[1:]) in valid for x in diff.index]] if len(diff) else diff
    donors=joint.groupby(level="donor").mean() if len(joint) else pd.Series(dtype=float)
    observed=float(units.mean()) if len(units) else None
    cytokines=set(units.index.get_level_values("cytokine")) if len(units) else set()
    classes=set(units.index.get_level_values("heldout_class")) if len(units) else set()
    coverage={"eligible_units":int(len(units)),"denominator_units":18,"every_cytokine":EXPECTED_CYTOKINES<=cytokines,"every_class":EXPECTED_CLASSES<=classes,"both_donors":set(donors.index)==EXPECTED_DONORS}
    wins=int((units>0).sum())
    donor_effects={str(k):float(v) for k,v in donors.items()}
    criteria={
        "technical_pairing_complete":pairing_complete,
        "each_donor_positive":coverage["both_donors"] and all(v>0 for v in donor_effects.values()),
        "donor_averaged_difference_at_least_0.02":observed is not None and observed>=minimum_effect,
        "wins_at_least_12_of_18":wins>=minimum_wins,
        "coverage_at_least_12":len(units)>=12,
        "coverage_every_cytokine":coverage["every_cytokine"],
        "coverage_every_class":coverage["every_class"],
    }
    boot=descriptive_two_way_bootstrap(joint,n=n_bootstrap) if len(joint) else {"lower_95":None,"upper_95":None,"bootstrap_kind":"descriptive_two_way_cluster"}
    return {"table":table,"method":method,"comparator":comparator,"metric":metric,"panel":panel,"difference":observed,"donor_effects":donor_effects,"wins":wins,"coverage":coverage,"criteria":criteria,"passed":all(criteria.values()),"uncertainty":boot,"population_p_value":None}


def _energy_ok(records, table, method, comparator, panel):
    diff,paired=paired_technical_differences(records,table=table,method=method,comparator=comparator,metric="energy_distance",panel=panel)
    counts=diff.groupby(level=["cytokine","heldout_class"]).count() if len(diff) else pd.Series(dtype=int)
    units=diff.groupby(level=["cytokine","heldout_class"]).mean()[counts==len(EXPECTED_DONORS)] if len(diff) else pd.Series(dtype=float)
    macro=float(units.mean()) if len(units) else None
    coverage=len(units)>=12 and EXPECTED_CYTOKINES<=set(units.index.get_level_values("cytokine")) and EXPECTED_CLASSES<=set(units.index.get_level_values("heldout_class")) if len(units) else False
    return bool(paired and coverage and macro is not None and macro<=0),{"stack_minus_comparator":macro,"eligible_units":len(units),"technical_pairing_complete":paired,"coverage":bool(coverage)}


def gate(records: pd.DataFrame, *, n_bootstrap=10_000):
    _require_columns(records)
    contrasts=[]
    energies=[]
    for table,method,comparator in CONTRAST_SPECS:
        for metric in ("delta_pearson","de_lfc_spearman"):
            contrasts.append(contrast_summary(records,table=table,method=method,comparator=comparator,metric=metric,n_bootstrap=n_bootstrap))
        ok,values=_energy_ok(records,table,method,comparator,GATE_PANEL)
        energies.append({"table":table,"method":method,"comparator":comparator,"passed":ok,"macro":values})
    passed=all(c["passed"] for c in contrasts) and all(e["passed"] for e in energies)
    return {
        "passed":bool(passed),"expansion_authorized":bool(passed),"gate_panel":GATE_PANEL,
        "technical_resamples_averaged_first":True,"contrasts":contrasts,"energy_checks":energies,
        "inference_scope":"two observed donors; bootstrap intervals are descriptive, not population-level significance",
    }
