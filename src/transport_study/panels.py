"""Make gene panels with selection-time leakage controls."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .contracts import canonical_hash
from .output_scale import normalize_log1p


@dataclass(frozen=True)
class GenePanel:
    panel_id:str
    genes:tuple[str,...]
    fit_cell_ids_hash:str
    selection_rule:str
    panel_hash:str


def fit_control_only_panel(control_counts,cell_ids,genes,*,n_genes=2000,sealed_y1=()):
    ids=tuple(map(str,cell_ids));gene_ids=tuple(map(str,genes));overlap=set(ids)&set(map(str,sealed_y1))
    if overlap:raise ValueError(f"control-only panel fit overlaps sealed Y1 ({len(overlap)} cells)")
    x=normalize_log1p(control_counts)
    if x.shape!=(len(ids),len(gene_ids)):raise ValueError("panel matrix/cell/gene dimensions disagree")
    variance=np.var(x,axis=0);order=sorted(range(len(gene_ids)),key=lambda i:(-variance[i],gene_ids[i],i))[:min(n_genes,len(gene_ids))]
    selected=tuple(gene_ids[i] for i in order);fit_hash=canonical_hash(ids)
    rule="control_only_log1p_variance_top_n_v1"
    return GenePanel("control_only_fixed",selected,fit_hash,rule,canonical_hash({"genes":selected,"fit":fit_hash,"rule":rule}))
