"""Make pinned Stack contexts. Run the manual path without a random mask."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Callable, Sequence
import numpy as np
from .contracts import canonical_hash

WINDOW_SIZE=512
PROMPT_ROWS=128
ICL_MASK_SCHEDULE=(.8,.6,.4,.2,0.)
ICL_CONTEXT_RATIOS=(.20,.25,.30,.35,.40)
ICL_SOURCE_ROWS=(231,256,282,308,333)
ICL_QUERY_ROWS=(281,256,230,204,179)
ICL_QUERY_POSITION_STARTS=(230,256,281,307,332)


@dataclass(frozen=True)
class StackWindow:
    stage: str
    condition: str
    cell_ids: tuple[str,...]
    source_rows: int
    focal_or_query_rows: int
    query_position_start: int
    actual_focal_or_query_start: int
    seed: int
    window_index: int

    @property
    def window_hash(self):return canonical_hash(asdict(self))
    def validate(self,sealed_y1=()):
        if len(self.cell_ids)!=WINDOW_SIZE or self.source_rows+self.focal_or_query_rows!=WINDOW_SIZE:raise ValueError("Stack windows must contain exactly 512 rows")
        overlap=set(self.cell_ids)&set(sealed_y1)
        if overlap:raise ValueError(f"Stack context overlaps sealed Y1 ({len(overlap)} cells)")
        if self.stage=="source_fit" and self.actual_focal_or_query_start!=333:raise ValueError("source fitting must use 333 context + 179 focal rows")


def official_icl_schedule():
    return [dict(step=i+1,mask_ratio=ICL_MASK_SCHEDULE[i],context_ratio=ICL_CONTEXT_RATIOS[i],source_rows=ICL_SOURCE_ROWS[i],query_rows=ICL_QUERY_ROWS[i],query_position_start=ICL_QUERY_POSITION_STARTS[i],source_permutation_seed=i+1) for i in range(5)]


def _paired_order(ids0,ids1,fine0,fine1,rng):
    ids0=np.asarray(ids0,dtype=object);ids1=np.asarray(ids1,dtype=object);fine0=np.asarray(fine0);fine1=np.asarray(fine1)
    if len(ids0)!=len(ids1):raise ValueError("P0/P1 must have equal selected size")
    left=[];right=[]
    labels=sorted(set(fine0))
    if set(labels)!=set(fine1):raise ValueError("P0/P1 fine-label sets differ")
    for label in labels:
        a=ids0[fine0==label].copy();b=ids1[fine1==label].copy()
        if len(a)!=len(b):raise ValueError(f"P0/P1 fine-label count differs for {label}")
        rng.shuffle(a);rng.shuffle(b);left.extend(a);right.extend(b)
    permutation=rng.permutation(len(left))
    return np.asarray(left,dtype=object)[permutation],np.asarray(right,dtype=object)[permutation]


def _cycle(values,start,n):
    values=np.asarray(values,dtype=object)
    if not len(values):raise ValueError("cannot build a window from zero cells")
    return tuple(values[(np.arange(n)+start)%len(values)].tolist())


def build_source_fit_windows(p0_ids,p1_ids,p0_fine,p1_fine,*,seed,sealed_y1=()):
    """Make paired condition-homogeneous windows. Move each barcode through focal rows."""
    rng=np.random.default_rng(seed);left,right=_paired_order(p0_ids,p1_ids,p0_fine,p1_fine,rng)
    windows=[];n_windows=math.ceil(len(left)/179)
    for index in range(n_windows):
        focal_start=index*179;context_start=(focal_start+179)%len(left)
        for condition,values in (("P0",left),("P1",right)):
            ids=_cycle(values,context_start,333)+_cycle(values,focal_start,179)
            window=StackWindow("source_fit",condition,ids,333,179,332,333,seed,index);window.validate(sealed_y1);windows.append(window)
    return windows


def build_query_windows(p0_context_ids,q0_ids,*,seed,sealed_y1=()):
    rng=np.random.default_rng(seed);context=np.asarray(p0_context_ids,dtype=object).copy();query=np.asarray(q0_ids,dtype=object).copy();rng.shuffle(context)
    windows=[]
    for index,start in enumerate(range(0,len(query),179)):
        ids=_cycle(context,index*333,333)+_cycle(query,start,179)
        window=StackWindow("query_decode","Q0",ids,333,179,332,333,seed,index);window.validate(sealed_y1);windows.append(window)
    return windows


def manual_no_mask_forward(model,raw_counts,*,query_position_start=332,intervention_layer=None,intervention:Callable|None=None,actual_query_start=333,return_activations=False):
    """Run Stack without the random-mask method. Use layer 0 for tokens and layers 1 through 9 for blocks."""
    import torch
    counts=torch.as_tensor(raw_counts,dtype=next(model.parameters()).dtype,device=next(model.parameters()).device)
    if counts.ndim!=3 or counts.shape[1]!=WINDOW_SIZE:raise ValueError("raw_counts must have shape [batch,512,genes]")
    if intervention_layer is not None and intervention is None:raise ValueError("intervention callable is required")
    library=counts.sum(-1,keepdim=True)
    tokens=model._reduce_and_tokenize(torch.log1p(counts))
    if getattr(model,"query_pos_embedding",None) is not None:
        tokens=tokens.clone();tokens[:,query_position_start:]+=model.query_pos_embedding[None,None]
    mask=torch.zeros(WINDOW_SIZE,WINDOW_SIZE,dtype=torch.bool,device=tokens.device);mask[:PROMPT_ROWS,PROMPT_ROWS:]=True;mask=mask[None,None]
    activations=[]
    def capture(x):return x.reshape(x.shape[0],x.shape[1],-1)
    def apply_query(x):
        flat=capture(x);moved=intervention(flat[:,actual_query_start:])
        if moved.shape!=flat[:,actual_query_start:].shape:raise ValueError("intervention changed query activation shape")
        flat=flat.clone();flat[:,actual_query_start:]=moved;return flat.reshape_as(x)
    if return_activations:activations.append(capture(tokens).detach().clone())
    if intervention_layer==0:tokens=apply_query(tokens)
    for layer_index,block in enumerate(model.layers,start=1):
        tokens,_=block(tokens,model.gene_pos_embedding,mask,False)
        if intervention_layer==layer_index:tokens=apply_query(tokens)
        if return_activations:activations.append(capture(tokens).detach().clone())
    final=capture(tokens)
    nb_mean,nb_dispersion,px_scale=model._compute_nb_parameters(final,library)
    result={"nb_mean":nb_mean,"nb_dispersion":nb_dispersion,"px_scale":px_scale,"library_size":library}
    if return_activations:result["activations"]=activations
    return result
