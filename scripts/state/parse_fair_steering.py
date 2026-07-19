"""FAIR Parse comparison — runs in .venv-state. Both our MMD steering and STATE's transition
predictions are decoded by STATE's OWN gene_decoder (used exactly, loaded from the model).
Only the transformation differs. Saves per-cytokine control/perturbed arrays for cell-eval."""
import warnings, logging, os
warnings.filterwarnings("ignore"); logging.disable(logging.INFO); os.environ["TQDM_DISABLE"]="1"
import numpy as np, anndata as ad, torch
from state.tx.models.state_transition import StateTransitionPerturbationModel

SC="/tmp/claude-0/-workspace/de16e521-5da6-4dcb-b532-7b4a0ce6cbd8/scratchpad"
D="artifacts/models/ST-SE-Parse/zeroshot"
dev="cuda"
m=StateTransitionPerturbationModel.load_from_checkpoint(f"{D}/split_0/checkpoints/final.ckpt", map_location=dev).eval().to(dev)
gd=m.gene_decoder
@torch.no_grad()
def decode(emb):  # [N,2058] SE-space -> [N,2000] HVG, STATE's exact decoder
    t=torch.as_tensor(np.asarray(emb),dtype=torch.float32,device=dev)
    return gd(t).reshape(-1,2000).float().cpu().numpy()

class LowRankMMD:  # copied verbatim from transport_study.operators (identical steering math)
    def __init__(s,rank=8,alpha=1.0,bandwidth_multiplier=1.0,movement=1e-2,factor_norm=1e-4,steps=300,lr=2e-2,seed=0):
        s.rank,s.alpha,s.bw_mult,s.movement,s.factor_norm,s.steps,s.lr,s.seed=rank,alpha,bandwidth_multiplier,movement,factor_norm,steps,lr,seed
    def fit(s,x0,x1):
        torch.manual_seed(s.seed); x0=torch.as_tensor(x0,dtype=torch.float32,device=dev); x1=torch.as_tensor(x1,dtype=torch.float32,device=dev)
        s.mu_=x0.mean(0); s.scale_=x0.std(0).clamp_min(1e-5); a,b=(x0-s.mu_)/s.scale_,(x1-s.mu_)/s.scale_; s.z_mu_=a.mean(0)
        d=a.shape[1]; r=min(s.rank,d)
        U=torch.nn.Parameter(torch.zeros(d,r,device=dev)); V=torch.nn.Parameter(torch.zeros(d,r,device=dev)); sh=torch.nn.Parameter(b.mean(0)-a.mean(0))
        torch.nn.init.normal_(U,std=1e-3); torch.nn.init.normal_(V,std=1e-3)
        med=torch.median(torch.cdist(a,b).detach()).clamp_min(1e-4)*s.bw_mult; opt=torch.optim.Adam([U,V,sh],lr=s.lr)
        def mmd(x,y):
            kxx=torch.exp(-torch.cdist(x,x).square()/(2*med.square()));kyy=torch.exp(-torch.cdist(y,y).square()/(2*med.square()));kxy=torch.exp(-torch.cdist(x,y).square()/(2*med.square()));n,mm=len(x),len(y)
            return (kxx.sum()-kxx.diag().sum())/(n*(n-1))+(kyy.sum()-kyy.diag().sum())/(mm*(mm-1))-2*kxy.mean()
        for _ in range(s.steps):
            moved=a+sh+(a-s.z_mu_)@V@U.T; loss=mmd(moved,b)+s.movement*(moved-a).square().mean()+s.factor_norm*(U.square().mean()+V.square().mean())
            opt.zero_grad(); loss.backward(); opt.step()
        s.U_,s.V_,s.sh_=U.detach(),V.detach(),sh.detach(); return s
    def transform(s,x):
        x=torch.as_tensor(x,dtype=torch.float32,device=dev); z=(x-s.mu_)/s.scale_
        return ((z+s.alpha*(s.sh_+(z-s.z_mu_)@s.V_@s.U_.T))*s.scale_+s.mu_).cpu().numpy()

tgt=ad.read_h5ad(f"{D}/split_0/eval_best.ckpt/adata_real.h5ad")   # B_Int_Memory (Q0/Y1)
src=ad.read_h5ad(f"{D}/split_4/eval_best.ckpt/adata_real.h5ad")   # CD14_Mono (P0/P1)
stp=ad.read_h5ad(f"{D}/split_0/eval_best.ckpt/adata_pred.h5ad")   # STATE prediction
def dense(a): return np.asarray(a.todense()) if hasattr(a,"todense") else np.asarray(a)
rng=np.random.default_rng(0); N=150
def samp(A,cy):
    i=np.where(A.obs.cytokine.values==cy)[0]; return rng.choice(i,min(N,len(i)),replace=False)
CYTOS=["IFN-alpha1","IFN-beta","IFN-gamma","IFN-lambda1","IL-6","TNF-alpha","IL-2","IL-4","IL-10","IFN-omega","IL-21","GM-CSF"]
out={}
for cy in CYTOS:
    qi,yi,p0i,p1i,pc,pp=samp(tgt,"PBS"),samp(tgt,cy),samp(src,"PBS"),samp(src,cy),samp(stp,"PBS"),samp(stp,cy)
    if min(map(len,[qi,yi,p0i,p1i,pc,pp]))<20: print("skip",cy,flush=True); continue
    Q0,Y1=dense(tgt.X[qi]),dense(tgt.X[yi])
    seQ0=tgt.obsm["X_state"][qi]
    tq=LowRankMMD().fit(src.obsm["X_state"][p0i],src.obsm["X_state"][p1i]).transform(seQ0)
    out[cy]=dict(
        our_ctrl=decode(seQ0), our_pert=decode(tq),           # ours: STATE's decoder on our steered emb
        state_ctrl=dense(stp.X[pc]), state_pert=dense(stp.X[pp]),  # STATE: its own decoder output
        floor_ctrl=Q0, floor_pert=np.maximum(Q0+(dense(src.X[p1i]).mean(0)-dense(src.X[p0i]).mean(0)),0),
        real_ctrl=Q0, real_pert=Y1)
    print("done",cy,flush=True)
np.savez(f"{SC}/parse_fair.npz", **{f"{cy}__{k}":v for cy,d in out.items() for k,v in d.items()})
print("SAVED", len(out), "cytokines DONE")
