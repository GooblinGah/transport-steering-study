"""Fit exploratory source-only state maps to OT barycentric pseudo-targets."""
from __future__ import annotations
import numpy as np


class StateDependentTransport:
    r"""Fit T(h,s)=h+b0+Bphi(s)+U diag(1+Cphi(s)) V^T(h-mu)."""
    def __init__(self,rank=8,mmd_weight=.1,movement=1e-2,factor_norm=1e-4,steps=500,lr=2e-2,seed=0):
        self.rank,self.mmd_weight,self.movement,self.factor_norm=rank,mmd_weight,movement,factor_norm
        self.steps,self.lr,self.seed=steps,lr,seed

    def fit(self,h,state,pseudo_targets,treated_source):
        """Use source-only OT barycenters as pseudo-targets. Use marginal MMD as a regularizer."""
        import torch
        torch.manual_seed(self.seed)
        h,s,pseudo,y=map(lambda x:torch.as_tensor(x,dtype=torch.float32),(h,state,pseudo_targets,treated_source))
        if h.shape!=pseudo.shape or len(h)!=len(s):raise ValueError("one source-only pseudo-target/state per source cell is required")
        d,p=h.shape[1],s.shape[1];r=min(self.rank,d)
        self.mu_=h.mean(0); self.state_mean_=s.mean(0); self.state_scale_=s.std(0).clamp_min(1e-5); s=(s-self.state_mean_)/self.state_scale_
        b0=torch.nn.Parameter((pseudo-h).mean(0));B=torch.nn.Parameter(torch.zeros(d,p));U=torch.nn.Parameter(torch.zeros(d,r));V=torch.nn.Parameter(torch.zeros(d,r));C=torch.nn.Parameter(torch.zeros(r,p))
        for z in (B,U,V,C):torch.nn.init.normal_(z,std=1e-3)
        med=torch.median(torch.cdist(pseudo,y).detach()).clamp_min(1e-4)
        def transform(x,phi):
            scale=1+phi@C.T
            return x+b0+phi@B.T+(((x-self.mu_)@V)*scale)@U.T
        def mmd(x,z):
            kxx=torch.exp(-torch.cdist(x,x).square()/(2*med.square()));kzz=torch.exp(-torch.cdist(z,z).square()/(2*med.square()));kxz=torch.exp(-torch.cdist(x,z).square()/(2*med.square()));n,m=len(x),len(z)
            return (kxx.sum()-kxx.diag().sum())/(n*(n-1))+(kzz.sum()-kzz.diag().sum())/(m*(m-1))-2*kxz.mean()
        opt=torch.optim.Adam([b0,B,U,V,C],lr=self.lr)
        for _ in range(self.steps):
            moved=transform(h,s);delta=moved-h
            loss=(moved-pseudo).square().mean()+self.mmd_weight*mmd(moved,y)+self.movement*delta.square().mean()+self.factor_norm*sum(z.square().mean() for z in (B,U,V,C))
            opt.zero_grad();loss.backward();opt.step()
        self.b0_,self.B_,self.U_,self.V_,self.C_=[z.detach() for z in (b0,B,U,V,C)]
        self.source_pseudotarget_mse_=float((transform(h,s)-pseudo).square().mean())
        return self

    def transform(self,h,state):
        import torch
        x=torch.as_tensor(h,dtype=torch.float32);s=(torch.as_tensor(state,dtype=torch.float32)-self.state_mean_)/self.state_scale_
        scale=1+s@self.C_.T
        return (x+self.b0_+s@self.B_.T+(((x-self.mu_)@self.V_)*scale)@self.U_.T).numpy()
