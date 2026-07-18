from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from .output_scale import library_sizes, log1p_to_expected_counts, normalize_log1p


def lognorm(x):
    return normalize_log1p(x)


def no_change(p0, p1, q0, **_):
    return np.asarray(q0, float).copy()


def mean_shift(p0, p1, q0, alpha=1.0, **_):
    if alpha == 0:
        return np.asarray(q0, float).copy()
    return np.maximum(0, q0 + alpha * (np.mean(p1, 0) - np.mean(p0, 0)))


def pca_mean_shift(p0, p1, q0, alpha=1.0, n_components=50, **_):
    if alpha == 0:
        return np.asarray(q0, float).copy()
    all_x = np.vstack([p0, p1, q0])
    k = min(n_components, all_x.shape[0] - 1, all_x.shape[1])
    pca = PCA(k, random_state=0).fit(all_x)
    z0, z1, zq = map(pca.transform, (p0, p1, q0))
    return np.maximum(0, pca.inverse_transform(zq + alpha * (z1.mean(0) - z0.mean(0))))


class LowRankMMD:
    """Fit a source-only affine map with unbiased RBF MMD."""
    def __init__(self, rank=8, alpha=1.0, bandwidth_multiplier=1.0, movement=1e-2, factor_norm=1e-4, steps=300, lr=2e-2, seed=0):
        self.rank, self.alpha, self.bw_mult = rank, alpha, bandwidth_multiplier
        self.movement, self.factor_norm, self.steps, self.lr, self.seed = movement, factor_norm, steps, lr, seed

    def fit(self, x0, x1):
        import torch
        torch.manual_seed(self.seed)
        x0, x1 = map(lambda x: torch.as_tensor(x, dtype=torch.float32), (x0, x1))
        self.mu_ = x0.mean(0); self.scale_ = x0.std(0).clamp_min(1e-5)
        a, b = (x0-self.mu_)/self.scale_, (x1-self.mu_)/self.scale_
        self.z_mu_ = a.mean(0)
        d = a.shape[1]; r = min(self.rank, d)
        U = torch.nn.Parameter(torch.zeros(d, r)); V = torch.nn.Parameter(torch.zeros(d, r)); shift = torch.nn.Parameter(b.mean(0)-a.mean(0))
        torch.nn.init.normal_(U, std=1e-3); torch.nn.init.normal_(V, std=1e-3)
        med = torch.median(torch.cdist(a, b).detach()).clamp_min(1e-4) * self.bw_mult
        opt = torch.optim.Adam([U, V, shift], lr=self.lr)
        def mmd(x, y):
            kxx=torch.exp(-torch.cdist(x,x).square()/(2*med.square())); kyy=torch.exp(-torch.cdist(y,y).square()/(2*med.square())); kxy=torch.exp(-torch.cdist(x,y).square()/(2*med.square()))
            n,m=len(x),len(y)
            return (kxx.sum()-kxx.diag().sum())/(n*(n-1)) + (kyy.sum()-kyy.diag().sum())/(m*(m-1)) - 2*kxy.mean()
        self.loss_before_ = float(mmd(a,b))
        for _ in range(self.steps):
            moved = a + shift + (a-self.z_mu_) @ V @ U.T
            delta=moved-a; loss=mmd(moved,b)+self.movement*delta.square().mean()+self.factor_norm*(U.square().mean()+V.square().mean())
            opt.zero_grad(); loss.backward(); opt.step()
        self.U_, self.V_, self.shift_ = U.detach(), V.detach(), shift.detach()
        self.loss_after_ = float(mmd(a + self.shift_ + (a-self.z_mu_)@self.V_@self.U_.T,b))
        return self

    def transform(self, x, alpha=None):
        import torch
        x=torch.as_tensor(x,dtype=torch.float32); z=(x-self.mu_)/self.scale_; a=self.alpha if alpha is None else alpha
        out=z+a*(self.shift_+(z-self.z_mu_)@self.V_@self.U_.T)
        return (out*self.scale_+self.mu_).numpy()


def pca_mmd(p0, p1, q0, n_components=50, **kwargs):
    if kwargs.get("alpha", 1.0) == 0:
        return np.asarray(q0, float).copy()
    all_x=np.vstack([p0,p1,q0]); k=min(n_components, all_x.shape[0]-1,all_x.shape[1]); pca=PCA(k,random_state=0).fit(all_x)
    model=LowRankMMD(**kwargs).fit(pca.transform(p0),pca.transform(p1))
    return np.maximum(0,pca.inverse_transform(model.transform(pca.transform(q0))))


def sinkhorn_coupling(x0, x1, reg_multiplier=.05, *, unbalanced=False, mass_penalty=1.0, max_iter=1000, tol=1e-8):
    """Calculate balanced entropic OT. Optionally, calculate the unbalanced sensitivity case."""
    cost = cdist(x0, x1, "sqeuclidean")
    positive = cost[cost > 0]
    median = np.median(positive) if positive.size else 1.0
    epsilon = max(float(reg_multiplier) * float(median), 1e-12)
    kernel = np.exp(np.clip(-cost / epsilon, -700, 0))
    a = np.full(len(x0), 1.0 / len(x0)); b = np.full(len(x1), 1.0 / len(x1))
    u = np.ones_like(a); v = np.ones_like(b)
    exponent = mass_penalty / (mass_penalty + epsilon) if unbalanced else 1.0
    for _ in range(max_iter):
        old = u.copy()
        u = np.power(a / np.maximum(kernel @ v, 1e-300), exponent)
        v = np.power(b / np.maximum(kernel.T @ u, 1e-300), exponent)
        if np.max(np.abs(u - old)) < tol:
            break
    coupling = (u[:, None] * kernel) * v[None, :]
    if not unbalanced:
        # Use final projections to remove accumulated numerical error.
        for _ in range(10):
            coupling *= np.divide(a, coupling.sum(1), out=np.ones_like(a), where=coupling.sum(1)>0)[:,None]
            coupling *= np.divide(b, coupling.sum(0), out=np.ones_like(b), where=coupling.sum(0)>0)[None,:]
    return coupling


def ot_barycentric_displacements(p0, p1, reg_multiplier=.05, *, unbalanced=False, mass_penalty=1.0):
    coupling = sinkhorn_coupling(p0, p1, reg_multiplier, unbalanced=unbalanced, mass_penalty=mass_penalty)
    row_mass = coupling.sum(1, keepdims=True)
    targets = np.divide(coupling @ p1, row_mass, out=np.asarray(p0, float).copy(), where=row_mass>0)
    return targets - p0


def local_transport(p0, p1, q0, k=16, reg_multiplier=.05, unbalanced=False, mass_penalty=1.0, **_):
    disp=ot_barycentric_displacements(p0,p1,reg_multiplier,unbalanced=unbalanced,mass_penalty=mass_penalty)
    nn=NearestNeighbors(n_neighbors=min(k,len(p0))).fit(p0); dist,idx=nn.kneighbors(q0)
    weights=np.exp(-dist/np.maximum(np.median(dist),1e-8)); weights/=weights.sum(1,keepdims=True)
    return np.maximum(0,q0+(disp[idx]*weights[...,None]).sum(1))


def gene_mean_shift_counts(p0_counts, p1_counts, q0_counts, alpha=1.0):
    """Calculate the gene baseline in log space. Return expected counts."""
    q_lib = library_sizes(q0_counts)
    pred_log = mean_shift(lognorm(p0_counts), lognorm(p1_counts), lognorm(q0_counts), alpha=alpha)
    return log1p_to_expected_counts(pred_log, q_lib)


def pca_mean_shift_counts(p0_counts, p1_counts, q0_counts, alpha=1.0, n_components=50):
    q_lib = library_sizes(q0_counts)
    pred_log = pca_mean_shift(lognorm(p0_counts), lognorm(p1_counts), lognorm(q0_counts), alpha=alpha, n_components=n_components)
    return log1p_to_expected_counts(pred_log, q_lib)


def pca_mmd_counts(p0_counts, p1_counts, q0_counts, n_components=50, **kwargs):
    q_lib = library_sizes(q0_counts)
    pred_log = pca_mmd(lognorm(p0_counts), lognorm(p1_counts), lognorm(q0_counts), n_components=n_components, **kwargs)
    return log1p_to_expected_counts(pred_log, q_lib)
