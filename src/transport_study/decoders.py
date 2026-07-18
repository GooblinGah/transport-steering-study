"""Make matched decoders for the representation-controlled comparison."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from .output_scale import normalize_log1p, log1p_to_expected_counts
from .contracts import canonical_hash


@dataclass
class MatchedRidgeDecoder:
    """Map standardized latent values to expected counts through ridge and common expression components."""
    n_expression_pcs: int = 50
    alphas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

    def fit_expression_basis(self, train_counts):
        expression = normalize_log1p(train_counts)
        k = min(self.n_expression_pcs, expression.shape[0] - 1, expression.shape[1])
        if k < 1:
            raise ValueError("at least two decoder-training cells are required")
        self.expression_basis_ = PCA(k, random_state=0).fit(expression)
        self.train_expression_pcs_ = self.expression_basis_.transform(expression)
        return self

    def fit_representation(self, train_latent):
        if not hasattr(self, "train_expression_pcs_"):
            raise RuntimeError("fit_expression_basis must be called first")
        z = np.asarray(train_latent, dtype=np.float64)
        if len(z) != len(self.train_expression_pcs_):
            raise ValueError("Stack and PCA decoders must use the identical ordered training cells")
        self.latent_scaler_ = StandardScaler().fit(z)
        cv = min(5, len(z))
        self.ridge_ = RidgeCV(alphas=self.alphas, cv=cv).fit(
            self.latent_scaler_.transform(z), self.train_expression_pcs_
        )
        return self

    def decode_log_expression(self, latent):
        pcs = self.ridge_.predict(self.latent_scaler_.transform(np.asarray(latent)))
        return self.expression_basis_.inverse_transform(pcs)

    def decode_expected_counts(self, latent, query_library_sizes):
        return log1p_to_expected_counts(self.decode_log_expression(latent), query_library_sizes)


def validate_decoder_training_ids(train_ids,y1_ids):
    ordered=tuple(map(str,train_ids)); sealed=set(map(str,y1_ids))
    overlap=set(ordered)&sealed
    if overlap:raise ValueError(f"decoder training overlaps sealed Y1 ({len(overlap)} cells)")
    if len(ordered)!=len(set(ordered)):raise ValueError("decoder training IDs must be unique")
    return canonical_hash(ordered)


def fit_matched_decoders(stack_train, pca_train, train_counts, *, train_ids=None, y1_ids=(), **kwargs):
    """Fit both decoders with one fixed expression basis and one cell order."""
    if len(stack_train)!=len(pca_train) or len(stack_train)!=len(train_counts):raise ValueError("matched decoders require identical ordered training cells")
    train_hash=validate_decoder_training_ids(range(len(train_counts)) if train_ids is None else train_ids,y1_ids)
    common = MatchedRidgeDecoder(**kwargs).fit_expression_basis(train_counts)
    stack = MatchedRidgeDecoder(common.n_expression_pcs, common.alphas)
    pca = MatchedRidgeDecoder(common.n_expression_pcs, common.alphas)
    for decoder in (stack, pca):
        decoder.expression_basis_ = common.expression_basis_
        decoder.train_expression_pcs_ = common.train_expression_pcs_
    basis_hash=canonical_hash({"components":common.expression_basis_.components_.round(12).tolist(),"mean":common.expression_basis_.mean_.round(12).tolist()})
    for decoder in (stack,pca): decoder.decoder_train_ids_hash_=train_hash;decoder.expression_basis_hash_=basis_hash
    return stack.fit_representation(stack_train), pca.fit_representation(pca_train)
