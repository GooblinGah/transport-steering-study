"""End-to-end driver: execute all methods on one manifest and emit metric rows.

Everything runs in Stack's 15012-gene space. Steering is applied at the final
Stack layer (after block 9): because the NB head is per-cell, transporting the
post-L9 embedding and decoding it is identical to intervening at layer 9 (verified
to 0.0 against the native path). ICL is the deterministic manual-path in-context
generation of the contract's step-5 layout, put on identical footing with steering.
"""
from __future__ import annotations
import json, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from .operators import LowRankMMD
from .decoders import fit_matched_decoders
from .output_scale import library_sizes, normalize_log1p, prepare_for_evaluation, synthetic_control_correct
from .evaluation import energy_distance, FixedEvaluatorSpace
from .panels import fit_control_only_panel
from .stack_context import build_source_fit_windows, build_query_windows, manual_no_mask_forward
from .contracts import canonical_hash

MMD_CONFIG = dict(rank=8, alpha=1.0, bandwidth_multiplier=1.0, movement=1e-2, steps=300, seed=0)
TRANSPORT_HASH = canonical_hash({"operator": "low_rank_mmd", **MMD_CONFIG})
PANEL_GENES = 2000
ICL_STEPS = 5


# --- Stack runtime ----------------------------------------------------------
def load_runtime(ckpt, genes_pkl, device="cuda"):
    import pickle, torch
    from stack.model_loading import load_model_from_checkpoint
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = load_model_from_checkpoint(str(ckpt), model_class="ICLFinetunedModel", device=device)
    model.eval()
    stack_genes = [str(g) for g in pickle.load(open(genes_pkl, "rb"))]
    return model, stack_genes


class GeneAligner:
    """Project Dong cells (by symbol) onto Stack's ordered gene vocabulary."""
    def __init__(self, dong_genes, stack_genes):
        idx = {g: i for i, g in enumerate(map(str, dong_genes))}
        self.colmap = np.array([idx.get(g, -1) for g in stack_genes])
        self.present = self.colmap >= 0
        self.n = len(stack_genes)

    def align(self, dense_rows):
        out = np.zeros((len(dense_rows), self.n), dtype=np.float32)
        out[:, self.present] = np.asarray(dense_rows)[:, self.colmap[self.present]]
        return out


def _forward(model, counts_2d, device, intervention_layer=None, intervention=None):
    import torch
    t = torch.as_tensor(counts_2d, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return manual_no_mask_forward(model, t, intervention_layer=intervention_layer,
                                      intervention=intervention, return_activations=intervention_layer is None)


def _focal_mean(cell_ids, values, start=333):
    acc = defaultdict(list)
    for i in range(start, len(cell_ids)):
        acc[cell_ids[i]].append(values[i])
    return {c: np.mean(v, 0) for c, v in acc.items()}


def _embed(model, counts_by_id, windows, device, key="activations"):
    """Average per-barcode post-L9 embeddings over focal appearances."""
    acc = defaultdict(list)
    for w in windows:
        cnt = np.stack([counts_by_id[c] for c in w.cell_ids])
        act = _forward(model, cnt, device)["activations"][-1][0].cpu().numpy()
        for c, v in _focal_mean(w.cell_ids, act).items():
            acc[c].append(v)
    return {c: np.mean(v, 0) for c, v in acc.items()}


def decode(model, emb, lib, device):
    import torch
    with torch.no_grad():
        nb = model.decode(torch.as_tensor(np.asarray(emb), dtype=torch.float32, device=device),
                          torch.as_tensor(np.asarray(lib), dtype=torch.float32, device=device))
    return nb.cpu().numpy()


def icl_predict(model, context_ids, context_counts, query_ids, query_counts, device, seed, steps=ICL_STEPS):
    """Deterministic in-context generation: re-feed the NB mean of the query cells
    for `steps` iterations, using `context` as the source condition."""
    ctx = {c: context_counts[i] for i, c in enumerate(context_ids)}
    cur = {c: query_counts[i].astype(np.float32) for i, c in enumerate(query_ids)}
    windows = build_query_windows(context_ids, query_ids, seed=seed)
    for _ in range(steps):
        acc = defaultdict(list)
        for w in windows:
            cnt = np.stack([ctx[c] if c in ctx else cur[c] for c in w.cell_ids])
            nb = _forward(model, cnt, device)["nb_mean"][0].cpu().numpy()
            for c, v in _focal_mean(w.cell_ids, nb).items():
                acc[c].append(v)
        cur = {c: np.mean(acc[c], 0) for c in query_ids}
    return np.stack([cur[c] for c in query_ids])


# --- metrics (cell-eval v0.6.6, the contract's pinned evaluator) -------------
# Correlation and DE metrics come from Arc's cell-eval: pearson_delta and
# de_spearman_lfc_sig (DE genes from pdex wilcoxon). Energy distance uses
# cell-eval's E-distance formula (energy_distance()) in the fixed 50-PC space.
CE_SKIP = ["discrimination_score_l1", "discrimination_score_l2", "discrimination_score_cosine",
           "pearson_edistance", "clustering_agreement"]  # aggregate metrics undefined for one perturbation


def evaluate_task(preds, q0_counts, y1_counts, panel_idx, evaluator, outdir, num_threads=16):
    """Score every method against Y1 with cell-eval, reusing the real DE across methods."""
    import anndata as ad
    from cell_eval import MetricsEvaluator
    import polars as pl
    genes = [f"g{int(i)}" for i in panel_idx]
    q0_log = prepare_for_evaluation(q0_counts)[:, panel_idx]
    y1_log = prepare_for_evaluation(y1_counts)[:, panel_idx]
    def _ad(pert_log):
        X = np.vstack([q0_log, pert_log]).astype(np.float32)
        obs = pd.DataFrame({"perturbation": ["control"] * len(q0_log) + ["pert"] * len(pert_log)})
        return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    real_ad = _ad(y1_log)
    de_real = None; deg_count = 0; deg_hash = ""
    out = {}
    for name, pred in preds.items():
        pred_log = prepare_for_evaluation(pred)[:, panel_idx]
        ev = MetricsEvaluator(adata_pred=_ad(pred_log), adata_real=real_ad.copy(), de_real=de_real,
                              control_pert="control", pert_col="perturbation", de_method="wilcoxon",
                              num_threads=num_threads, outdir=outdir)
        if de_real is None:
            de_real = ev.de_comparison.real.data
            sig = de_real.filter(pl.col("fdr") < 0.05)
            deg_count = int(sig.height); deg_hash = canonical_hash(sorted(sig["feature"].to_list()))
        res, _ = ev.compute(profile="full", skip_metrics=CE_SKIP, write_csv=False)
        row = res.to_pandas().set_index("perturbation").loc["pert"]
        lfc = row.get("de_spearman_lfc_sig"); dp = row.get("pearson_delta")   # absent if a metric was skipped
        de_lfc = float(lfc) if deg_count >= 10 and lfc is not None and np.isfinite(lfc) else None
        out[name] = {"delta_pearson": float(dp) if dp is not None and np.isfinite(dp) else None,
                     "de_lfc_spearman": de_lfc,
                     "energy_distance": energy_distance(pred_log, y1_log, evaluator)}
    return out, deg_count, deg_hash


# --- per-manifest driver ----------------------------------------------------
def run_manifest(model, aligner, counts_lookup, fine_lookup, manifest, device, seed=0, outdir="/tmp/ce", num_threads=16):
    """counts_lookup(cell_id)->raw Dong dense row; fine_lookup(cell_id)->fine label."""
    roles = {r: list(manifest[r]) for r in ("p0", "p1", "q0", "y1")}
    ids = roles["p0"] + roles["p1"] + roles["q0"] + roles["y1"]
    aligned = aligner.align(np.stack([counts_lookup(c) for c in ids]))
    cnt = {c: aligned[i] for i, c in enumerate(ids)}
    C = lambda r: np.stack([cnt[c] for c in roles[r]])
    p0c, p1c, q0c, y1c = C("p0"), C("p1"), C("q0"), C("y1")
    q0_lib = library_sizes(q0c)

    # Stack post-L9 embeddings (source-fit windows for P0/P1, query window for Q0).
    src_w = build_source_fit_windows(roles["p0"], roles["p1"],
                                     [fine_lookup(c) for c in roles["p0"]],
                                     [fine_lookup(c) for c in roles["p1"]], seed=seed, sealed_y1=roles["y1"])
    emb = _embed(model, cnt, src_w, device)
    emb_p0 = np.stack([emb[c] for c in roles["p0"]]); emb_p1 = np.stack([emb[c] for c in roles["p1"]])
    q_emb = _embed(model, cnt, build_query_windows(roles["p0"], roles["q0"], seed=seed, sealed_y1=roles["y1"]), device)
    emb_q0 = np.stack([q_emb[c] for c in roles["q0"]])

    # Transport maps (identical config for Stack and PCA representations).
    import torch
    to_dev = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mmd_stack = LowRankMMD(**MMD_CONFIG).fit(to_dev(emb_p0), to_dev(emb_p1))
    tq_stack = mmd_stack.transform(emb_q0)

    from sklearn.decomposition import PCA
    train_log = normalize_log1p(np.vstack([p0c, p1c, q0c]))   # PCA basis fit on P0,P1,Q0 only (contract §3.2)
    Z = PCA(min(50, len(train_log) - 1, aligner.n), random_state=0).fit(train_log)
    zp0, zp1, zq0 = (Z.transform(normalize_log1p(x)) for x in (p0c, p1c, q0c))
    mmd_pca = LowRankMMD(**MMD_CONFIG).fit(zp0, zp1)
    tq_pca = mmd_pca.transform(zq0)

    # Matched ridge decoders (one expression basis + one cell order, matched hashes).
    stack_dec, pca_dec = fit_matched_decoders(
        np.vstack([emb_p0, emb_p1, emb_q0]), np.vstack([zp0, zp1, zq0]),
        np.vstack([p0c, p1c, q0c]), train_ids=roles["p0"] + roles["p1"] + roles["q0"], y1_ids=roles["y1"])

    # Fixed control-only panel + evaluator (never see Y1).
    ctrl_ids = roles["p0"] + roles["q0"]
    panel = fit_control_only_panel(np.vstack([p0c, q0c]), ctrl_ids,
                                   [f"g{i}" for i in range(aligner.n)], n_genes=PANEL_GENES, sealed_y1=roles["y1"])
    panel_idx = np.array([int(g[1:]) for g in panel.genes])
    evaluator = FixedEvaluatorSpace(50).fit(prepare_for_evaluation(np.vstack([p0c, p1c, q0c]))[:, panel_idx],
                                            gene_ids=panel.genes)

    # Predictions (anchored synthetic-control correction, contract §4).
    def corrected(pert, noop):
        return synthetic_control_correct(q0c, pert, noop)

    tables = {"stack_mmd_native": "end_to_end", "stack_icl_native": "end_to_end",
              "stack_mmd_matched_ridge": "representation_controlled",
              "pca_mmd_matched_ridge": "representation_controlled"}
    preds = {
        "stack_mmd_native": corrected(decode(model, tq_stack, q0_lib, device), decode(model, emb_q0, q0_lib, device)),
        "stack_icl_native": corrected(icl_predict(model, roles["p1"], p1c, roles["q0"], q0c, device, seed),
                                      icl_predict(model, roles["p0"], p0c, roles["q0"], q0c, device, seed)),
        "stack_mmd_matched_ridge": corrected(stack_dec.decode_expected_counts(tq_stack, q0_lib),
                                             stack_dec.decode_expected_counts(emb_q0, q0_lib)),
        "pca_mmd_matched_ridge": corrected(pca_dec.decode_expected_counts(tq_pca, q0_lib),
                                           pca_dec.decode_expected_counts(zq0, q0_lib)),
    }

    metrics, deg_count, deg_hash = evaluate_task(preds, q0c, y1c, panel_idx, evaluator, outdir, num_threads)
    rows = []
    for name, vals in metrics.items():
        rep = tables[name] == "representation_controlled"
        base = dict(task_id=manifest["task_id"], donor=manifest["donor"], cytokine=manifest["cytokine"],
                    heldout_class=manifest["heldout_class"], method=name, comparison_table=tables[name],
                    resample=manifest["resample"], generation_seed=0, window_seed=seed, eligible=True,
                    evaluator_panel="control_only_fixed", sampling_mode=manifest["sampling_mode"],
                    manifest_hash=manifest["manifest_hash"], panel_hash=panel.panel_hash,
                    evaluator_hash=evaluator.evaluator_hash, deg_count=deg_count, deg_set_hash=deg_hash,
                    decoder_train_ids_hash=stack_dec.decoder_train_ids_hash_ if rep else "",
                    expression_basis_hash=stack_dec.expression_basis_hash_ if rep else "",
                    transport_config_hash=TRANSPORT_HASH if rep else "")
        for metric in ("delta_pearson", "de_lfc_spearman", "energy_distance"):
            rows.append(base | {"metric": metric, "value": vals[metric]})
    return rows


def run_study(manifest_dir, metadata_dir, ckpt, genes_pkl, raw_adata, out, device="cuda", limit=None, num_threads=16):
    import anndata as ad, logging, os
    os.environ.setdefault("TQDM_DISABLE", "1")          # silence pdex/cell-eval progress bars
    logging.disable(logging.INFO)
    manifest_dir, out = Path(manifest_dir), Path(out)
    ce_out = str(out.parent / "cell_eval_tmp")
    model, stack_genes = load_runtime(ckpt, genes_pkl, device)
    obs = pd.read_parquet(Path(metadata_dir) / "obs.parquet")
    from .manifests import standardize_fine_label
    fine = obs["cell_type0528"].map(standardize_fine_label)
    pos = obs["raw_position"].to_dict(); fine_map = fine.to_dict()

    raw = ad.read_h5ad(raw_adata)
    X = raw.X.tocsr() if hasattr(raw.X, "tocsr") else raw.X
    aligner = GeneAligner([str(g) for g in raw.var_names], stack_genes)
    def counts_lookup(cid):
        r = pos[cid]; row = X[r]
        return np.asarray(row.todense()).ravel() if hasattr(row, "todense") else np.asarray(row).ravel()

    files = sorted(f for f in manifest_dir.glob("*fine_matched*.json") if json.loads(f.read_text())["eligible"])
    if limit: files = files[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for i, f in enumerate(files):
        m = json.loads(f.read_text())
        all_rows += run_manifest(model, aligner, counts_lookup, lambda c: fine_map[c], m, device,
                                 outdir=ce_out, num_threads=num_threads)
        print(f"[{i+1}/{len(files)}] {m['task_id']}", flush=True)
        if (i + 1) % 25 == 0:  # checkpoint for a long run
            pd.DataFrame(all_rows).to_parquet(out)
    df = pd.DataFrame(all_rows)
    df.to_parquet(out)
    print("wrote", out, df.shape)
    return df
