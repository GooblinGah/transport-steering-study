import numpy as np
import torch
import pytest

from transport_study.run import GeneAligner, _focal_mean, _load_crossfit_transport_configs, evaluate_task, run_manifest
from transport_study.evaluation import FixedEvaluatorSpace
from transport_study.contracts import TaskManifest
from transport_study.output_scale import prepare_for_evaluation


def test_gene_aligner_projects_by_symbol():
    al = GeneAligner(["B", "A", "C"], ["A", "B", "Z"])   # Z absent in Dong -> zero
    out = al.align(np.array([[10.0, 20.0, 30.0]]))       # dong order B,A,C
    assert out.shape == (1, 3)
    assert out[0, 0] == 20.0 and out[0, 1] == 10.0 and out[0, 2] == 0.0  # A,B,Z


def test_focal_mean_averages_per_barcode_over_focal_rows():
    ids = tuple(f"ctx{i}" for i in range(333)) + ("a", "b", "a") + tuple("z" for _ in range(176))
    vals = np.arange(len(ids))[:, None].astype(float)
    fm = _focal_mean(ids, vals, start=333)
    assert fm["a"][0] == pytest.approx((333 + 335) / 2)   # rows 333 and 335
    assert fm["b"][0] == pytest.approx(334)
    assert "ctx0" not in fm                                 # context rows excluded


def test_crossfit_transport_configs_are_opposite_donor_and_on_grid(tmp_path):
    path = tmp_path / "selected.json"
    valid = {
        "H2D2": {"selected_on_donor": "H3D2", "rank": 8, "alpha": 1.0,
                  "bandwidth_multiplier": 1.0, "movement": 0.01},
        "H3D2": {"selected_on_donor": "H2D2", "rank": 4, "alpha": 0.5,
                  "bandwidth_multiplier": 2.0, "movement": 0.001},
    }
    import json
    path.write_text(json.dumps(valid))
    assert _load_crossfit_transport_configs(path)["H2D2"]["rank"] == 8
    valid["H2D2"]["rank"] = 7
    path.write_text(json.dumps(valid))
    with pytest.raises(ValueError, match="outside the registered grid"):
        _load_crossfit_transport_configs(path)


def test_evaluate_task_uses_cell_eval_and_rewards_correct_shift(tmp_path):
    """The two correlation metrics come from cell-eval (pearson_delta, de_spearman_lfc_sig)."""
    rng = np.random.default_rng(1)
    G = 400
    q0 = rng.poisson(4, size=(30, G)).astype(float)
    shift = np.zeros(G); shift[:120] = 12.0
    y1 = np.maximum(q0 + rng.poisson(np.abs(shift) + 1, size=(30, G)), 0).astype(float)
    good = np.maximum(q0 + shift, 0.0)                          # right direction
    bad = np.maximum(q0 + shift[::-1], 0.0)                     # shift on the wrong genes
    panel_idx = np.arange(G)
    ev = FixedEvaluatorSpace(20).fit(prepare_for_evaluation(np.vstack([q0, y1]))[:, panel_idx])
    metrics, deg_count, deg_hash = evaluate_task(
        {"good": good, "bad": bad}, q0, y1, panel_idx, ev, str(tmp_path), num_threads=4)
    assert deg_count >= 10 and deg_hash                          # cell-eval/pdex found DE genes
    assert metrics["good"]["delta_pearson"] > metrics["bad"]["delta_pearson"]
    assert metrics["good"]["de_lfc_spearman"] is not None
    assert metrics["good"]["energy_distance"] <= metrics["bad"]["energy_distance"]


# --- tiny end-to-end model + manifest smoke test ---------------------------
class _Block(torch.nn.Module):
    def forward(self, x, pos, mask, ret): return x + 0.01, None

class _Fake(torch.nn.Module):
    """Minimal StateICL-shaped model: 9 blocks, per-cell NB head, decode()."""
    def __init__(self, n_genes=400, n_hidden=4, token_dim=2):
        super().__init__()
        self.n_cells = 512; self.n_genes = n_genes
        self.reduce = torch.nn.Linear(n_genes, n_hidden * token_dim)
        self.head = torch.nn.Linear(n_hidden * token_dim, n_genes * 2)
        self.query_pos_embedding = torch.nn.Parameter(torch.zeros(n_hidden, token_dim))
        self.gene_pos_embedding = torch.nn.Parameter(torch.zeros(n_hidden, token_dim))
        self.layers = torch.nn.ModuleList([_Block() for _ in range(9)])
        self._nh, self._td = n_hidden, token_dim
    def _reduce_and_tokenize(self, x):
        return self.reduce(x).reshape(x.shape[0], x.shape[1], self._nh, self._td)
    def _compute_nb_parameters(self, final, lib):
        out = self.head(final.reshape(-1, self._nh * self._td)).reshape(final.shape[0], final.shape[1], self.n_genes, 2)
        px = torch.softmax(out[..., 0], -1)
        return px * lib, torch.nn.functional.softplus(out[..., 1]), px
    def decode(self, emb, lib):
        e = emb.unsqueeze(0) if emb.ndim == 2 else emb
        l = lib.reshape(1, -1, 1) if lib.ndim <= 1 else lib
        return self._compute_nb_parameters(e, l)[0].squeeze(0)


def test_run_manifest_emits_all_method_metric_rows(tmp_path):
    G = 400
    genes = [f"g{i}" for i in range(G)]
    al = GeneAligner(genes, genes)
    rng = np.random.default_rng(3)
    ids = {f"p0_{i}": rng.poisson(5, G).astype(float) for i in range(4)}
    ids |= {f"p1_{i}": rng.poisson(9, G).astype(float) for i in range(4)}
    ids |= {f"q0_{i}": rng.poisson(5, G).astype(float) for i in range(4)}
    ids |= {f"y1_{i}": rng.poisson(9, G).astype(float) for i in range(4)}
    fine = {c: ("CD4" if int(c.split("_")[1]) < 2 else "CD8") for c in ids if c[:2] in ("p0", "p1")}
    fine |= {c: "B" for c in ids if c[:2] in ("q0", "y1")}
    m = TaskManifest("H2D2", "IL-6", "B",
                     tuple(f"p0_{i}" for i in range(4)), tuple(f"p1_{i}" for i in range(4)),
                     tuple(f"q0_{i}" for i in range(4)), tuple(f"y1_{i}" for i in range(4)),
                     {}, tuple(genes), 1, 0).payload() | {"task_id": "t", "manifest_hash": "h"}
    rows = run_manifest(_Fake(G), al, lambda c: ids[c], lambda c: fine[c], m, "cpu", seed=0,
                        outdir=str(tmp_path), num_threads=2)
    assert len(rows) == 12 and len({r["method"] for r in rows}) == 4
    assert {r["metric"] for r in rows} == {"delta_pearson", "de_lfc_spearman", "energy_distance"}
    for r in rows:
        if r["comparison_table"] == "representation_controlled":
            assert r["transport_config_hash"] and r["expression_basis_hash"] and r["decoder_train_ids_hash"]
        assert r["value"] is None or isinstance(r["value"], float)   # numeric (nan tolerated on degenerate fake data)
