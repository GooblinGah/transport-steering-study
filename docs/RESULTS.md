# Dong gate results (2026-07-19)

Full run: 700 eligible fine-matched manifests (role size 40, 17 two-donor units +
one B unit reaching 18 for delta metrics), 4 methods, 3 metrics → 8,400 records.
Evaluation via cell-eval v0.6.6 (`pearson_delta`, `de_spearman_lfc_sig` with pdex
Wilcoxon DE; energy distance in the fixed 50-PC control-only space).

## Verdict: **GATE FAILS — expansion not authorized.**

All four correlation contrasts fail. Neither hypothesis is supported on the two
observed donors.

## Per-method means (over units; delta/DE-LFC higher = better, energy lower = better)

| method | delta-Pearson | DE-LFC spearman | energy distance |
|---|---:|---:|---:|
| **stack_icl_native** (ICL)            | **0.301** | **0.697** | **7.50** |
| stack_mmd_native (MMD steering)       | 0.275 | 0.637 | 7.88 |
| pca_mmd_matched_ridge                 | 0.185 | 0.529 | 12.12 |
| stack_mmd_matched_ridge               | 0.174 | 0.442 | 9.74 |

DE-LFC is available for only 10/18 units — most tasks have <10 significant DE genes
on the control-only panel (median deg_count = 3; only 304/700 manifests reach ≥10).

## The three questions

**Q1 — Does explicit MMD transport beat Stack's in-context learning? → No.**
Native ICL slightly outperforms MMD steering on every metric: delta-Pearson
0.301 vs 0.275 (donor-avg diff **−0.027**, transport wins 6/18 units, both donors
negative), DE-LFC 0.697 vs 0.637, and energy distance 7.50 vs 7.88 (lower better).
Transport steering is competitive but does not improve on ICL.

**Q2 — Does the Stack representation beat matched PCA? → No advantage on correlation.**
With identical MMD operator and matched ridge decoders, Stack and PCA are tied on
delta-Pearson (0.174 vs 0.185, PCA marginally higher; diff **−0.008**, 7/18) and PCA
is better on DE-LFC (0.442 vs 0.529). The one place Stack wins is **energy distance**
(9.74 vs 12.12) — the representation-controlled energy check is the only gate
criterion that passed, i.e. Stack's latent geometry captures the perturbed
distribution's *shape* better, but that does not translate into a better mean-shift
or DE recovery under a matched linear decoder.

**Q3 — Does state-conditioning improve transport?** Not evaluated: per contract §13
the state-conditioning comparison and any SciPlex3/STATE expansion run only if the
Dong gate passes. It did not.

## Energy checks
- end_to_end: **fail** (Stack-MMD − ICL = +0.39; MMD is distributionally worse).
- representation_controlled: **pass** (Stack − PCA = −2.55; Stack distributionally better).

## Honest caveats
1. **Two donors, descriptive only** — no population-level significance (contract §12).
2. **Small units** — role size 40 (amended down from 128; data too sparse for larger).
   Per-unit std ≈ 0.18 on delta-Pearson, so a −0.01 to −0.03 gap is within noise.
3. **MMD is un-tuned** — fixed hyperparameters (rank 8, α 1, 300 steps) instead of the
   contract §9 donor cross-fit over rank/strength/bandwidth/movement. A tuned map is
   the most likely thing to move the Q1 verdict and is the recommended next step.
4. **DE-LFC underpowered** — the control-only panel (top-variance control genes) often
   excludes the genes that actually respond to treatment, leaving only 10/18 units with
   ≥10 DE genes. This metric is the weakest leg of the gate here.
5. The native NB decoder (trained at scale) dominates the 120-cell matched ridge
   decoder, so end_to_end and representation_controlled absolute scores are not
   directly comparable — by design (contract separates the two tables).

## Bottom line
Explicit MMD transport steering of Stack's final-layer latents did **not** beat Stack's
native in-context learning by the gate's ≥0.02 margin, and under a matched linear decoder
the Stack representation showed no correlation advantage over PCA (only a distributional
energy-distance one). Expansion is not authorized.

**But the more informative reading** (see `RECOVERY_ANALYSIS.md`): the *training-free*
MMD steering recovers **88%** of the aligned ICL's Δ-Pearson (0.266 vs 0.302) and closes
**~70%** of the gap to a no-foundation-model gene-space floor (0.183) — an affine map in
the frozen residual stream, fit on source cells at test time, nearly matching the model's
own alignment-trained 5-step inference. That recovery *is* the Stack representation: the
same operator in gene space reaches only 0.183.

Highest-value follow-ups: (1) donor-cross-fit MMD tuning (§9); (2) repeat the whole
comparison with Arc's **STATE** model (see `STATE_COMPARISON.md`).
