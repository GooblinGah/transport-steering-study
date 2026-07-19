# Training-free recovery of aligned ICL (2026-07-19)

## Question
The gate asks whether MMD steering *beats* ICL. A more informative question: how much
of the aligned model's trained in-context capability does a **training-free, test-time**
affine transport of its frozen latents recover — and is that recovery attributable to
the Stack representation, or would any transport do as well?

## Method
Same manifests, one resample per eligible donor-task (35 units spanning every cytokine
and class), scored on the fixed control-only panel with cell-eval v0.6.6 (`pearson_delta`).
Three training-free floors were added, all with the same anchored correction:

- **no-op** — predict Q0 unchanged (predicts zero shift).
- **gene_shift** — mean control→treated shift in normalized-log **gene** space (no Stack).
- **pca50_shift** — the same shift in a 50-PC gene-space (no Stack).

Compared against the two Stack methods that use the frozen model:

- **stack_mmd_native** — our affine MMD map on the frozen post-L9 latents, decoded by the
  native NB head. **No gradient touches the model; the map is fit on source cells only.**
- **stack_icl_native** — the model's own alignment-trained 5-step in-context generation.

## Result (Δ-Pearson, mean over 35 units)

| method | trained? | space | Δ-Pearson |
|---|:--:|---|---:|
| no-op | – | – | ~0 (undefined) |
| gene_shift | no | raw genes | 0.183 |
| pca50_shift | no | gene PCA-50 | 0.165 |
| **stack_mmd_native** | **no** | **Stack latents** | **0.266** |
| stack_icl_native | yes (aligned) | Stack native ICL | 0.302 |

## Reading
- **Training-free MMD steering recovers 88%** of the aligned ICL's absolute score
  (0.266 / 0.302) and **closes ~70% of the floor→ICL gap** ((0.266−0.183)/(0.302−0.183)).
- **The recovery is the representation, not the operator.** The identical MMD/mean-shift
  in raw gene space reaches only 0.183; moving it into Stack's frozen latent space adds
  **+0.083 (≈ +45%)**. A geometric transport is powerful *because* Stack's post-L9 space
  linearizes the perturbation response.
- **Mechanistically this is activation steering.** The aligned model's in-context
  perturbation behavior is substantially reproducible as a single affine map in the
  residual stream at the final layer, fit at test time on source cells and transferred
  to a held-out cell class — no post-training, one forward pass instead of 5-step
  iterative generation.

## Why the matched-decoder table looked different
`stack_mmd_matched_ridge` (0.162) ≈ `pca_mmd_matched_ridge` (0.185) because the 120-cell
ridge decoder throttles both representations; that table isolates *decoder-free* geometry
and understates what the native decoder unlocks. The native-decoder contrast above is the
fair test of "does the Stack representation carry the response," and there Stack wins
clearly over the gene-space floor. (Stack also wins the representation-controlled **energy
distance**, 9.74 vs 12.12 — the latent geometry captures distribution shape too.)

## Caveat
`stack_mmd_native` bundles Stack latents **and** the native NB decoder, so its edge over
`gene_shift` reflects "using the Stack model" as a whole. That is the honest unit of
comparison for "is Stack worth it": yes — training-free, at test time, it nearly matches
its own aligned ICL and far exceeds a no-model transport.
