# STATE comparison: our steering vs Arc's State Transition model (2026-07-19)

Repeats the study with Arc's **STATE** models: the State **Embedding** model (SE-600M) as
the representation for our steering, and the State **Transition** model (ST-SE-Parse) as the
trained perturbation predictor. Run on a 35-unit subset (one resample per eligible
donor-task, spanning every cytokine and class) so it is directly comparable to the Stack
metrics on the same manifests.

## Fair-comparison design (as specified)
The transformation is the *only* thing that differs between our steering and the transition
model:

- **state_mmd** — our identical `LowRankMMD` map, fit on SE(P0)→SE(P1), applied to SE(Q0).
- **state_transition** — STATE's trained ST model, applied to the same SE(Q0).
- **Both** produce a predicted perturbed **SE embedding** (2058-dim), decoded to expression by
  the **same** matched ridge decoder (fit per task on SE→expression), with the same anchored
  synthetic-control correction, evaluated by the same cell-eval panel.

STATE-SE genes: 15128/21819 Dong genes mapped to ESM embeddings. Cytokines mapped to the ST
vocabulary (IFN-β→IFN-beta, IFN-γ→IFN-gamma, IL-6, TNF-α→TNF-alpha, IFN-α2→IFN-alpha1,
IFN-III/IL-29→IFN-lambda1); held-out classes to ST cell types (B→B_Naive, CD4→CD4_Memory,
CD8→CD8_Memory, Monocyte→CD14_Mono, Dendritic→cDC); control PBS.

## Results (mean over 35 units)

| method | representation | transform | decoder | Δ-Pearson | energy↓ |
|---|---|---|---|---:|---:|
| Stack ICL | Stack | trained ICL | native NB | **0.302** | **7.54** |
| Stack-MMD | Stack | our MMD | native NB | 0.266 | 7.94 |
| **STATE transition** | STATE-SE | **trained ST** | matched ridge | **0.193** | 8.42 |
| PCA-MMD | gene PCA-50 | our MMD | matched ridge | 0.185 | 12.09 |
| **STATE-MMD** | STATE-SE | **our MMD** | matched ridge | **0.181** | 9.33 |
| Stack-MMD (matched) | Stack | our MMD | matched ridge | 0.162 | 9.84 |

## Findings

**1. Our steering ≈ STATE's transition model (the headline).** On the fair SE-space,
matched-decoder comparison, training-free MMD steering (0.181) essentially ties the trained
State Transition model (0.193): our steering wins 16/35 units, mean diff −0.012 — within the
per-unit noise. Test-time steering recovers **~94%** of the trained transition model's
delta-Pearson. This mirrors the Stack finding (Stack-MMD ≈ Stack-ICL) on a second, independent
foundation model.

**2. STATE-SE is a competitive steering representation.** Under identical MMD + matched
decoder, STATE-SE (0.181) ≈ PCA (0.185) > Stack-matched (0.162) on mean-shift, and STATE-SE
has the best energy distance of the matched-decoder methods (9.33 < 9.84 < 12.09).

**3. Across trained predictors, Stack's ICL leads on Dong.** Stack ICL (0.302) > State
Transition (0.193). But note the asymmetry: Stack ICL uses Stack's native NB decoder, while
State Transition here is decoded by our matched ridge (its own 2000-HVG gene decoder could not
be aligned to Dong — the HVG gene names are not published with the ST-SE-Parse artifacts, and
the "first-2000-gene_names" hypothesis was tested and rejected, delta-corr 0.03). So this
cross-model number isolates the SE-space *transformation + shared ridge*, not STATE's full
native pipeline, and likely understates the State Transition model.

## Caveats
- 35 units, one resample each; two donors; descriptive only (per-unit std ≈ 0.18).
- STATE-SE steering used fixed MMD hyperparameters (rank 8, α 1) — same un-tuned config as Stack.
- ST cell-type/cytokine label mapping is approximate (IFN-α2→IFN-alpha1, broad→fine cell types).
- The State Transition model is evaluated in SE-embedding space + shared ridge, not its native
  2000-HVG gene decoder (gene names unavailable). This makes the state_mmd vs state_transition
  contrast fair (same decoder) but caps the cross-model comparison to Stack.

## Downloads
Beyond the raw Dong data, this phase downloaded from Hugging Face: `arcinstitute/SE-600M`
(~14 GB incl. checkpoint) and `arcinstitute/ST-SE-Parse` (zeroshot/split_0, ~0.5 GB). So the
raw data is **not** the only required download once STATE is involved. (For the Stack phase,
the Stack checkpoint was already present under `artifacts/models/`, so there the raw data is
the main external input.)
