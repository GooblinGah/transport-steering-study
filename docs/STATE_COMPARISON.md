# STATE comparison: our steering vs Arc's State Transition model (2026-07-19)

Repeats the study with Arc's **STATE** (paper: Adduri et al., biorxiv 2025.06.26.661135):
the State **Embedding** model (SE-600M) as the representation for our steering, and the
State **Transition** model (ST-SE-Parse) as the trained perturbation predictor.

## The two things that made the first attempt misleading
An earlier version ran STATE on the **Dong** data with a **matched ridge** decoder for our
steering and STATE's native decoder for STATE, and reported a near-tie (~0.18 vs ~0.19).
Both choices were wrong, and the paper explains why:

1. **Off-label dataset transfer.** The paper is explicit that ST does "context-level
   generalization **within a dataset** rather than **zero-shot dataset transfer**." ST-SE-Parse
   is trained on Parse; applying it to Dong (different lab/platform) is outside its envelope,
   so everything collapsed toward the floor.
2. **Unequal decoders.** Our steering got a weak 120-cell ridge (R²≈0.26 on Parse) while STATE
   got its trained decoder. Not a fair transformation-vs-transformation test.

Also note: the "floor" throughout this study is exactly the paper's own **"perturbation mean"
baseline** (average the observed perturbed−control shift, apply to controls). The paper openly
states deep models "do not consistently outperform linear models" for context generalization.

## The correct experiment: in-distribution Parse, STATE's own decoder for both
- **Dataset:** Parse (STATE's home turf). Held-out cell type **B_Intermediate_Memory**
  (STATE's own zeroshot split_0). Source cell type: CD14_Mono. Real Parse cells, SE embeddings,
  shared 2000-HVG space — anonymized gene names are irrelevant because every method lives in
  the same 2000 columns and is scored against real Parse cells.
- **Same decoder for both:** both our MMD-steered SE embedding **and** STATE's transition
  prediction are decoded by **STATE's actual `gene_decoder`** (loaded from the model, not
  rebuilt; it reconstructs real cells at mean-corr 0.92). Only the transformation differs
  (our `LowRankMMD` vs STATE's trained ST transformer).
- STATE uses its **official** predictions (`adata_pred`) — its best, native, in-distribution result.
- 12 cytokines; cell-eval `pearson_delta`, `de_spearman_lfc_sig`, energy distance.

## Result (mean over 12 cytokines)

| method | transform | decoder | Δ-Pearson | DE-LFC | energy ↓ |
|---|---|---|---:|---:|---:|
| **STATE transition** | trained ST | STATE gene_decoder | **0.472** | **0.796** | 0.475 |
| our MMD steering | our LowRankMMD | STATE gene_decoder | 0.261 | 0.358 | 1.516 |
| perturbation-mean floor | mean shift | none (raw genes) | 0.206 | 0.633 | 0.424 |

## Findings

**1. STATE's transition model clearly wins in-distribution.** ~0.47 vs ~0.26 delta-Pearson
(STATE wins 9/11 scored cytokines), and it dominates DE-LFC and energy. The ~0.47 matches the
paper's reported Parse performance, so the pipeline is sound. **Roughly 2× our steering.**

**2. Our training-free steering beats the naive floor only on mean-shift correlation**
(0.261 vs 0.206) and is *worse* than the floor on DE-LFC (0.36 vs 0.63) and energy (1.52 vs
0.42) — it nudges the average direction right but is distributionally noisy.

**3. The Stack result does NOT replicate for STATE.** On Stack, training-free MMD steering
~matched Stack's ICL; here it is nowhere near STATE's purpose-built transition model. Stack's
in-context learning is a weaker/different bar than STATE's trained transition model on Parse.

## Caveats
- STATE's `gene_decoder` was trained on STATE's *own* predicted latents, so decoding our
  MMD-transported embeddings through it is slightly out-of-distribution for the decoder — but
  it is the fairest shared decoder available, and it does not close the gap.
- 12 cytokines, one held-out cell type, one source type; descriptive.
- CD14_Mono→B_Intermediate_Memory is a single cross-cell-type transfer; the Dong design averaged
  more source/target combinations.

## Bottom line
Given a fair, in-distribution, same-decoder test, **STATE's trained transition model is ~2×
better than our training-free steering** and clearly beats the perturbation-mean floor. The
earlier "steering ties the transition model" was an artifact of off-label data + an unequal
decoder. Our steering's only edge over the naive baseline is a modest bump in mean-shift
correlation; it loses on DE recovery and distribution shape.

## Downloads (beyond the raw Dong data)
`arcinstitute/SE-600M` (~14 GB), `arcinstitute/ST-SE-Parse` (checkpoints + eval adatas for
splits 0 and 4). `arc-state` installed in an isolated `.venv-state` (its triton 3.7 conflicts
with Stack's torch 2.5.1). The 52 GB `State-Parse-Filtered/parse_concat_full.h5ad` was not
needed — the per-split eval adatas provide real cells with SE embeddings.
