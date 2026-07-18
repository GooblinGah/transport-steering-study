# Dong Test Contract

## 1. Purpose

The Dong test answers three questions.

1. Does explicit Stack transport perform better than Stack in-context learning?
2. Does the Stack representation perform better than a 50-component PCA representation?
3. Does source-only state information improve a global transport map?

The test uses two donors, six cytokines, and three held-out broad cell classes.
This design gives 36 donor tasks and 18 donor-averaged units.

## 2. Terms

- **Source cells** are cells from the two broad classes that are not held out.
- **Query cells** are control cells from the held-out class.
- **Target cells** are treated cells from the held-out class.
- **Sealed target expression** is target expression that no fit or selection step can use.
- **Unit** is one cytokine and broad-class pair after the two donor results are averaged.
- **Native decoder** is the decoder supplied with a model.
- **Matched decoder** is a decoder with the same design and training policy for each representation.
- **Technical resample** is a repeated calculation with a new cell sample or seed.
- **Gate** is the set of numerical requirements that controls study expansion.

## 3. Result tables

### 3.1 End-to-end table

Use deterministic means from the native Stack negative-binomial decoder.
Use the official Stack in-context learning procedure with deterministic mean output.
Use PCA inverse transformation for the end-to-end PCA result.

Use this table to assess explicit transport against in-context learning.
Do not use this table to assess Stack representation quality against PCA representation quality.

### 3.2 Representation-controlled table

Apply the same MMD transport operator to Stack coordinates and PCA coordinates.
Use the same transport grid and donor cross-fit rule for both representations.

Fit one common expression target basis for each task and fold.
Use 50 expression principal components in this basis.
Fit the basis on the ordered `P0`, `P1`, and `Q0` cells only.

Fit one ridge decoder for each representation.
Use the same ordered training cells, output genes, folds, and regularization grid.
Use standardized latent coordinates as decoder input.
Map the coordinates to the common expression target basis.

Record the decoder training-cell hash and the expression-basis hash.
Require equal hashes for the Stack and PCA decoder pair.
Reject a decoder if its training cells overlap `Y1`.

Use only this table to assess Stack transport geometry against PCA transport geometry.

## 4. Output scale

Gene and PCA methods operate in normalized log-expression space.
First, normalize each input cell to 10,000 counts.
Then, apply `log1p` one time.

After prediction, apply `expm1`.
Clip negative values to zero.
Rescale each predicted cell to its observed `Q0` library size.

Stack produces deterministic negative-binomial expected counts.
Use the observed `Q0` library size for each Stack query cell.

Apply this anchored correction to all primary predictions:

```text
corrected = clip(Q0 + perturbed_mean - noop_mean, 0)
```

Rescale the corrected cell to its `Q0` library size.
Record the mass and fraction removed by clipping.

This anchored correction is a rule of this test.
The official Stack tutorial uses a different procedure.
It compares a generated treated profile with a separately generated control profile.
Report that official tutorial contrast in a secondary table.

Store all primary method outputs as expected counts.
Apply the common evaluation transform exactly one time.
The common transform is normalization to 10,000 counts followed by `log1p`.

## 5. Cell classes and composition

Use these broad and fine classes.

- B: B cells
- T: CD4 T cells and CD8 T cells
- Myeloid: monocytes and dendritic cells

Exclude NK cells and plasma cells.
Reject an unknown fine label.

Run a metadata-only cell-count audit before you select a role size.
The audit can use `Y1` labels for count and composition checks.
The audit cannot use `Y1` expression.

Test role sizes of 512, 256, and 128 cells.
Select one role size for all eligible tasks.

Balance `P0` and `P1` equally across the two source broad classes.
Use identical fine-label quotas in `P0` and `P1`.
Use identical fine-label quotas in `Q0` and `Y1`.

Select quotas that are nearest to the control composition.
Use treated-cell availability as a capacity limit.
Require four available cells for each required composite fine label.
Select at least one cell from each required fine label.

Record an ineligible task and its reason.
Do not stop the full audit because one task is ineligible.

A role-size tier must meet all these coverage requirements.

- It must contain at least 12 of the 18 two-donor units.
- It must contain at least four units for each held-out broad class.
- It must contain at least two broad classes for each cytokine.

Stop model execution if no listed role size meets these requirements.
Add a smaller role size only through a recorded study amendment.

For the secondary natural-composition table, keep `P0` and `P1` fine-label matched.
Sample `Q0` and `Y1` independently from their natural compositions.

## 6. Stack activation context

Stack uses attention between cells in each 512-cell window.
Thus, record the membership and composition of every window.
Also, record the cell order and the window seed.

### 6.1 Source-map fit

Encode `P0` and `P1` in separate condition-homogeneous windows.
Use this layout for a `P0` window:

```text
333 P0 context rows + 179 P0 focal rows
```

Use the equivalent layout for a `P1` window.
Match the fine-label sequence by position between the two windows.

Do not put `Q0` or `Y1` in a source-fit window.
Fit the map only on focal rows 333 through 511.

Rotate source barcodes through focal positions.
Make sure that each source barcode has focal activation coverage.
Average repeated activations for each barcode before you fit MMD.

The pinned implementation starts the query-position marker at row 332.
The true focal segment starts at row 333.
Keep the row-332 behavior in the primary parity run.
Use row 333 in a sensitivity run.

### 6.2 Query application

Use this layout:

```text
333 P0 context rows + 179 Q0 query rows
```

Modify only the true `Q0` query rows.
Do not put `Y1` in the window.

### 6.3 Context sensitivity tests

Run these sensitivity tests.

- Start the query-position marker at row 333.
- Use the same `P0` anchor context for both source conditions.
- Use all five official context ratios.
- Use alternative matched context compositions.
- Permute cells within each segment.

## 7. Stack in-context learning

Use five generation steps.
Use a prompt ratio of 0.25.
Change the context ratio from 0.20 to 0.40.

Use this exact schedule.

| Step | Mask ratio | Source rows | Query rows | Query marker start | Source seed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8 | 231 | 281 | 230 | 1 |
| 2 | 0.6 | 256 | 256 | 256 | 2 |
| 3 | 0.4 | 282 | 230 | 281 | 3 |
| 4 | 0.2 | 308 | 204 | 307 | 4 |
| 5 | 0.0 | 333 | 179 | 332 | 5 |

Keep the pinned rounding behavior in the official parity table.
In four steps, one source row receives the query-position marker.
Use corrected boundaries in a sensitivity table.

Use deterministic negative-binomial means for primary mean and differential-expression metrics.
Use the unchanged stochastic sampler only in a secondary table.

## 8. Manual Stack forward path

Input raw counts to the manual path.
Calculate each observed library size before tokenization.
Apply `log1p` before the Stack reduction layer.

Extract activations after tokenization and after blocks 1 through 9.
Do not call the native random-mask forward method.

Apply an intervention only to query rows at the selected layer.
Run all remaining blocks after the intervention.
Decode with the native negative-binomial head.
Return the deterministic negative-binomial mean.

Verify that the unmodified manual path equals the native deterministic path.
Use the same window, attention mask, segment boundaries, and library sizes for this test.

## 9. Transport operators

Use this affine MMD map:

```text
T(h) = h + alpha * [b + U * V^T * (h - mu)]
```

Standardize coordinates with `P0` statistics.
Reverse the standardization before decoding.

Tune rank, strength, bandwidth, and movement penalty by donor cross-fit.
Use the same selected MMD configuration for all cytokines and target classes in a donor.

Tune the OT entropy multiplier over `0.01`, `0.05`, and `0.1`.
Multiply each value by the median source squared distance.
Use unbalanced OT only as a sensitivity test.

Include these negative controls.

- Set transport strength to zero.
- Shuffle cytokine maps.
- Use a random vector with matched norm.

## 10. State conditioning

On Dong, prioritize continuous latent neighborhoods and baseline immune or metabolic programs.
Use Tricycle on Dong only as an exploratory analysis.
Use Tricycle as a primary state measure for proliferating SciPlex3 cells.

Fit a state-dependent map to source-only OT barycentric pseudo-targets.
Use marginal MMD only as an additional regularizer.
Do not fit a state-dependent field from marginal MMD alone.

Use this map:

```text
T(h,s) = h + b0 + B*phi(s) + U*diag(1 + C*phi(s))*V^T*(h - mu)
```

Interpret all inferred local fields as source transport.
Do not interpret them as paired-cell treatment effects.

## 11. Gene panels and evaluation

Use a fixed control-only gene panel for the gate.
Select this panel before any method sees `Y1` expression.
Record the panel genes, training-cell hash, selection rule, and panel hash.

Freeze all predictions before you call true differentially expressed genes.
Restrict the differential-expression test universe to the fixed control-only panel.
Mark DE-LFC unavailable when a task has fewer than ten true DE genes.

Also report the paper-comparable analysis.
For that analysis, select 2,000 highly variable genes from `Q0` and `Y1`.
Perform this selection only after all predictions are frozen.
This paper-comparable table cannot change the gate result.

Use cell-eval `v0.6.6` for the primary paper-comparable table.
Use cell-eval `v0.8.1` for a secondary table and the empirical data ceiling.

Fit one fixed 50-component evaluator space on permitted training data.
Use the same evaluator-space hash for all methods in a comparison.

## 12. Gate calculation

Use only the fixed control-only gene panel for the gate.
Calculate exactly four contrasts.

1. Compare native Stack-MMD with native Stack ICL for delta Pearson.
2. Compare native Stack-MMD with native Stack ICL for DE-LFC correlation.
3. Compare matched Stack-MMD with matched PCA-MMD for delta Pearson.
4. Compare matched Stack-MMD with matched PCA-MMD for DE-LFC correlation.

Pair method records within each manifest and seed.
Reject missing or duplicate technical pairs.
Average generation seeds within each manifest.
Then, average the 20 locked cell resamples.

This process gives one effect for each donor, cytokine, and class.
Average the two donor effects equally to make each of the 18 units.
Do not treat technical resamples as biological replicates.

Each contrast must meet all these requirements.

- The macro effect must be positive in each donor.
- The donor-averaged macro correlation improvement must be at least 0.02.
- Stack-MMD must win at least 12 of the fixed 18 units.
- At least 12 two-donor units must be eligible.
- The eligible units must include every cytokine and broad class.

A tie is not a win.
An ineligible or missing unit is not a win.

Within each decoder regime, Stack-MMD macro energy distance must not exceed its comparator.

Use 10,000 crossed cytokine and class bootstrap samples for a descriptive interval.
Keep the two donors as a fixed paired block.
Do not calculate a bootstrap population p-value.
Do not apply a Holm correction.
Do not use the interval limit as a gate requirement.

State this inference limit with each result:

```text
The result describes two observed donors.
It does not show population-level statistical significance.
```

## 13. Expansion controls

Do not start SciPlex3 or STATE unless all Dong gate requirements pass.

For STATE-SE, treat a native reconstruction decoder as provisional.
Use it only after an official decoder passes a round-trip reconstruction test.
Reject an unknown perturbation label.

If the Dong gate fails, stop the expansion.
Report whether explicit transport improved on Stack in-context learning.
Do not claim a Stack representation advantage unless the matched-decoder contrast passes.
