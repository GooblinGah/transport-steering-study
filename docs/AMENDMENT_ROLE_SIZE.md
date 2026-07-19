# Amendment: role-size tiers (2026-07-19)

## Context
`docs/DONG_CONTRACT.md` §5 preregisters role sizes 512, 256, 128 and requires a
recorded amendment before adding a smaller size. On the deposited Dong data
(`artifacts/manifests/CELL_COUNT_AUDIT.json`) **no preregistered tier is feasible**:
zero of 36 donor-tasks qualify at any of 512/256/128.

## Cause
The held-out-target rule needs the full role size in the target class, and the
source rule needs role_size/2 per source broad class. Two fine classes are scarce
in the two study donors (H2D2, H3D2):

| fine class | cells per donor-condition pool (min / median / max) |
|---|---|
| B         | 36 / 58 / 135 |
| Dendritic | 4 / 6 / 19 |

B cells cannot supply 128 as a target, nor 64 (=128/2) as a source, so every tier
≥128 fails.

## Feasibility sweep (metadata-only, no Y1 expression used)
Reusing the unchanged audit logic across candidate sizes:

| role size | eligible units | B units | every class? | every cytokine? | gate coverage |
|---:|---:|---:|:--:|:--:|:--:|
| 128 | 0  | 0 | no  | no  | fail |
| 64  | 12 | 0 | **no** | yes | fail (no B) |
| 48  | 14 | 2 | no (B<4) | yes | fail |
| **40** | **17** | **5** | **yes** | **yes** | **PASS** |
| 36  | 18 | 6 | yes | yes | PASS |
| 32  | 18 | 6 | yes | yes | PASS |

## Decision
Append `64, 48, 40, 36, 32` to `ROLE_SIZES`. The **preregistered selection rule is
unchanged** ("largest tier meeting all coverage requirements"), and it now selects
**40** — the highest-powered feasible size. 40 passes every gate-coverage
requirement: 17 ≥ 12 eligible two-donor units, every broad class present (B=5),
every cytokine covered. This is the minimal, principled amendment: extend the
candidate list, do not change the rule.

Alternatives 36/32 give the full 18/18 units at ~10–20% fewer cells per role and
remain available if a future run prefers complete unit coverage over per-unit power.

## Inference scope (unchanged)
Two observed donors; results are descriptive, not population-level significant.
