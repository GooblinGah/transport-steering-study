# STATE comparison scripts

Reproduce the STATE phase (see `docs/STATE_COMPARISON.md`). Run against the isolated
`.venv-state` (arc-state) for embedding/inference and the main `.venv` for steering/eval.

1. `state emb transform --model-folder artifacts/models/SE-600M --input <dong_cells>.h5ad
   --output <se_emb>.h5ad --embed-key X_state` — SE-600M cell embeddings.
2. `floor_baseline.py` — Stack-free gene-space floors (no-op, gene mean-shift, PCA-50).
3. `state_mmd.py` — our LowRankMMD steering on SE embeddings + matched ridge decode.
4. `state tx infer --model-dir artifacts/models/ST-SE-Parse/zeroshot/split_0 --embed-key
   X_state --pert-col cytokine --control-pert PBS --all-perts` — native ST predictions.
5. `state_transition.py` — decode ST's predicted SE embedding via the same matched ridge.

Paths are the scratch paths used during the run; adjust for your environment.
