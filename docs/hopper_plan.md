# Hopper Plan

Active valid frontier for this pass:
- accepted merged record: `#549`
- latest plausible valid open leader to port: `#1060`
- completed local reference baseline: `logs/root-pr549-softqat.txt` at `1.11956668`
- reference baseline snapshot: `snapshots/train_gpt_2026-03-25_pre753_pr549_softqat_root.py`
- archived pre-pivot dense-GPTQ snapshot: `snapshots/train_gpt_2026-03-29_pre1060_valid_dense_gptq_root.py`
- archived GDN snapshot: `snapshots/train_gpt_2026-03-27_prepivot_pr875_gdn_root.py`
- active root delta: `#1060`-derived 11-layer scaffold with reserved-time full-Hessian GPTQ6, a prune-funded `BigramHash(3072,112)` upgrade, and Gemma-style hybrid attention `L,L,G,L,L,G,L,L,G,L,G`

## Run Order
- Run 1: Gemma-style hybrid `#1060`-derived promoted lane
- Optional proxy only: `configs/h100/root_pr1060_gemma_hybrid_proxy_1xh100.json` on one H100 if you need a cheap directional read; do not treat it as timing or legality proof for the 8x lane.

## Run 1
- Goal: keep the strongest plausible valid open `#1060` family intact, then spend exactly one architecture delta on Gemma-style hybrid local/global attention while preserving the prune-funded `BigramHash(3072,112)` follow-up.
- Command generation: `py -3.11 scripts/prepare_h100_run.py configs/h100/root_pr1060_gemma_hybrid_b3072_prune.json`
- Success criteria: beat the completed `#589`-style reference baseline of `1.11956668`, stay under `16,000,000` bytes, keep total train-plus-calibration wallclock inside `600s`, and keep the final record-facing metric on `sliding_window_exact`.
- Risk: full-Hessian reserved-time GPTQ may consume more of the 600s budget than the retired in-training collector lane.
- Pod sanity check: `rg -n "ATTN_PATTERN|LOCAL_ATTN_WINDOW|TRAIN_LOADER_MODE|GPTQ_RESERVE_MS|gptq:start reserved_train_data|gptq:calibrated|ttt_" train_gpt.py`
- Timing expectation: unknown until the first H100 truth run; no second H100 lane is active until this base lands cleanly.

## Notes
- `#753` / eval-cache / hybrid work remains archived pending rule clarification and is not part of the active record ladder.
- The `#875` GDN branch is archived after the 1xH100 proxy landed far off-family and did not justify more scarce H100 runs.
- `#1047` and `#1056` are excluded from the active ladder for validity reasons; `#875` is excluded because the reported score is not trustworthy.
- Tokenizer WIP is archived for this pass and should not be mixed into the active ladder.
- The completed `#549` / `#589` local truth run is reference-only now; do not spend more H100 compute replaying accepted donor PRs.
- Exact `#1060` donor parity is a local sanity lane only; do not spend an H100 run on the pure donor replay.
- `flash-attn` remains optional and cluster-side.
