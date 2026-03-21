# Top Candidates

## Operational Frontier
- Operational SOTA for this pass is local `pr332:records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA` at `val_bpb=1.1320`, `bytes_total=15,652,352`.
- The merged upstream README still shows the March 20 winner at `1.1428`, so the root port is intentionally targeting the ahead-of-README PR frontier.

## Active Root Mainline
- root `train_gpt.py` is now a near-verbatim `#332` port.
- Exact baseline config: `configs/h100/root_pr332_repro.json`
- Main improvement config: `configs/h100/root_pr332_b458k_bigram2560.json`
- Fallback config: `configs/h100/root_pr332_b458k_bigram2432.json`
- Eval-only follow-up: `configs/h100/root_pr332_b458k_bigram2560_eval4096.json`
- The newest fixed-metric `8xA100` rerun in `logs/root-pr332-b458k-bigram3584 (2).txt` is plausible but still not submission-safe because it is over cap at `16,069,767` bytes and is not `8xH100` truth.

## Main Prediction
- Prediction: `#332 + TRAIN_BATCH_TOKENS=458752 + ITERATIONS=11000 + BIGRAM_VOCAB_SIZE=2560` is the safest next compliant rescue of the strong fixed-metric `b458k` lane.
- This is an inference, not a measured claim.

## Why This Delta
- `#332` reports `15,652,352` bytes, leaving `347,648` bytes of cap headroom under `16,000,000`.
- The newest fixed-metric `3584` rerun came in at `16,069,767` total bytes with a compressed model size of `16,004,657`, so the active problem is a `69,767`-byte overage in the model blob rather than code size.
- At the current ratio `16,004,657 / 28,057,988 ~= 0.5704`, each BigramHash row is worth about `130 * 0.5704 ~= 74.2` compressed bytes, using `128` quantized bytes plus `2` bytes of per-row scale as the raw estimate.
- `3584 -> 2560` removes `1024` rows and is the first clean trim that should save about `75,933` compressed bytes at the current ratio.
- `3584 -> 2432` is the fallback if `2560` lands under cap with too little margin; it should save about `85,425` compressed bytes at the current ratio.
- A nearby accepted record already showed that more BigramHash capacity helped: `8192 -> 10240` improved by `0.0008 bpb` in `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md`.
- `#332` also explicitly argues that moving from `786432` to `524288` batch tokens improved quality because it bought about 22% more optimization steps in the same budget.
- A further reduction from `524288` to `458752` is a conservative 12.5% step-cost cut. If step time scaled roughly with tokens, `74 ms -> ~64.75 ms`, which implies roughly `9260` train steps in 600 seconds instead of `~8054`.
- Because of that, `ITERATIONS` must be raised above `9000`; otherwise the run risks hitting the static iteration ceiling before the wallclock limit.

## Immediate Ladder
- Run 1: rerun the exact `TRAIN_BATCH_TOKENS=458752`, `ITERATIONS=11000`, `BIGRAM_VOCAB_SIZE=2560` branch with the fixed metric code
- Run 2: use `BIGRAM_VOCAB_SIZE=2432` only if Run 1 is still over cap or lands under cap with too little margin
- Run 3: Run 1 plus `EVAL_SEQ_LEN=4096 EVAL_BATCH_SEQS=16` only if Run 1 stays inside the eval budget and lands comfortably under cap

## Ranking Policy
- Rank by lower post-export `val_bpb`
- Require `bytes_total < 16,000,000`
- Break close ties by higher train tokens processed under the same 600-second budget

## Deferred Ideas
- `EVAL_SEQ_LEN=4096` is a run-level follow-up, not the primary code delta
- Recurrent/shared-width transformers remain YELLOW
- Sparse high-precision outlier retention remains YELLOW
- Triton remains blocked until a winning post-`#332` GREEN lane exists
