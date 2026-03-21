# Top Candidates

## Operational Frontier
- Operational SOTA for this pass is local `pr332:records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA` at `val_bpb=1.1320`, `bytes_total=15,652,352`.
- The merged upstream README still shows the March 20 winner at `1.1428`, so the root port is intentionally targeting the ahead-of-README PR frontier.

## Active Root Mainline
- root `train_gpt.py` is now a near-verbatim `#332` port.
- Exact baseline config: `configs/h100/root_pr332_repro.json`
- Main improvement config: `configs/h100/root_pr332_b458k_bigram4096.json`
- Eval-only follow-up: `configs/h100/root_pr332_b458k_bigram4096_eval4096.json`

## Main Prediction
- Prediction: `#332 + TRAIN_BATCH_TOKENS=458752 + ITERATIONS=11000 + BIGRAM_VOCAB_SIZE=4096` is more likely than not to beat exact `#332` on post-export `val_bpb`.
- This is an inference, not a measured claim.

## Why This Delta
- `#332` reports `15,652,352` bytes, leaving `347,648` bytes of cap headroom under `16,000,000`.
- Doubling `BIGRAM_VOCAB_SIZE` from `2048` to `4096` adds `2048 * 128 = 262,144` weights.
- At an int6-like payload this is about `196,608` raw bytes, plus about `4,096` bytes for additional fp16 row scales, leaving margin before compression.
- A nearby accepted record already showed that more BigramHash capacity helped: `8192 -> 10240` improved by `0.0008 bpb` in `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md`.
- `#332` also explicitly argues that moving from `786432` to `524288` batch tokens improved quality because it bought about 22% more optimization steps in the same budget.
- A further reduction from `524288` to `458752` is a conservative 12.5% step-cost cut. If step time scaled roughly with tokens, `74 ms -> ~64.75 ms`, which implies roughly `9260` train steps in 600 seconds instead of `~8054`.
- Because of that, `ITERATIONS` must be raised above `9000`; otherwise the run risks hitting the static iteration ceiling before the wallclock limit.

## Immediate Ladder
- Run 1: exact `#332` root reproduction
- Run 2: exact `#332` plus `TRAIN_BATCH_TOKENS=458752`, `ITERATIONS=11000`, and `BIGRAM_VOCAB_SIZE=4096`
- Run 3: Run 2 plus `EVAL_SEQ_LEN=4096 EVAL_BATCH_SEQS=16` only if Run 2 stays inside the eval budget

## Ranking Policy
- Rank by lower post-export `val_bpb`
- Require `bytes_total < 16,000,000`
- Break close ties by higher train tokens processed under the same 600-second budget

## Deferred Ideas
- `EVAL_SEQ_LEN=4096` is a run-level follow-up, not the primary code delta
- Recurrent/shared-width transformers remain YELLOW
- Sparse high-precision outlier retention remains YELLOW
- Triton remains blocked until a winning post-`#332` GREEN lane exists
