# Top Candidates

## Operational Frontier
- Accepted upstream README still shows the March 20 winner at `1.1428`.
- The active reference donor for root remains local `pr332:records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA` at `1.1320`.
- The new root lane is not a clone of later PRs; it is a PR529-inspired dense reset that keeps root self-contained and under the line cap.

## Active Root Mainline
- Frozen fallback snapshot: `snapshots/train_gpt_2026-03-23_root_shared10_qgain_outliers64.py`
- Dense fallback snapshot: `snapshots/train_gpt_2026-03-23_root_pr332_b458k_b1024_warmup200_xlast2.py`
- Dense H100 truth baseline: `configs/h100/root_snapshot_b1024_warmup200_xlast2.json`
- GPTQ-only dense baseline: `configs/h100/root_gptq12_b1024_warmup200_xlast2.json`
- Funded dense GPTQ lane: `configs/h100/root_gptq14_mlp1792_b4096_warmup200_xlast2.json`
- Funded dense GPTQ lane + TTT-lite: `configs/h100/root_gptq14_mlp1792_b4096_warmup200_xlast2_ttt.json`

## Main Prediction
- The shared-depth / q-gain / sparse-outlier branch improved local A100 results but saturated far above the target range.
- The next meaningful jump is more likely to come from real stored-byte savings in the exporter plus a denser model, not more micro-tuning of the shared lane.
- Packed low-bit GPTQ-style export should buy real artifact headroom on the dense root baseline.
- That headroom is best spent first on `NUM_LAYERS=14`, `MLP_HIDDEN=1792`, and `BIGRAM_VOCAB_SIZE=4096`.
- TTT-lite is an explicit yellow follow-up, not the new default. It should only be judged after the GPTQ-funded dense lane is healthy on bytes and exact roundtrip.

## Why This Delta
- The previous root branch was spending complexity on shared-weight specialization tricks while still losing too much on post-export quality.
- GPTQ-style packing targets the actual bottleneck: large matrix storage cost inside the artifact cap.
- A funded dense model is a cleaner use of extra bytes than more parameter reuse once the shared lane has flattened.
- TTT-lite is kept doc-isolated, eval-only, and narrow:
  - `lm_head`
  - `c_q` and `c_v` of the last 4 blocks
  - one update per chunk after scoring
  - reset between documents

## Immediate Ladder
- Run 1: frozen dense snapshot on H100
- Run 2: root GPTQ-only dense baseline
- Run 3: root GPTQ-funded dense lane with `NUM_LAYERS=14`, `MLP_HIDDEN=1792`, and `BIGRAM_VOCAB_SIZE=4096`
- Run 4: Run 3 plus `TTT_ENABLED=1`

## Ranking Policy
- Rank by lower exact roundtrip `val_bpb` first.
- Treat sliding-window `val_bpb` as the secondary optimization target for tie-breaking and follow-up eval work.
- Require `bytes_total < 16,000,000`.
- Require H100 truth before promotion.
