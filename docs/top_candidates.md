# Top Candidates

## Operational Frontier
- Operational SOTA for this pass is local `pr332:records/track_10min_16mb/2026-03-21_12L_GradQuant_PartialRoPE_EMA` at `val_bpb=1.1320`, `bytes_total=15,652,352`.
- The merged upstream README still shows the March 20 winner at `1.1428`, so the root port is intentionally targeting the ahead-of-README PR frontier.

## Active Root Mainline
- Frozen dense fallback: `snapshots/train_gpt_2026-03-23_root_pr332_b458k_b1024_warmup200_xlast2.py`
- Frozen dense baseline config: `configs/h100/root_snapshot_b1024_warmup200_xlast2.json`
- New primary root lane: `configs/h100/root_shared8_mlp1664_b1024_warmup200_xlast2.json`
- Eval-only follow-up: `configs/h100/root_shared8_mlp1664_b1024_warmup200_xlast2_eval4096.json`

## Main Prediction
- The dense local lane has saturated around the low `1.160x` sliding band on `8xA100`.
- The next meaningful gain is more likely to come from effective depth-per-byte than from more warmup, BigramHash, or schedule micro-tuning.
- The first shared lane should therefore keep 12 applications, store only 8 unique full blocks, and spend the saved bytes on `MLP_HIDDEN=1664`.

## Why This Delta
- Parameter sharing attacks the real bottleneck in this benchmark: stored bytes under a hard artifact cap.
- The shared lane keeps the successful dense recipe fixed:
  - `SHUFFLE_DATA=1`
  - `WARMUP_STEPS=200`
  - `BIGRAM_VOCAB_SIZE=1024`
  - `XSA_LAST_N=2`
  - `EVAL_STRIDE=64`
- The new root change is deliberately narrow:
  - `UNIQUE_BLOCKS=8` over 12 applications
  - `MLP_HIDDEN=1664`
  - no MoE, no activation change, no new quantization path

## Immediate Ladder
- Run 1: frozen dense snapshot on H100
- Run 2: shared-depth root lane with `UNIQUE_BLOCKS=8` and `MLP_HIDDEN=1664`
- Run 3: `EVAL_SEQ_LEN=4096 EVAL_BATCH_SEQS=16` only on the winner of Runs 1-2

## Ranking Policy
- Rank by lower sliding-window `val_bpb`
- Require exact roundtrip to stay in-family and non-regressive
- Require `bytes_total < 16,000,000`
- Break close ties by higher train tokens processed under the same 600-second budget
