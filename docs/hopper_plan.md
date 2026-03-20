# Hopper Plan

Current public best to beat:
- `Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init` at `mean_val_bpb=1.17475315`

## 4xA100 Promotion Gate
- Lane A single-point proxy baseline: `configs/a100/seq2048_fp16embed_slide64_control.json`
- Lane B single-point proxy baseline: `configs/a100/10l_muonwd_overtone_slide64_primary.json`
- Lane A proxy sweep: `configs/a100/lane_a_proxy_matrix.json`
- Lane B proxy sweep: `configs/a100/lane_b_proxy_matrix.json`
- Lane A full rerun surface: `configs/a100/lane_a_full_matrix.json`
- Lane B full rerun surface: `configs/a100/lane_b_full_matrix.json`
- Rule: run the four `TRAIN_BATCH_TOKENS=524288` proxy points first, keep only the better compile mode per lane, then sweep higher batch sizes, then rerun one full sliding finalist per lane.

RUN THIS MANUALLY ON 4xA100
- command: `py -3.11 scripts/prepare_a100_run.py configs/a100/lane_a_proxy_matrix.json`
- command: `py -3.11 scripts/prepare_a100_run.py configs/a100/lane_b_proxy_matrix.json`
- command: `py -3.11 scripts/prepare_a100_run.py configs/a100/lane_a_full_matrix.json`
- command: `py -3.11 scripts/prepare_a100_run.py configs/a100/lane_b_full_matrix.json`
- expected outputs: printed 4xA100 command blocks for proxy and full sliding runs
- success criteria: identify one full-eval finalist per lane before any H100 run

## H100 Candidate Selection
- No H100 candidate is final until the A100 throughput pass finishes.
- If Lane A and Lane B are within `0.003 bpb` on full sliding eval, prefer the faster lane with higher `train_tokens_seen`.
- If Lane B wins clearly, use `configs/h100/10l_muonwd_overtone_slide64.json`.
- If Lane A wins or is within the tie band and faster, use `configs/h100/seq2048_fp16embed_slide64.json`.

## Provisional H100 Lane B Candidate
- Use only if Lane B wins the A100 throughput pass.

RUN THIS MANUALLY ON H100
- config: `configs/h100/10l_muonwd_overtone_slide64.json`
- environment:
  - `ADAMW_WEIGHT_DECAY=0.01`
  - `COMPILE_MODE=fullgraph`
  - `COMPUTE_DTYPE=auto`
  - `EVAL_BATCH_SEQS=1024`
  - `EVAL_MODE=sliding`
  - `EVAL_SEQ_LEN=1024`
  - `EVAL_STRIDE=64`
  - `KEEP_FLOAT_EXTRA=tok_emb.weight`
  - `MATRIX_LR=0.04`
  - `MAX_WALLCLOCK_SECONDS=600`
  - `MLP_HIDDEN=0`
  - `MUON_WEIGHT_DECAY=0.02`
  - `NUM_LAYERS=10`
  - `RESID_MIX_INIT=phase_transition`
  - `SCALAR_LR=0.04`
  - `SDPA_BACKEND=auto`
  - `TIED_EMBED_INIT_MODE=overtone`
  - `TIED_EMBED_LR=0.10`
  - `TRAIN_BATCH_TOKENS=524288`
  - `TRAIN_LOG_EVERY=50`
  - `TRAIN_SEQ_LEN=1024`
  - `VAL_LOSS_EVERY=1000`
  - `WARMDOWN_ITERS=2500`
- command: `torchrun --standalone --nproc_per_node=8 train_gpt.py`
- runtime budget: 10 minutes train, 10 minutes eval
- expected outputs: `logs/<RUN_ID>.txt`, `final_model.int8.ptz`, final roundtrip metrics
- risks: the root-trainer port may still underperform the record implementation, or the A100 win may not transfer cleanly to H100
- success criteria: run this only if Lane B wins the A100 throughput pass, then target the current public bar near `1.1748` without exceeding `16,000,000` bytes

## Provisional H100 Lane A Candidate
- Use only if Lane A wins the A100 throughput pass, or if Lane B is within the tie band but slower.

RUN THIS MANUALLY ON H100
- config: `configs/h100/seq2048_fp16embed_slide64.json`
- environment:
  - `COMPILE_MODE=fullgraph`
  - `COMPUTE_DTYPE=auto`
  - `EVAL_BATCH_SEQS=1024`
  - `EVAL_MODE=sliding`
  - `EVAL_SEQ_LEN=2048`
  - `EVAL_STRIDE=64`
  - `KEEP_FLOAT_EXTRA=tok_emb.weight`
  - `MATRIX_LR=0.032`
  - `MAX_WALLCLOCK_SECONDS=600`
  - `MLP_HIDDEN=992`
  - `SCALAR_LR=0.032`
  - `SDPA_BACKEND=auto`
  - `TIED_EMBED_LR=0.04`
  - `TRAIN_BATCH_TOKENS=524288`
  - `TRAIN_LOG_EVERY=50`
  - `TRAIN_SEQ_LEN=2048`
  - `VAL_LOSS_EVERY=1000`
  - `WARMDOWN_ITERS=3600`
- command: `torchrun --standalone --nproc_per_node=8 train_gpt.py`
- runtime budget: 10 minutes train, 10 minutes eval
- expected outputs: `logs/<RUN_ID>.txt`, `final_model.int8.ptz`, final roundtrip metrics
- risks: Lane A may simply have a lower ceiling even if it is faster
- success criteria: run this only if Lane A wins the A100 throughput pass, or as the chosen winner under the tie-band rule

## Systems Follow-Up
- Benchmark `SDPA_BACKEND x COMPILE_MODE` only after the winning lane is identified.
- First emit the per-point manual handoff blocks locally, then execute the printed `torchrun` blocks one by one on H100.

RUN THIS MANUALLY ON H100
- config: `configs/h100/systems_sdpa_compile_matrix.json`
- environment: use the matrix points encoded in the config plus the winning lane env
- command: `py -3.11 scripts/prepare_h100_run.py configs/h100/systems_sdpa_compile_matrix.json`
- runtime budget: 10 minutes per matrix point
- expected outputs: emitted manual blocks for each matrix point, then per-point throughput and final roundtrip metrics once those blocks are executed by a human
- risks: compile overhead hiding steady-state gains, backend-specific regressions
- success criteria: identify the best backend/compile combination for the winning lane
