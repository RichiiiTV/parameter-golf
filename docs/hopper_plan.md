# Hopper Plan

Current public best to beat:
- `Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init` at `mean_val_bpb=1.17475315`

## 4xA100 Promotion Gate
- Control: `configs/a100/seq2048_fp16embed_slide64_control.json`
- Primary: `configs/a100/10l_muonwd_overtone_slide64_primary.json`
- Fallback: `configs/a100/10l_muonwd_lowlr_slide64_fallback.json`
- Rule: run the control first, then the primary. Only promote the 10-layer path to H100 if it wins clearly on post-export `val_bpb` without breaking bytes or eval policy.

## Primary H100 Candidate
- Root-trainer port of the accepted top-1 stack:
  - `NUM_LAYERS=10`
  - decoupled `MUON_WEIGHT_DECAY=0.02`
  - `ADAMW_WEIGHT_DECAY=0.01`
  - `TIED_EMBED_INIT_MODE=overtone`
  - `RESID_MIX_INIT=phase_transition`
  - clean sliding eval
  - fp16 `tok_emb.weight` export passthrough

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
- risks: root-trainer port may underperform the record implementation, aggressive LR may widen the pre/post-export gap
- success criteria: beat the legacy seq2048 control and close on or below `1.1748` post-export `val_bpb` without exceeding `16,000,000` bytes

## Fallback H100 Candidate
- Same 10-layer shape and decay setup, but lower LR and longer warmdown to protect quantization robustness.

RUN THIS MANUALLY ON H100
- config: `configs/h100/10l_muonwd_lowlr_slide64.json`
- environment:
  - `ADAMW_WEIGHT_DECAY=0.01`
  - `COMPILE_MODE=fullgraph`
  - `COMPUTE_DTYPE=auto`
  - `EVAL_BATCH_SEQS=1024`
  - `EVAL_MODE=sliding`
  - `EVAL_SEQ_LEN=1024`
  - `EVAL_STRIDE=64`
  - `KEEP_FLOAT_EXTRA=tok_emb.weight`
  - `MATRIX_LR=0.02`
  - `MAX_WALLCLOCK_SECONDS=600`
  - `MLP_HIDDEN=0`
  - `MUON_WEIGHT_DECAY=0.02`
  - `NUM_LAYERS=10`
  - `RESID_MIX_INIT=flat`
  - `SCALAR_LR=0.02`
  - `SDPA_BACKEND=auto`
  - `TIED_EMBED_INIT_MODE=normal`
  - `TIED_EMBED_LR=0.03`
  - `TRAIN_BATCH_TOKENS=524288`
  - `TRAIN_LOG_EVERY=50`
  - `TRAIN_SEQ_LEN=1024`
  - `VAL_LOSS_EVERY=1000`
  - `WARMDOWN_ITERS=20000`
- command: `torchrun --standalone --nproc_per_node=8 train_gpt.py`
- runtime budget: 10 minutes train, 10 minutes eval
- expected outputs: `logs/<RUN_ID>.txt`, `final_model.int8.ptz`, final roundtrip metrics
- risks: may give up too much raw learning speed to the primary recipe
- success criteria: beat the legacy seq2048 control while keeping a tighter pre/post-export gap than the primary if the primary is unstable

## Legacy H100 Control
- Keep the old seq2048 challenger for direct relative comparison. It is no longer the primary target.

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
- risks: no longer reflects the current public frontier
- success criteria: serve as the relative baseline for the new 10-layer path

## Systems Follow-Up
- Benchmark `SDPA_BACKEND x COMPILE_MODE` only after a 10-layer candidate has won.
- First emit the per-point manual handoff blocks locally, then execute the printed `torchrun` blocks one by one on H100.

RUN THIS MANUALLY ON H100
- config: `configs/h100/systems_sdpa_compile_matrix.json`
- environment: use the matrix points encoded in the config plus the primary 10-layer challenger env
- command: `py -3.11 scripts/prepare_h100_run.py configs/h100/systems_sdpa_compile_matrix.json`
- runtime budget: 10 minutes per matrix point
- expected outputs: emitted manual blocks for each matrix point, then per-point throughput and final roundtrip metrics once those blocks are executed by a human
- risks: compile overhead hiding steady-state gains, backend-specific regressions
- success criteria: identify the best backend/compile combination for the winning 10-layer candidate
