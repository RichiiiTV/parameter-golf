# Hopper Plan

Current public best to beat:
- `SlidingWindowEval` at `post-export val_bpb=1.19250007`

## Primary Challenger
- Strict composition of accepted record levers only:
  - `LongContextSeq2048` training
  - `FP16Embed_WD3600` export handling
  - `SlidingWindowEval` scoring
- Keep the baseline architecture unchanged.

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
- risks: compile overhead, eval-time budget, artifact-byte regressions
- success criteria: beat `SlidingWindowEval` at `post-export val_bpb=1.19250007` without exceeding `16,000,000` bytes

## Follow-Up 1
- Same challenger plus EMA endpoint smoothing
- Run only after the primary challenger has a truth result.

RUN THIS MANUALLY ON H100
- config: `configs/h100/seq2048_fp16embed_slide64_ema.json`
- environment:
  - `COMPILE_MODE=fullgraph`
  - `COMPUTE_DTYPE=auto`
  - `EMA_DECAY=0.999`
  - `EMA_START_STEP=2000`
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
- risks: EMA endpoint lag, compile overhead, eval-time budget, artifact-byte regressions
- success criteria: reduce pre/post export gap without regressing final score

## Systems Follow-Up
- Benchmark `SDPA_BACKEND x COMPILE_MODE` only after the primary challenger has run.
- First emit the per-point manual handoff blocks locally, then execute the printed `torchrun` blocks one by one on H100.

RUN THIS MANUALLY ON H100
- config: `configs/h100/systems_sdpa_compile_matrix.json`
- environment: use the matrix points encoded in the config plus the challenger env
- command: `py -3.11 scripts/prepare_h100_run.py configs/h100/systems_sdpa_compile_matrix.json`
- runtime budget: 10 minutes per matrix point
- expected outputs: emitted manual blocks for each matrix point, then per-point throughput and final roundtrip metrics once those blocks are executed by a human
- risks: compile overhead hiding steady-state gains, backend-specific regressions
- success criteria: identify the best backend/compile combination that survives the final eval budget
