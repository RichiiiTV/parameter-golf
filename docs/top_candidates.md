# Top Candidates

## Current Public Best
- `Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init`: `mean_val_bpb=1.17475315`, `artifact_bytes=15,374,243`
- Record path: `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`
- Interpretation: the public frontier now combines clean sliding eval, fp16 tied-embedding export, 10 layers, decoupled Muon weight decay, overtone tied-embedding init, and phase-transition `resid_mix` init

## Root Candidate Ladder
- Legacy control: `seq2048 + fp16 tok_emb export + sliding eval stride 64`
- Primary root challenger: `10L + Muon WD + overtone init + phase-transition resid_mix + sliding eval stride 64 + fp16 tok_emb export`
- Fallback root challenger: `10L + Muon WD + low LR + long warmdown + sliding eval stride 64 + fp16 tok_emb export`

## Primary Root Challenger
- `NUM_LAYERS=10`
- `TRAIN_SEQ_LEN=1024`
- `TIED_EMBED_LR=0.10`
- `MATRIX_LR=0.04`
- `SCALAR_LR=0.04`
- `WARMDOWN_ITERS=2500`
- `MUON_WEIGHT_DECAY=0.02`
- `ADAMW_WEIGHT_DECAY=0.01`
- `TIED_EMBED_INIT_MODE=overtone`
- `RESID_MIX_INIT=phase_transition`
- `KEEP_FLOAT_EXTRA=tok_emb.weight`
- `EVAL_MODE=sliding`
- `EVAL_SEQ_LEN=1024`
- `EVAL_STRIDE=64`
- `MLP_HIDDEN=0`

## 4xA100 Preflight Order
- Control: `configs/a100/seq2048_fp16embed_slide64_control.json`
- Primary: `configs/a100/10l_muonwd_overtone_slide64_primary.json`
- Fallback: `configs/a100/10l_muonwd_lowlr_slide64_fallback.json`
- Promotion rule: only send the 10-layer path to H100 if it clearly beats the control on post-export `val_bpb` at the same 600-second budget

## Top 10 Low-Hanging Fruits
- Port the top-1 10-layer stack into the root trainer without adding challenge-edge features
- Keep `tok_emb.weight` in fp16 at export
- Use clean sliding eval with `stride=64`
- Add decoupled Muon weight decay only to matrix params
- Add AdamW decay only to token/scalar groups
- Use overtone tied-embedding init in the tied-weight path
- Use phase-transition `resid_mix` init across blocks
- Keep the low-LR, long-warmdown 10-layer fallback ready if the primary overfits or quantizes poorly
- Benchmark `SDPA_BACKEND x COMPILE_MODE` only after the winning 10-layer candidate is identified
- Keep Triton out until a standard-backend loss is measured on H100

## Top 5 Hopper-Specific Experiments
- Primary H100 run: `configs/h100/10l_muonwd_overtone_slide64.json`
- Fallback H100 run: `configs/h100/10l_muonwd_lowlr_slide64.json`
- Legacy control H100 run: `configs/h100/seq2048_fp16embed_slide64.json`
- Post-winner `SDPA_BACKEND x COMPILE_MODE` matrix
- EMA only after the 10-layer path has a truth result

## Top 5 Triton Opportunities Or Reasons To Avoid Triton
- Avoid custom attention kernels
- Avoid custom GEMM
- Prefer standard SDPA/cuDNN/Inductor first
- Consider export packing only after profiling
- Keep Triton blocked until the 10-layer path has H100 truth data

## Top 5 Risky But Interesting Challenge-Edge Ideas
- LoRA TTT
- Mixed int8/int6 export
- Eval-time adaptation
- Recurrent/shared blocks
- Tokenizer changes

## Top 3 Immediate Next Steps
- Run the 4xA100 control and primary challenger back-to-back
- Run the 4xA100 fallback only if the primary recipe is unstable or loses too much post-export quality
- Promote only the winning 10-layer path to H100 manual truth runs

## Local Proxy Notes
- `gtx1080ti-smoke`: `pre/post val_bpb = 3.94218527 / 3.95601011`, `bytes_total=5,050,048`
- `gtx1080ti-proxy1024`: `pre/post val_bpb = 4.01842393 / 4.03490743`, `bytes_total=5,049,873`
- `gtx1080ti-export-proxy`: `pre/post val_bpb = 4.01853171 / 4.01945346`, `bytes_total=5,818,837`
- `gtx1080ti-10l-feature-smoke`: `pre/post val_bpb = 4.03572367 / 4.03573868`, `bytes_total=6,417,208`, `eval_time_ms=2740`
- Local conclusion: `KEEP_FLOAT_EXTRA=tok_emb.weight` is a strong export-gap lever on GTX, shrinking the post-export gap by about `18x`, but it costs about `0.77 MB` and still needs A100/H100 truth runs before promotion
