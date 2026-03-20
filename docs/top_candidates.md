# Top Candidates

## Current Public Best
- `Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init`: `mean_val_bpb=1.17475315`, `artifact_bytes=15,374,243`
- Record path: `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`
- Interpretation: the public frontier now combines clean sliding eval, fp16 tied-embedding export, 10 layers, decoupled Muon weight decay, overtone tied-embedding init, and phase-transition `resid_mix` init

## Root Candidate Ladder
- Lane A control: `seq2048 + fp16 tok_emb export + sliding eval stride 64`
- Lane B primary: `10L + Muon WD + overtone init + phase-transition resid_mix + sliding eval stride 64 + fp16 tok_emb export`
- Throughput-first rule: do not add fallback model branches, EMA, seq4096, or export codec changes until Lane A vs Lane B is decided on A100

## Throughput Pass
- `Lane A`: keep the legacy seq2048 control fixed except for `COMPILE_MODE`, `TRAIN_BATCH_TOKENS`, and proxy vs full eval scope
- `Lane B`: keep the current 10-layer root port fixed except for `COMPILE_MODE`, `TRAIN_BATCH_TOKENS`, and proxy vs full eval scope
- Rank proxy runs by:
  1. lower `post-export val_bpb`
  2. higher `train_tokens_seen`
  3. lower `pre_post_gap`

## 4xA100 Preflight Order
- First pass:
  - `configs/a100/seq2048_fp16embed_slide64_control.json`
  - `configs/a100/10l_muonwd_overtone_slide64_primary.json`
  - `configs/a100/lane_a_proxy_matrix.json`
  - `configs/a100/lane_b_proxy_matrix.json`
- Finalist pass:
  - `configs/a100/lane_a_full_matrix.json`
  - `configs/a100/lane_b_full_matrix.json`
- Promotion rule: only send the winning lane to H100 after one full sliding rerun per lane finalist

## Top 10 Low-Hanging Fruits
- Use proxy-only end-of-run eval on A100 preflights
- Keep `VAL_LOSS_EVERY=0` on A100 preflights
- Sweep `COMPILE_MODE=off|fullgraph` only
- Sweep `TRAIN_BATCH_TOKENS=524288|786432|1048576`
- Keep `SDPA_BACKEND=auto` fixed on A100
- Keep `tok_emb.weight` in fp16 at export
- Keep sliding eval fixed at `stride=64`
- Log `script_path`, `trainer_sha256`, `val_scope`, `val_max_seqs`, `train_tokens_seen`, and `ms_per_step`
- Use full sliding reruns only for lane finalists
- Keep Triton and new model ideas out until the throughput pass finishes

## Top 5 Hopper-Specific Experiments
- Winning Lane A or Lane B config promoted from full A100 eval
- Legacy seq2048 control on H100 only if Lane A wins
- 10-layer root port on H100 only if Lane B wins
- Post-winner `SDPA_BACKEND x COMPILE_MODE` matrix
- No EMA or extra model branches until a winning lane is identified

## Top 5 Triton Opportunities Or Reasons To Avoid Triton
- Avoid custom attention kernels
- Avoid custom GEMM
- Prefer standard SDPA/cuDNN/Inductor first
- Consider export packing only after profiling
- Keep Triton blocked until the throughput pass identifies a winning lane

## Top 5 Risky But Interesting Challenge-Edge Ideas
- LoRA TTT
- Mixed int8/int6 export
- Eval-time adaptation
- Recurrent/shared blocks
- Tokenizer changes

## Top 3 Immediate Next Steps
- Run the four A100 proxy baselines first at `TRAIN_BATCH_TOKENS=524288`
- Sweep higher batch sizes only on the better compile mode within each lane
- Rerun one full sliding finalist per lane before deciding the H100 candidate

## Latest A100 Findings
- Lane A control on 4xA100: `post-export val_bpb=1.34368971`, `train_tokens_seen=644,874,240`, `step_stop=1230`
- Lane B primary on 4xA100: `post-export val_bpb=1.34109285`, `train_tokens_seen=540,540,928`, `step_stop=1031`
- Interpretation: Lane B only barely beat Lane A while processing materially fewer tokens, so throughput recovery is the right next move

## Local Proxy Notes
- `gtx1080ti-smoke`: `pre/post val_bpb = 3.94218527 / 3.95601011`, `bytes_total=5,050,048`
- `gtx1080ti-proxy1024`: `pre/post val_bpb = 4.01842393 / 4.03490743`, `bytes_total=5,049,873`
- `gtx1080ti-export-proxy`: `pre/post val_bpb = 4.01853171 / 4.01945346`, `bytes_total=5,818,837`
- `gtx1080ti-10l-feature-smoke`: `pre/post val_bpb = 4.03572367 / 4.03573868`, `bytes_total=6,417,208`, `eval_time_ms=2740`
- Local conclusion: `KEEP_FLOAT_EXTRA=tok_emb.weight` is a strong export-gap lever on GTX, shrinking the post-export gap by about `18x`, but it costs about `0.77 MB` and still needs A100/H100 truth runs before promotion
