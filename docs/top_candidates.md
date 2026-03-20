# Top Candidates

## Current Public Best
- `10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04`: `mean_val_bpb=1.14276`, `artifact_bytes≈15.9M`
- Record path: `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
- Runner-up: `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` at `1.1458`
- Interpretation: the active frontier is now mixed int5/int6 export plus SmearGate/BigramHash/SWA, not the older March 19 top-1 stack.

## Active H100 Ladder
- `2026-03-20_PreProjRMSNorm_Int5MLP_Bigram10240_SWA` with `USE_PREPROJ_RMSNORM=0`
- `2026-03-20_PreProjRMSNorm_Int5MLP_Bigram10240_SWA` with `USE_PREPROJ_RMSNORM=1`
- batch sweep on the winner at `786432`, `917504`, `1048576`
- three-seed rerun on the winning setting
- only then consider recurrent/shared-width or sparse outlier retention

## Top 10 Low-Hanging Fruits
- Fork the March 20 top record instead of extending root `train_gpt.py`
- Keep mixed int5/int6 export exactly as the parent record does it
- Keep BigramHash at `10240`
- Keep SmearGate and SWA unchanged on the first contender
- Add only pre-projection RMSNorm as the first new mechanism
- Keep sliding eval fixed at `stride=64`
- Keep `VAL_LOSS_EVERY=0` by default in the contender
- Sweep `TRAIN_BATCH_TOKENS` after mechanism selection
- Compare candidates by post-export `val_bpb` first, not pre-export loss
- Keep Triton blocked until a winning post-March-20 lane exists

## Top 5 Hopper-Specific Experiments
- March 20 parent reproduction
- pre-proj RMSNorm enabled on the same stack
- `TRAIN_BATCH_TOKENS=786432`
- `TRAIN_BATCH_TOKENS=917504`
- `TRAIN_BATCH_TOKENS=1048576`

## Top 5 Triton Opportunities Or Reasons To Avoid Triton
- Avoid custom attention kernels; SDPA is already vendor-tuned
- Avoid custom GEMM; cuBLAS is not the bottleneck to out-engineer first
- Prefer standard PyTorch/cuDNN/Inductor until a real H100 hotspot is measured
- Only revisit export-side Triton if sparse outlier retention survives paper review and profiling
- Keep Triton out of the root mainline

## Top 5 Risky But Interesting Challenge-Edge Ideas
- pre-projection RMSNorm as a mechanism change
- recurrent/shared-width transformers
- sparse or decoupled high-precision outlier preservation
- tokenizer/vocabulary redesign
- eval-time auxiliary state or adaptation

## Top 3 Immediate Next Steps
- Run the new record fork with `USE_PREPROJ_RMSNORM=0`
- Run the same fork with `USE_PREPROJ_RMSNORM=1`
- Sweep batch tokens on whichever of those two wins

## Ranking Policy
- Rank by lower post-export `val_bpb`
- Require `bytes_total < 16,000,000`
- Break close ties by higher train tokens processed under the same 600-second budget

## Manual Interface
- The active SOTA path is self-contained inside `records/track_10min_16mb/2026-03-20_PreProjRMSNorm_Int5MLP_Bigram10240_SWA`
- Submit the exact `sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="..."` commands listed in that folder’s `README.md`
