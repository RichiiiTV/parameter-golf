# Top Candidates

## Current Public Best
- `Muon WD + 10 layer`: `mean_val_bpb=1.17475315`, `artifact_bytes=15,374,243`
- Record path: `records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`
- Interpretation: the active target is no longer sliding eval alone; it is the full top-1 stack.

## Active H100 Ladder
- `h100-root-top1-repro`
- `h100-root-top1-seq2048`
- `h100-root-top1-seq4096`
- `h100-preproj-rmsnorm`
- `h100-recurrent-shared-width`
- `h100-sparse-outlier-retention`

## Top 10 Low-Hanging Fruits
- Reproduce the official top-1 stack cleanly in the reset root trainer
- Keep sliding eval fixed at `stride=64`
- Keep `tok_emb.weight` in fp16 at export
- Keep 10 layers as the baseline public-best depth
- Sweep sequence length through `1024`, `2048`, and `4096`
- Keep the root trainer close to the upstream/public-record structure
- Compare candidates by post-export `val_bpb` first, not pre-export loss
- Preserve byte headroom for later export-side ideas
- Keep active workflow H100-only and manual-only
- Keep Triton blocked until a winning GREEN lane exists

## Top 5 Hopper-Specific Experiments
- `h100-root-top1-repro`
- `h100-root-top1-seq2048`
- `h100-root-top1-seq4096`
- winning GREEN lane plus CUDA Graph capture
- winning GREEN lane plus modest batch-token sweep

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
- Run the clean H100 top-1 reproduction
- Run the seq2048 and seq4096 throughput branches
- Promote the best GREEN lane before implementing any YELLOW branch

## Ranking Policy
- Rank by lower post-export `val_bpb`
- Require `bytes_total < 16,000,000`
- Break close ties by higher train tokens processed under the same 600-second budget

## Manual Interface
- Use `py -3.11 scripts/prepare_h100_run.py <config>`
- Submit the emitted `sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="..."` command on the H100 cluster
