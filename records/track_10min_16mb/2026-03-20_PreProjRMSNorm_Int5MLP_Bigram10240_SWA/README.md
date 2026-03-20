# PreProj RMSNorm Ablation On 10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04

Record fork built on `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`.

Status:
- 4xA100 same-folder ablation: `USE_PREPROJ_RMSNORM=0 -> 1.28120795`, `USE_PREPROJ_RMSNORM=1 -> 1.30713253`
- Decision: pre-projection RMSNorm underperformed by `+0.02592458 bpb`; the active path from this folder is now `USE_PREPROJ_RMSNORM=0`
- H100 runs not executed yet from this folder.
- `submission.json` is intentionally provisional until real H100 runs exist.

## Goal

Keep the current March 20 top stack intact and push throughput first:
- keep the parent stack unchanged by default
- use this folder for parent-stack reproduction and batch-token sweeps
- keep pre-projection RMSNorm as an explicit deferred ablation, not the mainline

## Exact Manual H100 Commands

Run these from this record folder.

### Stage 1: One-seed sanity

Baseline reproduction on the active path:

```bash
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_parent USE_PREPROJ_RMSNORM=0 torchrun --standalone --nproc_per_node=8 train_gpt.py"
```

Deferred mechanism ablation only if you explicitly want to revisit it:

```bash
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_on USE_PREPROJ_RMSNORM=1 torchrun --standalone --nproc_per_node=8 train_gpt.py"
```

### Stage 2: Throughput sweep on the active path

Sweep `TRAIN_BATCH_TOKENS` on `USE_PREPROJ_RMSNORM=0`:

```bash
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_b786k USE_PREPROJ_RMSNORM=0 TRAIN_BATCH_TOKENS=786432 torchrun --standalone --nproc_per_node=8 train_gpt.py"
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_b917k USE_PREPROJ_RMSNORM=0 TRAIN_BATCH_TOKENS=917504 torchrun --standalone --nproc_per_node=8 train_gpt.py"
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_b1048k USE_PREPROJ_RMSNORM=0 TRAIN_BATCH_TOKENS=1048576 torchrun --standalone --nproc_per_node=8 train_gpt.py"
```

### Stage 3: Three-seed promotion

Use the winning Stage 2 setting on `USE_PREPROJ_RMSNORM=0` and run:

```bash
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_seed42 SEED=42 USE_PREPROJ_RMSNORM=0 TRAIN_BATCH_TOKENS=786432 torchrun --standalone --nproc_per_node=8 train_gpt.py"
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_seed1337 SEED=1337 USE_PREPROJ_RMSNORM=0 TRAIN_BATCH_TOKENS=786432 torchrun --standalone --nproc_per_node=8 train_gpt.py"
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_seed2024 SEED=2024 USE_PREPROJ_RMSNORM=0 TRAIN_BATCH_TOKENS=786432 torchrun --standalone --nproc_per_node=8 train_gpt.py"
```

Update the commands above if a higher batch token setting wins Stage 2.

## Stack Kept From The Parent Record

- mixed int5/int6 quantization with `zstd-22` fallback to `zlib`
- BigramHash `10240`
- SmearGate
- 10 layers, 512 dim, 8 heads, 4 KV heads
- MLP 3x expansion
- Muon/AdamW weight decay at `0.04`
- SWA with `start_frac=0.4`, `every=50`
- seq2048 and sliding eval `stride=64`

## Deferred Mechanism

`USE_PREPROJ_RMSNORM=1` inserts learnable RMSNorm immediately before:
- `attn.c_q`
- `attn.c_k`
- `attn.c_v`
- `attn.proj`
- `mlp.fc`
- `mlp.proj`
- `bigram.proj` when `bigram_dim != model_dim`

The new RMSNorm scales initialize to `1.0`.

Current status:
- same-folder 4xA100 ablation was negative
- do not spend H100 on this mechanism until the parent-stack throughput lane is settled

## Small Throughput Cleanup

- `VAL_LOSS_EVERY=0` by default
- `TRAIN_LOG_EVERY=200` by default

The intent is to remove mid-train validation overhead without changing the actual end-of-run scoring path.
