# PreProj RMSNorm + 10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04

Planned record fork built on `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`.

Status:
- H100 runs not executed yet from this folder.
- `submission.json` is intentionally provisional until real H100 runs exist.

## Goal

Keep the current March 20 top stack intact and test one bold paper-backed mechanism:
- add dedicated learnable RMSNorm layers immediately before the matrix-heavy projections
- leave the existing outer norms, SmearGate, BigramHash, SWA, mixed int5/int6 export, and sliding eval unchanged

## Exact Manual H100 Commands

Run these from this record folder.

### Stage 1: One-seed sanity

Baseline reproduction without the new mechanism:

```bash
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_parent USE_PREPROJ_RMSNORM=0 torchrun --standalone --nproc_per_node=8 train_gpt.py"
```

Contender with the new mechanism enabled:

```bash
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_on USE_PREPROJ_RMSNORM=1 torchrun --standalone --nproc_per_node=8 train_gpt.py"
```

### Stage 2: Throughput sweep on the better mechanism setting

Use the better of `USE_PREPROJ_RMSNORM=0` or `1`, then sweep:

```bash
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_b786k USE_PREPROJ_RMSNORM=1 TRAIN_BATCH_TOKENS=786432 torchrun --standalone --nproc_per_node=8 train_gpt.py"
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_b917k USE_PREPROJ_RMSNORM=1 TRAIN_BATCH_TOKENS=917504 torchrun --standalone --nproc_per_node=8 train_gpt.py"
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_b1048k USE_PREPROJ_RMSNORM=1 TRAIN_BATCH_TOKENS=1048576 torchrun --standalone --nproc_per_node=8 train_gpt.py"
```

### Stage 3: Three-seed promotion

Use the winning Stage 2 setting and run:

```bash
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_seed42 SEED=42 USE_PREPROJ_RMSNORM=1 TRAIN_BATCH_TOKENS=786432 torchrun --standalone --nproc_per_node=8 train_gpt.py"
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_seed1337 SEED=1337 USE_PREPROJ_RMSNORM=1 TRAIN_BATCH_TOKENS=786432 torchrun --standalone --nproc_per_node=8 train_gpt.py"
sbatch -c 6 --mem=20G --gres=gpu:8 -p batch_gpu -q 3h --wrap="RUN_ID=preproj_rmsnorm_seed2024 SEED=2024 USE_PREPROJ_RMSNORM=1 TRAIN_BATCH_TOKENS=786432 torchrun --standalone --nproc_per_node=8 train_gpt.py"
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

## New Mechanism

`USE_PREPROJ_RMSNORM=1` inserts learnable RMSNorm immediately before:
- `attn.c_q`
- `attn.c_k`
- `attn.c_v`
- `attn.proj`
- `mlp.fc`
- `mlp.proj`
- `bigram.proj` when `bigram_dim != model_dim`

The new RMSNorm scales initialize to `1.0`.

## Small Throughput Cleanup

- `VAL_LOSS_EVERY=0` by default
- `TRAIN_LOG_EVERY=200` by default

The intent is to remove mid-train validation overhead without changing the actual end-of-run scoring path.
