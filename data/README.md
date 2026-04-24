# Data Workflows

The active legal baseline in this repo is SP8192 and expects a custom matched FineWeb export.

Canonical active layout:
- `data/datasets/fineweb10B_sp8192/`
- `data/tokenizers/fineweb_8192_bpe.model`
- `data/tokenizers/fineweb_8192_bpe.vocab`
- `data/manifest.json`

Default custom export route used by the active configs:
- `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf`
- `MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets`

The public Hugging Face dataset repo still only publishes SP1024. Use that only for the non-frontier MLX helper or historical debugging.

## Downloading The Active SP8192 Cache

Download the active SP8192 cache with:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets \
python3 data/cached_challenge_fineweb.py --variant sp8192
```

This populates `./data/datasets/fineweb10B_sp8192/` and `./data/tokenizers/`.
By default it downloads the full validation split and 8B training tokens (80 train shards).

Before launching `torchrun`, verify that the active config points at the downloaded tokenizer family and custom repo:

```bash
python3 scripts/check_run_ready.py configs/h100/root_sp8192_pr1493_accepted_8xh100.json
```

For a smaller bootstrap subset while iterating:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 1
```

Validation is always downloaded in full from the fixed `fineweb_val_*` split. Training on the first `N` train shards keeps the same frozen shuffled prefix for that tokenizer family.

## Public SP1024 Fallback

If you explicitly want the archived public cache for `train_gpt_mlx.py` or historical comparison:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

That uses the public default repo `willdepueoai/parameter-golf`. Do not treat it as the active SP8192 surface for `train_gpt.py`.
