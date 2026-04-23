# Data Workflows

This repo is now targeted at the SP8192 frontier lane.

Canonical active layout:
- `data/datasets/fineweb10B_sp8192/`
- `data/tokenizers/fineweb_8192_bpe.model`
- `data/manifest.json`

Historical SP1024 assets can still exist locally for snapshots and old logs, but the active root/config surface expects SP8192.

## Downloading Published Data

Download the active SP8192 cache with:

```bash
python3 data/cached_challenge_fineweb.py --variant sp8192
```

This populates `./data/datasets/fineweb10B_sp8192/` and `./data/tokenizers/`.
By default it downloads the full validation split and 8B training tokens (80 train shards).

For a smaller local smoke subset:

```bash
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 1
```

The downloader is manifest-driven. It can still fetch older variants such as `sp1024`, but those are not the active record-intent path anymore.

## Alternate Published Roots

If you have your own published export root:

```bash
MATCHED_FINEWEB_REPO_ID=your-hf-username/your-dataset-repo \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=your_export_root \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 100
```

Validation is always downloaded in full from the fixed `fineweb_val_*` split. Training on the first `N` train shards keeps the same frozen shuffled prefix for that tokenizer family.
