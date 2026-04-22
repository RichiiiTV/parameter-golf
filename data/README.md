# Data Workflows

This directory contains the active SP1024 dataset layout for the cleaned baseline repo.

Canonical local layout:
- `data/datasets/fineweb10B_sp1024/`
- `data/tokenizers/`
- `data/manifest.json`

## Downloading Published Data

Download the cached FineWeb SP1024 export with:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`.
By default it downloads the full validation split and 8B training tokens (80 train shards).

To fetch more training shards, pass `--train-shards`:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 180
```

The downloader is manifest-driven and can fetch only a prefix of train shards from a larger published export. With the current shard size of `100_000_000` tokens, `10B` retokenized training tokens is `100` train shards:

```bash
MATCHED_FINEWEB_REPO_ID=your-hf-username/your-dataset-repo \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=your_50B_export_root \
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 100
```

Validation is always downloaded in full from the fixed `fineweb_val_*` split. Training on the first `N` train shards means training on the prefix of the same frozen shuffled export, so the data order stays aligned with the baseline for that tokenizer family.

The default published repo is `willdepueoai/parameter-golf`, with the export rooted under the repo subdirectory `datasets/`.
