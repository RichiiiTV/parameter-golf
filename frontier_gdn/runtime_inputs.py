from __future__ import annotations

import glob
import re
from pathlib import Path

CUSTOM_REPO_EXAMPLE = "kevclark/parameter-golf"


def infer_vocab_family_from_path(path_str: str) -> int | None:
    path = path_str.replace("\\", "/")
    patterns = (r"sp(\d+)", r"fineweb_(\d+)_bpe\.model", r"fineweb_(\d+)_bpe\.vocab")
    for pattern in patterns:
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
    return None


def ensure_local_runtime_inputs(args) -> None:
    data_vocab = infer_vocab_family_from_path(args.data_path)
    tokenizer_vocab = infer_vocab_family_from_path(args.tokenizer_path)
    mismatch_reasons: list[str] = []
    if data_vocab is not None and data_vocab != args.vocab_size:
        mismatch_reasons.append(f"DATA_PATH implies sp{data_vocab} but VOCAB_SIZE={args.vocab_size}")
    if tokenizer_vocab is not None and tokenizer_vocab != args.vocab_size:
        mismatch_reasons.append(f"TOKENIZER_PATH implies sp{tokenizer_vocab} but VOCAB_SIZE={args.vocab_size}")
    if data_vocab is not None and tokenizer_vocab is not None and data_vocab != tokenizer_vocab:
        mismatch_reasons.append(f"DATA_PATH implies sp{data_vocab} but TOKENIZER_PATH implies sp{tokenizer_vocab}")
    if mismatch_reasons:
        raise ValueError(
            "Inconsistent tokenizer family configuration: "
            + "; ".join(mismatch_reasons)
            + ". Use matching DATA_PATH, TOKENIZER_PATH, and VOCAB_SIZE values."
        )

    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Missing tokenizer model: {args.tokenizer_path}. "
            f"Download the active SP8192 cache with `MATCHED_FINEWEB_REPO_ID={CUSTOM_REPO_EXAMPLE} "
            "python data/cached_challenge_fineweb.py --variant sp8192` or point TOKENIZER_PATH at an existing SentencePiece model."
        )
    if not glob.glob(args.train_files):
        raise FileNotFoundError(
            f"No training shards found for pattern: {args.train_files}. "
            f"Download the active SP8192 cache with `MATCHED_FINEWEB_REPO_ID={CUSTOM_REPO_EXAMPLE} "
            "python data/cached_challenge_fineweb.py --variant sp8192` or point DATA_PATH at the correct dataset root."
        )
    if not glob.glob(args.val_files):
        raise FileNotFoundError(
            f"No validation shards found for pattern: {args.val_files}. "
            f"Download the active SP8192 cache with `MATCHED_FINEWEB_REPO_ID={CUSTOM_REPO_EXAMPLE} "
            "python data/cached_challenge_fineweb.py --variant sp8192` or point DATA_PATH at the correct dataset root."
        )
