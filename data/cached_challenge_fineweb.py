from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DEFAULT_PUBLIC_REPO_ID = "willdepueoai/parameter-golf"
ACTIVE_FRONTIER_REPO_ID = "kevclark/parameter-golf"
REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", DEFAULT_PUBLIC_REPO_ID)
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
TOKENIZERS_DIR = ROOT / "tokenizers"
BUILTIN_TOKENIZERS = {
    "sp_bpe_1024": {
        "name": "sp_bpe_1024",
        "kind": "sentencepiece_bpe",
        "vocab_size": 1024,
        "bos_id": 1,
        "eos_id": 2,
        "model_path": "tokenizers/fineweb_1024_bpe.model",
        "vocab_path": "tokenizers/fineweb_1024_bpe.vocab",
    },
    "sp_bpe_8192": {
        "name": "sp_bpe_8192",
        "kind": "sentencepiece_bpe",
        "vocab_size": 8192,
        "bos_id": 1,
        "eos_id": 2,
        "model_path": "tokenizers/fineweb_8192_bpe.model",
        "vocab_path": "tokenizers/fineweb_8192_bpe.vocab",
    },
}
BUILTIN_DATASETS = {
    "fineweb10B_sp1024": {
        "name": "fineweb10B_sp1024",
        "tokenizer_name": "sp_bpe_1024",
        "tokenizer_kind": "sentencepiece_bpe",
        "path": "datasets/fineweb10B_sp1024",
        "train_glob": "datasets/fineweb10B_sp1024/fineweb_train_*.bin",
        "val_glob": "datasets/fineweb10B_sp1024/fineweb_val_*.bin",
        "vocab_size": 1024,
        "bos_id": 1,
        "eos_id": 2,
        "stats": {"files_train": 195, "files_val": 1},
    },
    "fineweb10B_sp8192": {
        "name": "fineweb10B_sp8192",
        "tokenizer_name": "sp_bpe_8192",
        "tokenizer_kind": "sentencepiece_bpe",
        "path": "datasets/fineweb10B_sp8192",
        "train_glob": "datasets/fineweb10B_sp8192/fineweb_train_*.bin",
        "val_glob": "datasets/fineweb10B_sp8192/fineweb_val_*.bin",
        "vocab_size": 8192,
        "bos_id": 1,
        "eos_id": 2,
        "stats": {"files_train": 195, "files_val": 1},
    },
}

def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


def local_path_for_remote(relative_path: str) -> Path:
    remote_path = Path(relative_path)
    if REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (REMOTE_ROOT_PREFIX,):
        remote_path = remote_path.relative_to(REMOTE_ROOT_PREFIX)
    if remote_path.parts[:1] == ("datasets",):
        return DATASETS_DIR.joinpath(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return TOKENIZERS_DIR.joinpath(*remote_path.parts[1:])
    return ROOT / remote_path
def build_remote_candidates(relative_path: str) -> list[Path]:
    raw = Path(relative_path)
    candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in (raw, raw.relative_to(REMOTE_ROOT_PREFIX) if REMOTE_ROOT_PREFIX and raw.parts[:1] == (REMOTE_ROOT_PREFIX,) else None):
        if candidate is None:
            continue
        key = candidate.as_posix()
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    return candidates
def is_remote_not_found(exc: Exception) -> bool:
    message = str(exc)
    name = exc.__class__.__name__
    return "404" in message or "Not Found" in message or "RemoteEntryNotFound" in name


def get(relative_path: str) -> None:
    destination = local_path_for_remote(relative_path)
    if destination.exists():
        return
    if destination.is_symlink():
        destination.unlink()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download missing dataset/tokenizer files. "
            "Install it with `pip install huggingface-hub` or use the prepared H100 image."
        ) from exc
    last_exc: Exception | None = None
    cached_source: Path | None = None
    for remote_path in build_remote_candidates(relative_path):
        try:
            cached_path = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=remote_path.name,
                    subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
                    repo_type="dataset",
                )
            )
            cached_source = cached_path.resolve(strict=True)
            break
        except Exception as exc:
            last_exc = exc
            if not is_remote_not_found(exc):
                raise
    if cached_source is None:
        attempted = ", ".join(path.as_posix() for path in build_remote_candidates(relative_path))
        raise RuntimeError(f"Unable to download {relative_path}. Attempted remote paths: {attempted}") from last_exc
    # HF cache entries may be snapshot symlinks. Resolve to the underlying blob so we
    # always materialize a real file in data/, not a broken relative symlink.
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached_source, destination)
    except OSError:
        shutil.copy2(cached_source, destination)


def manifest_path() -> Path:
    return local_path_for_remote(f"{REMOTE_ROOT_PREFIX}/manifest.json")


def load_manifest(*, skip_manifest_download: bool) -> dict:
    path = manifest_path()
    if not path.is_file():
        if skip_manifest_download:
            return {}
        get(f"{REMOTE_ROOT_PREFIX}/manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))
def resolve_dataset_entry(manifest: dict, dataset_name: str) -> dict | None:
    entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_name), None)
    if entry is not None:
        return entry
    return dict(BUILTIN_DATASETS.get(dataset_name, {})) or None
def resolve_tokenizer_entry(manifest: dict, tokenizer_name: str) -> dict | None:
    entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if entry is not None:
        return entry
    return dict(BUILTIN_TOKENIZERS.get(tokenizer_name, {})) or None


def artifact_paths_for_tokenizer(tokenizer_entry: dict) -> list[str]:
    artifacts = []
    for key in ("model_path", "vocab_path", "path"):
        value = tokenizer_entry.get(key)
        if value:
            artifacts.append(str(value))
    if not artifacts:
        raise ValueError(f"tokenizer entry is missing downloadable artifacts: {tokenizer_entry}")
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download challenge FineWeb shards from Hugging Face")
    parser.add_argument(
        "train_shards_positional",
        nargs="?",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=80,
        help="Number of training shards to download for the selected variant. Defaults to 80.",
    )
    parser.add_argument(
        "--variant",
        default="sp8192",
        help="Tokenizer family to download, for example sp1024, sp8192, or byte260.",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip downloading manifest.json.",
    )
    parser.add_argument(
        "--with-docs",
        action="store_true",
        help="Also download docs_selected.jsonl and its sidecar for tokenizer retraining or dataset re-export.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = dataset_dir_for_variant(args.variant)
    train_shards = args.train_shards_positional if args.train_shards_positional is not None else args.train_shards
    if train_shards < 0:
        raise ValueError("train_shards must be non-negative")
    if REPO_ID == DEFAULT_PUBLIC_REPO_ID and dataset_dir != "fineweb10B_sp1024":
        raise ValueError(
            f"{args.variant} is not published in the public HF repo {REPO_ID}. "
            "Use `--variant sp1024` for the public cache, or set "
            f"`MATCHED_FINEWEB_REPO_ID={ACTIVE_FRONTIER_REPO_ID}` and "
            "`MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets` for the active SP8192 frontier export."
        )

    manifest = load_manifest(skip_manifest_download=args.skip_manifest)
    manifest_dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir), None)
    if manifest_dataset_entry is None and REPO_ID == DEFAULT_PUBLIC_REPO_ID:
        published = ", ".join(sorted(x.get("name", "<unknown>") for x in manifest.get("datasets", [])))
        raise ValueError(
            f"{args.variant} is not published in the public HF repo {REPO_ID}. "
            f"Published dataset variants there are: {published or 'none'}. "
            "Use `--variant sp1024` for the public cache, or set "
            f"`MATCHED_FINEWEB_REPO_ID={ACTIVE_FRONTIER_REPO_ID}` and "
            "`MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets` for the active SP8192 frontier export."
        )
    dataset_entry = resolve_dataset_entry(manifest, dataset_dir)
    if dataset_entry is None:
        raise ValueError(
            f"dataset {dataset_dir} not found in {REMOTE_ROOT_PREFIX}/manifest.json or built-in fallback metadata"
        )
    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train", 0))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val", 0))
    if max_train_shards <= 0 or val_shards <= 0:
        raise ValueError(f"dataset metadata for {dataset_dir} is missing shard counts: {dataset_entry}")
    if train_shards > max_train_shards:
        raise ValueError(
            f"{args.variant} only has {max_train_shards} training shards on {REPO_ID}, requested {train_shards}"
        )
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = resolve_tokenizer_entry(manifest, tokenizer_name)
    if tokenizer_entry is None:
        raise ValueError(
            f"tokenizer {tokenizer_name} not found in {REMOTE_ROOT_PREFIX}/manifest.json or built-in fallback metadata"
        )

    if args.with_docs:
        get(f"{REMOTE_ROOT_PREFIX}/docs_selected.jsonl")
        get(f"{REMOTE_ROOT_PREFIX}/docs_selected.source_manifest.json")

    dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{dataset_dir}"
    for i in range(val_shards):
        get(f"{dataset_prefix}/fineweb_val_{i:06d}.bin")
    for i in range(train_shards):
        get(f"{dataset_prefix}/fineweb_train_{i:06d}.bin")

    for artifact_path in artifact_paths_for_tokenizer(tokenizer_entry):
        get(f"{REMOTE_ROOT_PREFIX}/{artifact_path}")


if __name__ == "__main__":
    main()
