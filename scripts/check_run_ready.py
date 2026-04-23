from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def infer_vocab_family(path_str: str) -> int | None:
    normalized = path_str.replace("\\", "/")
    patterns = (r"sp(\d+)", r"fineweb_(\d+)_bpe\.model", r"fineweb_(\d+)_bpe\.vocab")
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))
    return None


def resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local inputs for a parameter-golf run config")
    parser.add_argument("config", help="Path to a JSON config file")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    env = {str(k): str(v) for k, v in dict(config.get("env", {})).items()}

    data_path = resolve_path(root, env["DATA_PATH"])
    tokenizer_path = resolve_path(root, env["TOKENIZER_PATH"])
    vocab_size = int(env["VOCAB_SIZE"])

    data_vocab = infer_vocab_family(str(data_path))
    tokenizer_vocab = infer_vocab_family(str(tokenizer_path))
    failures: list[str] = []

    if data_vocab is not None and data_vocab != vocab_size:
        failures.append(f"DATA_PATH implies sp{data_vocab} but VOCAB_SIZE={vocab_size}")
    if tokenizer_vocab is not None and tokenizer_vocab != vocab_size:
        failures.append(f"TOKENIZER_PATH implies sp{tokenizer_vocab} but VOCAB_SIZE={vocab_size}")
    if data_vocab is not None and tokenizer_vocab is not None and data_vocab != tokenizer_vocab:
        failures.append(f"DATA_PATH implies sp{data_vocab} but TOKENIZER_PATH implies sp{tokenizer_vocab}")
    if not tokenizer_path.is_file():
        failures.append(f"Missing tokenizer: {tokenizer_path}")

    train_files = sorted(data_path.glob("fineweb_train_*.bin"))
    val_files = sorted(data_path.glob("fineweb_val_*.bin"))
    if not train_files:
        failures.append(f"No training shards under {data_path}")
    if not val_files:
        failures.append(f"No validation shards under {data_path}")

    print(f"config: {config_path}")
    print(f"data_path: {data_path}")
    print(f"tokenizer_path: {tokenizer_path}")
    print(f"vocab_size: {vocab_size}")
    print(f"train_shards_found: {len(train_files)}")
    print(f"val_shards_found: {len(val_files)}")

    if failures:
        print("status: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("status: PASS")


if __name__ == "__main__":
    main()
