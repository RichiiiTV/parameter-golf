from __future__ import annotations

import argparse
import ast
import base64
import importlib
import json
import lzma
from pathlib import Path


REQUIRED_PAYLOAD_SNIPPETS = (
    "Path(__file__).read_text(encoding='utf-8')",
    "bytes_total>16_000_000",
    "flash_attn_interface",
    "SentencePieceProcessor",
)

H100_IMPORTS = (
    "brotli",
    "sentencepiece",
    "torch",
    "flash_attn_interface",
)


def resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (root / path).resolve()


def extract_payload(script_path: Path) -> str:
    source = script_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(script_path))
    candidates = [node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    payload = max(candidates, key=len, default="")
    if len(payload) < 1000:
        raise ValueError(f"Could not find compressed payload in {script_path}")
    raw = lzma.decompress(
        base64.b85decode(payload),
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2}],
    )
    text = raw.decode("utf-8")
    compile(text, f"<{script_path}:payload>", "exec")
    return text


def check_config_alignment(root: Path, config_path: Path, payload: str) -> list[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    env = {str(k): str(v) for k, v in dict(config.get("env", {})).items()}
    failures: list[str] = []
    vocab_size = env.get("VOCAB_SIZE")
    data_dir = env.get("DATA_DIR")
    data_path = env.get("DATA_PATH")
    tokenizer_path = env.get("TOKENIZER_PATH")

    if "DATA_DIR" not in payload or "datasets_dir=os.path.join(data_dir,'datasets'" not in payload:
        failures.append("payload no longer derives dataset/tokenizer paths from DATA_DIR")
    if not data_dir:
        failures.append("config env must set DATA_DIR explicitly for the active #1493 payload")
    if vocab_size and data_dir and data_path:
        derived_data_path = resolve_path(root, str(Path(data_dir) / "datasets" / f"fineweb10B_sp{vocab_size}"))
        configured_data_path = resolve_path(root, data_path)
        if derived_data_path != configured_data_path:
            failures.append(f"DATA_DIR derives {derived_data_path}, but DATA_PATH is {configured_data_path}")
    if vocab_size and data_dir and tokenizer_path:
        derived_tokenizer_path = resolve_path(root, str(Path(data_dir) / "tokenizers" / f"fineweb_{vocab_size}_bpe.model"))
        configured_tokenizer_path = resolve_path(root, tokenizer_path)
        if derived_tokenizer_path != configured_tokenizer_path:
            failures.append(f"DATA_DIR derives {derived_tokenizer_path}, but TOKENIZER_PATH is {configured_tokenizer_path}")
    return failures


def check_imports() -> tuple[list[str], list[str]]:
    available = []
    failures = []
    for module_name in H100_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name} ({exc.__class__.__name__}: {exc})")
        else:
            available.append(module_name)
    return available, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Static preflight for the compressed active train_gpt.py payload")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/h100/root_sp8192_pr1493_accepted_8xh100.json",
        help="H100 JSON config to check against the active payload",
    )
    parser.add_argument("--script", default="train_gpt.py", help="Compressed root training script to inspect")
    parser.add_argument("--require-imports", action="store_true", help="Fail unless H100 runtime imports are installed")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    script_path = resolve_path(root, args.script)
    config_path = resolve_path(root, args.config)
    payload = extract_payload(script_path)

    failures = [f"missing payload marker: {snippet}" for snippet in REQUIRED_PAYLOAD_SNIPPETS if snippet not in payload]
    failures.extend(check_config_alignment(root, config_path, payload))

    available_imports, import_failures = check_imports()
    if args.require_imports and import_failures:
        failures.append(f"H100 runtime import failures: {', '.join(import_failures)}")

    print(f"script: {script_path}")
    print(f"config: {config_path}")
    print(f"payload_lines: {len(payload.splitlines())}")
    print(f"payload_bytes: {len(payload.encode('utf-8'))}")
    print(f"h100_imports_available: {', '.join(available_imports) or 'none'}")
    if import_failures:
        print(f"h100_import_failures: {', '.join(import_failures)}")

    if failures:
        print("status: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("status: PASS")


if __name__ == "__main__":
    main()
