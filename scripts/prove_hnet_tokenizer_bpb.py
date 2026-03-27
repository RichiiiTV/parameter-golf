from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.hnet_tokenizer import HNetChunkTokenizer
try:
    import sentencepiece as spm
except ImportError:
    spm = None


def load_data_shard(file: Path) -> np.ndarray:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    return np.fromfile(file, dtype="<u2", count=int(header[2]), offset=256 * np.dtype("<i4").itemsize)


def iter_doc_bytes(path: Path, num_docs: int):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_docs:
                break
            yield json.loads(line)["text"].encode("utf-8", errors="replace")


def iter_doc_tokens(pattern: str, bos_id: int):
    started = False
    current: list[int] = []
    for file in sorted(glob.glob(pattern)):
        for tok in load_data_shard(Path(file)):
            token_id = int(tok)
            if token_id == bos_id:
                if started:
                    yield current
                current = []
                started = True
            elif started:
                current.append(token_id)
    if started:
        yield current


def count_bytes_from_luts(pattern: str, base: np.ndarray, leading: np.ndarray, boundary: np.ndarray) -> int:
    total = 0
    prev = None
    for file in sorted(glob.glob(pattern)):
        for tok in load_data_shard(Path(file)):
            token_id = int(tok)
            if prev is not None:
                total += int(base[token_id]) + int(bool(leading[token_id]) and not bool(boundary[prev]))
            prev = token_id
    return total


def build_sentencepiece_luts(sp: Any, vocab_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.zeros((vocab_size,), dtype=np.int16)
    leading = np.zeros((vocab_size,), dtype=np.bool_)
    boundary = np.ones((vocab_size,), dtype=np.bool_)
    for token_id in range(int(sp.vocab_size())):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        boundary[token_id] = False
        if sp.is_byte(token_id):
            base[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            leading[token_id] = True
            piece = piece[1:]
        base[token_id] = len(piece.encode("utf-8"))
    return base, leading, boundary


def prove_hnet(args: argparse.Namespace) -> None:
    tok = HNetChunkTokenizer.from_json(args.tokenizer_path)
    base, _, _ = tok.build_byte_luts()
    raw_total = decoded_total = lut_total = 0
    for doc_idx, (raw_bytes, token_ids) in enumerate(zip(iter_doc_bytes(args.docs_jsonl, args.num_val_docs), iter_doc_tokens(args.val_glob, tok.bos_id), strict=True), start=1):
        decoded = tok.decode_token_ids(token_ids)
        if decoded != raw_bytes:
            raise ValueError(f"H-tokenizer decode mismatch at val doc {doc_idx}")
        raw_total += len(raw_bytes)
        decoded_total += len(decoded)
        lut_total += int(base[np.asarray(token_ids, dtype=np.int64)].sum()) if token_ids else 0
    if raw_total != decoded_total or raw_total != lut_total:
        raise ValueError(f"H-tokenizer byte mismatch raw={raw_total} decoded={decoded_total} lut={lut_total}")
    print(f"hnet_ok docs={args.num_val_docs} raw_bytes={raw_total} decoded_bytes={decoded_total} lut_bytes={lut_total}")


def prove_sentencepiece(args: argparse.Namespace) -> None:
    if spm is None:
        raise RuntimeError("sentencepiece is required for the stock tokenizer parity check")
    sp = spm.SentencePieceProcessor(model_file=str(args.stock_sp_model))
    base, leading, boundary = build_sentencepiece_luts(sp, int(sp.vocab_size()))
    raw_total = decoded_total = 0
    bos_id = int(sp.bos_id())
    for doc_idx, (raw_bytes, token_ids) in enumerate(zip(iter_doc_bytes(args.docs_jsonl, args.num_val_docs), iter_doc_tokens(args.stock_sp_val_glob, bos_id), strict=True), start=1):
        decoded = sp.decode(token_ids).encode("utf-8")
        if decoded != raw_bytes:
            raise ValueError(f"SentencePiece decode mismatch at val doc {doc_idx}")
        raw_total += len(raw_bytes)
        decoded_total += len(decoded)
    lut_total = count_bytes_from_luts(args.stock_sp_val_glob, base, leading, boundary)
    if raw_total != decoded_total or raw_total != lut_total:
        raise ValueError(f"SentencePiece byte mismatch raw={raw_total} decoded={decoded_total} lut={lut_total}")
    print(f"sentencepiece_ok docs={args.num_val_docs} raw_bytes={raw_total} decoded_bytes={decoded_total} lut_bytes={lut_total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prove exact byte accounting for the H-tokenizer lane")
    parser.add_argument("--docs-jsonl", type=Path, required=True)
    parser.add_argument("--val-glob", required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--num-val-docs", type=int, default=50_000)
    parser.add_argument("--stock-sp-model", type=Path)
    parser.add_argument("--stock-sp-val-glob")
    args = parser.parse_args()
    prove_hnet(args)
    if args.stock_sp_model and args.stock_sp_val_glob:
        prove_sentencepiece(args)


if __name__ == "__main__":
    main()
