from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression-check shared SentencePiece BPB byte accounting")
    parser.add_argument(
        "--tokenizer",
        default="data/tokenizers/fineweb_1024_bpe.model",
        help="SentencePiece model to validate. Defaults to the local public SP1024 tokenizer.",
    )
    return parser.parse_args()


def choose_boundary_token_id(sp, vocab_size: int) -> int:
    for token_id in range(vocab_size):
        if sp.id_to_piece(token_id).startswith("▁"):
            return token_id
    raise ValueError("No boundary token found in SentencePiece vocabulary")


def main() -> None:
    args = parse_args()
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer model: {tokenizer_path}")

    try:
        import sentencepiece as spm
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "check_sentencepiece_bpb.py requires both sentencepiece and torch. "
            "Install the active runtime with `pip install -r requirements-h100-fla.txt` or run it inside the prepared H100 environment."
        ) from exc

    from frontier_gdn.byte_scoring import build_sentencepiece_luts, token_byte_counts

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    vocab_size = sp.vocab_size()
    device = torch.device("cpu")
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, vocab_size, device)
    boundary_token_id = choose_boundary_token_id(sp, vocab_size)
    samples = [
        "hello world",
        "two  spaces here",
        "punctuation, works fine.",
        "UTF-8 cafe and naive bytes",
        "line one\nline two",
    ]

    for sample in samples:
        ids = sp.encode(sample)
        if not ids:
            continue
        prev_ids = [boundary_token_id] + ids[:-1]
        tgt = torch.tensor(ids, dtype=torch.long, device=device)
        prev = torch.tensor(prev_ids, dtype=torch.long, device=device)
        counted = int(
            token_byte_counts(
                tgt,
                prev,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            ).sum().item()
        )
        expected = len(sample.encode("utf-8"))
        if counted != expected:
            raise SystemExit(
                f"FAIL: byte-count mismatch for {sample!r}: helper counted {counted}, expected {expected}"
            )
        print(f"PASS: {sample!r} -> {counted} bytes")

    print("status: PASS")


if __name__ == "__main__":
    main()
