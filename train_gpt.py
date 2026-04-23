#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 8192))
DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
ARCH_MODE = os.environ.get("ARCH_MODE", "K")
os.environ.setdefault("VOCAB_SIZE", str(VOCAB_SIZE))
os.environ.setdefault("DATA_PATH", DATA_PATH)
os.environ.setdefault("TOKENIZER_PATH", TOKENIZER_PATH)
os.environ.setdefault("ARCH_MODE", ARCH_MODE)
os.environ.setdefault("MAX_WALLCLOCK_SECONDS", "600")
os.environ.setdefault("VAL_LOSS_EVERY", "0")
os.environ.setdefault("EVAL_COMPILE_ENABLED", "0")

_REQUIRED_PKGS = [
    "triton==3.2.0",
    "flash-linear-attention==0.4.2",
    "fla-core==0.4.2",
    "transformers==5.5.4",
    "tokenizers==0.22.2",
    "safetensors==0.7.0",
    "zstandard",
]


def _ensure_fla_runtime_available() -> None:
    try:
        from fla.layers.gated_deltanet import GatedDeltaNet  # noqa: F401
    except Exception as exc:
        packages = ", ".join(_REQUIRED_PKGS)
        raise RuntimeError(
            "Missing FLA frontier runtime dependencies. "
            "Install them with `pip install -r requirements-h100-fla.txt` before running the SP8192 lane. "
            f"Required packages: {packages}"
        ) from exc


def main() -> None:
    _ensure_fla_runtime_available()
    from frontier_gdn.train_gdn_7k import main as frontier_main

    frontier_main()


if __name__ == "__main__":
    main()
