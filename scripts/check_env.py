from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_config(path: str | None) -> dict[str, object]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local environment for parameter-golf workflows")
    parser.add_argument("--config", help="Optional JSON config to validate", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    env = dict(config.get("env", {}))
    report: dict[str, object] = {
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "config": args.config,
        "torch_importable": False,
        "cuda_available": False,
    }
    try:
        import torch  # type: ignore

        report["torch_importable"] = True
        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["device_name"] = torch.cuda.get_device_name(0)
            report["device_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:  # pragma: no cover - informational
        report["torch_error"] = str(exc)

    try:
        report["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError as exc:  # pragma: no cover - informational
        report["nvidia_smi_error"] = str(exc)

    data_path = Path(env.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024"))
    tokenizer_path = Path(env.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"))
    report["data_path_exists"] = data_path.exists()
    report["tokenizer_path_exists"] = tokenizer_path.exists()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

