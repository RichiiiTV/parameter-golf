from __future__ import annotations

import sys
from pathlib import Path


LIMIT = 1500
ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "train_gpt.py", ROOT / "train_gpt_mlx.py"]


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def main() -> None:
    failures = []
    for path in TARGETS:
        count = line_count(path)
        print(f"{path.name}: {count} lines")
        if count >= LIMIT:
            failures.append(f"{path.name} has {count} lines; limit is {LIMIT - 1}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
